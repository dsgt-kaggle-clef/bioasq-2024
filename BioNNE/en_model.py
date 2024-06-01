'''Model for English nested NER'''
import argparse
import os
import json
import re
from pathlib import Path
from pprint import pprint
import requests
import time
import datetime
from tokens import * 
import textsynth_api as llm
import helper_utils as utils
import numpy as np
import mocker
import config
import db_utils
from umls_module import UMLSModule
from spacy_lm_module import SpacyLMModule
from doc_ner_module import DocNER, COUNTER
ENTITY_LIST_TO_TRY = config.ENTITY_LIST
# ENTITY_LIST_TO_TRY = ['FINDING']

def run_model(save_prediction):
    '''
    recognize selected label (i.e FINDING, CHEM) in the text
    1. use 2 fixed examples 
    1-1 alternative is to use the closest examples
    2. test 5 dev examples
    3. calculate the accuracy, after removing duplicate and non-existent entities
    '''
    llm_log_file = open('logs/llm_nne_history.txt', "a")
    train_path = f'processed_data/bionne/en/train.json'
    dev_path = f'processed_data/bionne/en/dev.json'
    test_path = f'processed_data/bionne/en/test.json'
    start_time = datetime.datetime.now()
    print('Config:', config.LLM_MODEL, 'fake_llm:', config.FAKE_LLM, 'seed:', config.LLM_SEED, 'add_instruction:', config.ADD_INSTRUCTION)
    cui_db, semantic_db, nonentity_db = db_utils.open_db()
    umls_cls = UMLSModule(cui_db, semantic_db, nonentity_db)

    with open(train_path, "r", encoding='utf-8') as read_file:
        train_dict = json.load(read_file)

    if config.RUN_MODE == 'test':
        with open(test_path, "r", encoding='utf-8') as read_file:
            test_dict = json.load(read_file)
    else:
        with open(dev_path, "r", encoding='utf-8') as read_file:
            dev_dict = json.load(read_file)

    with open(config.MOCK_PATH, "r", encoding='utf-8') as read_file:
        llm_response = json.load(read_file)
    cur_time = int(time.time())
    prediction_path = f'output/{cur_time}_en.jsonl'
    if save_prediction:
        print('Prediction will save to:', prediction_path)
        prediction_file = open(prediction_path, 'w')
        llm_answer_file = open(f'output/{cur_time}_llm_answer.jsonl', 'w')

    train_cnt = 0
    example_split_counter = {}
    for entity in config.ENTITY_LIST:
        example_split_counter[entity] = 0

    example_split = [] # The list of examples that will be used for few-shot learning
    rule_split = [] # The list of examples that will be used to compile production rules
    for sample in train_dict:
        train_cnt += 1
        if train_cnt%2 == 0:
            for entity in config.ENTITY_LIST:
                if entity in train_dict[sample]:
                    example_split_counter[entity] += 1
            example_split.append(sample)
        else:
            rule_split.append(sample)
    print('Example Split Counter:', example_split_counter)

    if config.RUN_MODE == 'train':
        run_set_dict = train_dict
        run_split = rule_split[0:8]
        print('Dataset:', 'Training rule set')
    elif config.RUN_MODE == 'validate':
        run_set_dict = dev_dict
        run_split = list(dev_dict.keys())
        print('Dataset:', 'Validation set')
    elif config.RUN_MODE == 'test':
        run_set_dict = test_dict
        run_split = list(test_dict.keys())[147:]
        print('Dataset:', 'Test set')

    total = len(run_split)
    print('Total record count', total)

    measurement_cnt = {}
    for tag in config.ENTITY_LIST:
        measurement_cnt[tag] = []
    
    accuracy_dict = {}
    for tag in config.ENTITY_LIST:
        accuracy_dict[tag] = {}
        for category in ['tp', 'fp', 'fn']:
            accuracy_dict[tag][category] = 0.0

    for index, rec in enumerate(run_split):
        # time.sleep(1)
        llm_ans_dict = {}
        llm_ans_dict['id'] = rec
        text = run_set_dict[rec]['text']
        text_lower = text.lower()
        spacy_lm = SpacyLMModule(text, umls_cls)
        doc_ner = DocNER(text)
        print(f'\n({index}/{total}) Processing:', rec, '\n')
        # Step 1. Find acronyms and their category
        abrv_list = spacy_lm.get_abbrev()
        print('\nAbbreviations: ')
        for doc_ent in abrv_list:
            print(doc_ent)
            doc_ner.add_word(doc_ent)
        
        # Step 2. Find the closest examples and perform LLM NER recognition
        llm_results = {}
        for selected_tag in ENTITY_LIST_TO_TRY:
            # Construct example prompt
            examples = utils.get_closest_examples(train_dict, example_split, selected_tag, text)
            example_prompt = utils.create_few_shot_prompt(train_dict, examples, selected_tag)
            if config.VERBOSE:
                print('Example Prompt: ', example_prompt)

            if not text.endswith('\n'):
                text += '\n'
            dev_prompt = f'[TEXT]: {text}\n[{selected_tag}]: '''
            final_prompt = example_prompt + dev_prompt
            if config.VERBOSE:
                print('Final Prompt: ')
                print(final_prompt)
                print('*'*80)
            llm_query = {
                'prompt': final_prompt,
                'max_tokens': 200,
                'stop': '###',
                'n': 2,
                'seed': config.LLM_SEED
            }
            if config.FAKE_LLM:
                response = ''
                if (rec in llm_response) and (selected_tag in llm_response[rec]):
                    response = llm_response[rec][selected_tag]
            else:
                resp = llm.make_textsynth_request(f'/v1/engines/{config.LLM_MODEL}/completions', llm_query, llm_log_file)
                response = resp['text']
                llm_ans_dict[selected_tag] = response

            print(f'\nq: {rec}', selected_tag, text[:80], '...')
            print(f'llm answer: {response}')
            llm_pred_entities = spacy_lm.process_llm_output(response)

            for entry in llm_pred_entities:
                span, mentions = llm_pred_entities[entry]['span'], llm_pred_entities[entry]['mentions']
                if entry not in doc_ner.name_table:
                    doc_ent = spacy_lm.initialize_basic_doc_ent(entry, span)
                    doc_ner.add_word(doc_ent)
                doc_ner.append_llm_ner_tag(entry, selected_tag, mentions)
            print('-'*80)

        
        # Step 2: get DISO (disease or cancer) entities from Spacy BioMed models
        ent_list = spacy_lm.get_spy_ner_entities()
        for entity in ent_list:
            if entity not in doc_ner.name_table:
                doc_ent = spacy_lm.initialize_basic_doc_ent(entity, ent_list[entity]['span'])
                doc_ner.add_word(doc_ent)
            doc_ner.append_spy_ner_tag(entity, ent_list[entity]['label'])

        # Step 3. Link recognized entities to UMLS
        doc_ner.populate_umls_tags(umls_cls)

        # Step 4. Use all information to classify the entities
        doc_ner.populate_final_tag()

        print()
        print('## Result: ' + '\n')

        doc_ner.print_name_table()
        doc_ner.print_tag_table()

        brat_cnt = 0
        brat_output = ''
        pred_loc_list = []

        if config.RUN_MODE != 'test':
            print()
            print('## Verification: ' + '\n')
        for selected_tag in config.ENTITY_LIST:
            if config.RUN_MODE == 'test':
                actual_entities = doc_ner.get_tagged_entities(selected_tag)
            else:
                if selected_tag in run_set_dict[rec]:
                    expected_entities = run_set_dict[rec][selected_tag]
                else:
                    expected_entities = []
                actual_entities = doc_ner.get_tagged_entities(selected_tag)
                print('-'*80)
                print('Entity:', selected_tag)
                print('Expect List: ', sorted(expected_entities, key=lambda x: x[0]))
                print()
                print('Actual List: ', sorted(actual_entities, key=lambda x: x[0]))
                # Calculate the accuracy for each category
                e_tp, e_fp, e_fn = utils.cnt_tp_fp_fn(actual_entities, expected_entities)
                accuracy_dict[selected_tag]['tp'] += e_tp
                accuracy_dict[selected_tag]['fp'] += e_fp
                accuracy_dict[selected_tag]['fn'] += e_fn
                precision, recall, f1 = utils.calculate_accuracy(actual_entities, expected_entities)
                if config.VERBOSE:
                    utils.compare_entities(actual_entities, expected_entities)
                measurement_cnt[selected_tag].append((precision, recall, f1))

            # Find the locations of the entity in the text
            for entity in actual_entities:
                res_loc = utils.find_all_loc_in_text(entity, text)
                if res_loc:
                    for loc_pair in res_loc:
                        brat_cnt += 1
                        text_clip = text[loc_pair[0]:loc_pair[1]]
                        # print(f"T{brat_cnt}\t{loc_pair[0]}\t{loc_pair[1]}\t{text_clip}\t{entity}")
                        brat_output += f'{selected_tag}\t{loc_pair[0]}\t{loc_pair[1]}\t{text_clip}\n'
                        pred_loc_list.append([loc_pair[0], loc_pair[1], selected_tag])

        if save_prediction:
            prediction_dict = {}
            prediction_dict['id'] = rec # TODO: move id after entity
            prediction_dict['entity'] = pred_loc_list
            prediction_dict['text'] = text
            # Write prediction_dict as a json line to jsonl file
            prediction_file.write(json.dumps(prediction_dict))
            prediction_file.write('\n')
            llm_answer_file.write(json.dumps(llm_ans_dict))
            llm_answer_file.write('\n')

    if config.RUN_MODE != 'test':
        print('-'*80)
        print('\n## Metrics\n')
        print('### Per record accuracy:')
        pprint(measurement_cnt)
        prec_list = []
        rec_list = []
        f1_list = []
        # Aggregate per-class accuracy
        print('\n### Per-class average score:')
        for entity in config.ENTITY_LIST:
            avg_pr, avg_rec, avg_f1 = utils.calculate_precision_recall_f1(accuracy_dict[entity]['tp'], accuracy_dict[entity]['fp'], accuracy_dict[entity]['fn'], config.VERBOSE)
            print(f'{entity} - Precision: {avg_pr:.4f}, Recall: {avg_rec:.4f}, F1: {avg_f1:.4f}')
            prec_list.append(avg_pr)
            rec_list.append(avg_rec)
            f1_list.append(avg_f1)

        # Average per-class accuracy
        print('### Total average score:')
        precision_np = np.array(prec_list)
        recall_np = np.array(rec_list)
        f1_np = np.array(f1_list)
        avg_pr = np.mean(precision_np)
        avg_rec = np.mean(recall_np)
        avg_f1 = np.mean(f1_np)
        print(f'Precision: {avg_pr:.4f}\nRecall: {avg_rec:.4f}\nF1: {avg_f1:.4f}')

    # Save the db content on disk
    db_utils.close_db([cui_db, semantic_db, nonentity_db])

    print('\n## Output\n')
    print(f'Start time: {start_time}')
    print(f'End time: {datetime.datetime.now()}')
    if save_prediction:
        print(f'Prediction saved to {prediction_path}')
    
    print('Total record processed', total)
    print('Counter:', COUNTER)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BioNNE model runner')
    parser.add_argument('--save-prediction', action='store_true', help="Save the prediction to a file")
    args = parser.parse_args()
    run_model(args.save_prediction)

