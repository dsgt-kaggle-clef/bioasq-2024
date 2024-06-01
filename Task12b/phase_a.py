import argparse
import datetime
import json
import re
import time
import torch
import search_utils
import textsynth_api as llm
import helper_utils as helper
import metrics
import config
import qa_module
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer, util
from prompt_utils import Question, prompt_config, keyword_extact_prompt
import model


# a_setup
def a_setup(input_filename):
    pass
    # Set up a folder for the run. All subsequent steps of phase a will be saved in this folder.
    curr_folder = f'output/{input_filename}'
    # Create the folder
    Path(curr_folder).mkdir(parents=True, exist_ok=True)

    # Save the used config in the run folder
    with open(f'{curr_folder}/config.json', 'w') as f:
        config_obj = {
            'SEARCH_WORD_MODE': config.SEARCH_WORD_MODE,
            'OPENAI_MODEL_NAME': config.OPENAI_MODEL_NAME,
            'LLM_MODEL': config.LLM_MODEL,
            'SNIP_EXTRACT_MODE': config.SNIP_EXTRACT_MODE,
            'PUBLIC_MAX_LENGTH': config.PUBMED_MAX_LENGTH,
        }
        json.dump(config_obj, f)

    # read inputs and return
    input_path = f'input/{input_filename}.json'
    questions = helper.parse_input(input_path)
    return questions, curr_folder

def get_ids_from_jsonl_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    json_lst = [json.loads(line) for line in lines]
    ids = set([j['id'] for j in json_lst])
    return ids

def one_query(ques_dict, spacy_model=None, mode='LLM'):
    ques_obj = helper.init_ques(ques_dict, stage='A')

    qbody = helper.sanity_question(ques_obj.body)
    if (qbody != ques_obj.body):
        print(f'Found malformed question, rewrite it as {qbody}')

    ir_log_obj = {'id': ques_obj.id, 'body': qbody, 'type': ques_obj.type, 'mode': config.SEARCH_WORD_MODE, 'outputs': []}
    if mode == 'SPACY':
        named_entity_list = search_utils.extract_entities(spacy_model, qbody)
        keywords = [t[0] for t in named_entity_list]
        ir_log_obj['outputs'].append(keywords)
    elif mode == 'LLM':
        keywords = model.llm_get_keywords(config.LLM_MODEL, qbody, ques_obj.id)
        ir_log_obj['outputs'].append(keywords)
    elif mode == 'MIXTRAL_47B':
        query_term = llm.query_expansion(qbody, config.FAKE_LLM)
        ir_log_obj['outputs'].append(query_term)
    elif mode == 'OPENAI':
        query_term = llm.query_expansion(qbody, config.FAKE_LLM, model_name='OPENAI')
        ir_log_obj['outputs'].append(query_term)
        pmid_list = search_utils.get_pmids(query_term, verbose=config.VERBOSE) # keeping this to allow for retries of the query
        # attempt retry when pmid_list has less than 3 results
        retry_count = 0
        while len(pmid_list) < 3:
            if retry_count == 0:
                new_query_term = llm.query_expansion_retry(qbody, query_term, config.FAKE_LLM)
                ir_log_obj['outputs'].append(new_query_term)
                print(f'Retrying with Keywords: {new_query_term}')

                pmid_list = search_utils.get_pmids(new_query_term, verbose=config.VERBOSE) # keeping this to allow for retries of the query
                query_term = new_query_term
                retry_count += 1
            else:
                # last try
                query_term = llm.query_expansion_last_try(qbody, config.FAKE_LLM)
                ir_log_obj['outputs'].append(query_term)
                print(f'Last try with Keywords: {query_term}')

                break
    else:
        assert(not 'Unrecognized search mode')

    return ir_log_obj


# a_query
def a_query(curr_folder, questions, spacy_model=None, query_file='query'):
    print('-'*50)
    print('Query Processing')
    print('-'*50)

    # Check if current folder contain the query file. If not, create it.
    query_log_path = f'{curr_folder}/{query_file}.jsonl'
    if not Path(query_log_path).exists():
        Path(query_log_path).touch()
    # read the query file
    ids = get_ids_from_jsonl_log(query_log_path)
    # filter out questions that already have queries
    if ids:
        print(f'Found {len(ids)} existing queries')
    questions = [q for q in questions if q['id'] not in ids]
    print(f'Processing {len(questions)} questions')

    # build the query given questions. If query exists for a question, skip it.
    total_rec = len(questions)
    for proc_rec, ques_dict in enumerate(questions):
        # if query exists, skip
        if ques_dict['id'] not in ids:
            
            proc_rec += 1
            if config.INFO_TRACE:
                print(f'({proc_rec}/{total_rec}) Processing {ques_dict["id"]} - {ques_dict["body"]}')

            ir_log_obj= one_query(ques_dict, spacy_model, mode=config.SEARCH_WORD_MODE)
            model.append_log(query_log_path, ir_log_obj)

    # open the query file and read the queries
    with open(query_log_path, 'r') as f:
        lines = f.readlines()
    questions = [json.loads(line) for line in lines]

    return questions
    
def one_ir(question, pmid_list, st_model):
    # Get embedding for qbody
    question_embedding = st_model.encode(question['body'])

    # Get the abstracts and embeddings
    articles = search_utils.query_by_pmids(pmid_list, verbose=config.VERBOSE)

    articles_embeddings = st_model.encode([article['abstract_raw'] for article in articles])
    similarity_scores = util.pytorch_cos_sim(question_embedding, articles_embeddings)
    # Sort the scores in descending order along with their indices
    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
    # Flatten the sorted indices to a 1D tensor since your query seems to be for a single question
    sorted_indices = sorted_indices.flatten()
    # Reorder the articles based on the sorted indices
    sorted_articles = [articles[i] for i in sorted_indices]

    # create snippets
    snippets = model.extract_snippets(sorted_articles, question['body'], st_model)

    # get pmids
    sorted_pmids = [article['pmid'] for article in sorted_articles]

    return sorted_pmids, snippets
    
# a_ir
def a_ir(questions_with_queries, curr_folder, st_model):
    print('-'*50)
    print('IR Processing')
    print('-'*50)
    # Check if current folder contain the ir file. If not, create it.
    ir_log_path = f'{curr_folder}/ir.jsonl'
    if not Path(ir_log_path).exists():
        Path(ir_log_path).touch()

    difficult_questions_log_path = f'{curr_folder}/difficult_questions.jsonl'
    if not Path(difficult_questions_log_path).exists():
        Path(difficult_questions_log_path).touch()

    ids = get_ids_from_jsonl_log(ir_log_path)
    if ids:
        print(f'Found {len(ids)} existing IR results')

    # filter out questions that already have queries
    questions_with_queries = [q for q in questions_with_queries if q['id'] not in ids]
    print(f'Processing IR for {len(questions_with_queries)} questions')

    for idx, question in enumerate(questions_with_queries):
        print(f'IR Processing {idx+1}/{len(questions_with_queries)}')
        # Query from Pubmed
        if question['mode'] == 'SPACY' or question['mode'] == 'LLM':
            keywords = question['outputs'][-1] # get the last try's keywords
            pmid_list = search_utils.query_by_keywords(keywords, verbose=config.VERBOSE)
        elif question['mode'] == 'MIXTRAL_47B' or question['mode'] == 'OPENAI':
            query_term = question['outputs'][-1] # get the last try's query
            pmid_list = search_utils.get_pmids(query_term, verbose=config.VERBOSE)

        if len(pmid_list) == 0:
            print(f'***No articles found for question {question["body"]} - {question["id"]}')
            # save in a file
            question['pubmed_resutls'] = []
            model.append_log(difficult_questions_log_path, question)
            continue
        else:
            sorted_pmids, snippets = one_ir(question, pmid_list, st_model)

            # save the results
            ir_log_obj = {'id': question['id'], 'body': question['body'], 'type': question['type'], 'documents': sorted_pmids, 'snippets': snippets}
            model.append_log(ir_log_path, ir_log_obj)

    return


def prepare_phase_a_outputs(curr_folder, input_filename):
    # read the ir file
    with open(f'{curr_folder}/ir.jsonl', 'r') as f:
        lines = f.readlines()
    ir_results = [json.loads(line) for line in lines]

    # read the difficult questions ir results
    with open(f'{curr_folder}/difficult_ir.jsonl', 'r') as f:
        lines = f.readlines()
    difficult_ir_results = [json.loads(line) for line in lines]


    phase_a_outputfile = f'{curr_folder}/phase_a_output.json'
    obj = {'questions': ir_results + difficult_ir_results}
    with open(phase_a_outputfile, 'w') as f:
        json.dump(obj, f)

    return

def retry_difficult_questions(curr_folder, st_model):
    # read the difficult questions
    difficult_questions_log_path = f'{curr_folder}/difficult_questions.jsonl'
    difficult_ir_output = f'{curr_folder}/difficult_ir.jsonl'

    if Path(difficult_questions_log_path).exists() and not Path(difficult_ir_output).exists(): 
        with open(difficult_questions_log_path, 'r') as f:
            lines = f.readlines()
        difficult_questions = [json.loads(line) for line in lines]

        Path(difficult_ir_output).touch()
    
        # run query on difficult questions in LLM mode
        for idx, question in enumerate(difficult_questions):
            print(f'Retrying difficult question {idx+1}/{len(difficult_questions)}: {question["body"]} - {question["id"]}')
            ir_log_obj= one_query(question, mode='LLM')
            
            # run IR on difficult questions
            keywords = ir_log_obj['outputs'][-1]
            pmid_list = search_utils.query_by_keywords(keywords, verbose=config.VERBOSE)

            if len(pmid_list) == 0:
                print(f'***No articles found for question {question["body"]} - {question["id"]}')
                sorted_pmids = []
                snippets = []
            else:
                sorted_pmids, snippets = one_ir(question, pmid_list, st_model)

            # save the results
            ir_log_obj = {'id': question['id'], 'body': question['body'], 'type': question['type'], 'documents': sorted_pmids, 'snippets': snippets}
            model.append_log(difficult_ir_output, ir_log_obj)

    return

        
# runner
def main():
    input_filename = '12B3PhaseA'   
    questions, curr_folder = a_setup(input_filename)

    if config.SEARCH_WORD_MODE == 'SPACY':
        spacy_model = search_utils.load_spacy_model()
    else:
        spacy_model = None
    questions_with_queries = a_query(curr_folder, questions, spacy_model)

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    a_ir(questions_with_queries, curr_folder, st_model)

    retry_difficult_questions(curr_folder, st_model)

    prepare_phase_a_outputs(curr_folder, input_filename)
    
    return

if __name__ == '__main__':
    main()