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

# -----------------------------------------------------------------------------
# LLM Question keyword extractor methods
# -----------------------------------------------------------------------------
def llm_get_keywords(cur_model, qbody, qid):
    if config.FAKE_LLM:
        if qid == '55031181e9bde69634000014':
            return ['Hirschsprung disease', 'mendelian', 'multifactorial disorder']
        elif qid == '55046d5ff8aee20f27000007':
            return ['EGFR', 'signaling molecule', 'ligands']
        elif qid == '54e25eaaae9738404b000017':
            return ['Papilin', 'protein', 'secreted']
        elif qid == '56bc751eac7ad10019000013':
            return ['Acrokeratosis paraneoplastica']
    else:
        prompt = keyword_extact_prompt.format(body=qbody)
        qbody_lower = qbody.lower()
        req = {
            'prompt': prompt,
            'max_tokens': 200,
            'stop': '###'
        }
        for cnt in range(2):
            res = llm_get_keywords_subroutine(cur_model, qbody_lower, req)
            if len(res) > 0:
                return res
        return []

def llm_get_keywords_subroutine(cur_model, qbody_lower, req):
    result = llm.textsynth_completion(cur_model, req, 'IR', config.FAKE_LLM)
    if config.VERBOSE:
        print(result)
        print("\n" + "-"*80 + "\n")
    response = result['text']
    raw_keywords = response.split(', ')
    final_keywords = []
    for kw in raw_keywords:
        striped_kw = kw.strip()
        striped_kw = striped_kw.strip('\n')
        if (qbody_lower.find(striped_kw.lower()) != -1) and (striped_kw != ''):
            final_keywords.append(striped_kw)
    return final_keywords

# -----------------------------------------------------------------------------
# LLM Question answering methods
# -----------------------------------------------------------------------------
def qa_process_llm_response(text, qtype):
    ''' Parse the LLM response to fit the format of ideal and exact answers '''
    garbled, garbled_pos = qa_module.qa_resp_has_garbled_text(text)
    should_regen = False
    if garbled:
        text = text[:garbled_pos]
    answer = {}
    if qtype == 'summary':
        # Summary type question, verify the ideal answer is within 200 characters
        answer['ideal'] = helper.get_first_200_words(text)
        answer['exact'] = []
    else:
        # Extract exact answer.
        exact_keyword = 'Exact answer:'
        pos = text.find(exact_keyword)
        if pos == -1:
            answer['ideal'] = helper.get_first_200_words(text)
            answer['exact'] = []
        else:
            answer['ideal'] = helper.get_first_200_words(text[:pos])
            exact_payload = text[pos+len(exact_keyword):]
            answer['exact'] = qa_module.qa_format_exact_answer(exact_payload, qtype)
        if qtype == 'yesno':
            lower_ideal = answer['ideal'].lower()
            if (lower_ideal.startswith('yes') or lower_ideal.startswith('no')) and len(lower_ideal) <= 4:
                # Ideal answer should be more than just yes or no
                should_regen = True
            if len(answer['exact']) == 0:
                # If yesno question has no exact answer, try to extract from ideal
                # If ideal answer starts with 'yes'/'no', then exact is 'yes'/'no'. 
                if lower_ideal.startswith('yes'):
                    answer['exact'] = 'yes'
                elif lower_ideal.startswith('no'):
                    answer['exact'] = 'no'
    return answer, should_regen

def qa_ask_llm(cur_model, ques, context, log_file=None):
    '''
    Given a question and the context (optional), return the answer from LLM with some
    post-processing.
    '''
    qbody = ques.body
    qtype = ques.type
    max_tries = 3
    ideal_answer_tag = 'Ideal answer:'
    question_config = prompt_config[qtype]
    if context:
        prompt = question_config['context_prompt'].format(context=context, body=qbody)
    else:
        prompt = question_config['prompt'].format(body=qbody)
    if config.VERBOSE:
        print("\n" + "-"*80 + "\n")
        print(f'Inspect new prompt: len={len(prompt)}')
        print(prompt)
        print("\n" + "-"*80 + "\n")
    req = { 'prompt': prompt,
            'max_tokens': question_config['max_tokens'],
            'stop': '###'}
    if config.VERBOSE:
        print(req)
        print(cur_model)
        print("\n" + "-"*80 + "\n")

    raw_answers = []

    next_step = "None"
    for cnt in range(max_tries):
        if next_step == "MORE_TOKEN":
            req['prompt'] = final
        elif next_step == "RESAMPLE":
            req['prompt'] = prompt
        else:
            req['prompt'] = prompt
        result = llm.textsynth_completion(cur_model, req, 'QA', config.FAKE_LLM, ques)

        if config.VERBOSE:
            print(result)
            print("\n" + "-"*80 + "\n")
        response = result['text']
        raw_answers.append(response)
        final = prompt + response

        if config.INFO_TRACE:
            print("\n" + "-"*80 + "\n")
            print(f'LLM QA cnt-{cnt+1}') 
            print(f'context: {context}')
            print(f'q: {qbody}')
            print(f'a: {response}')

        # Find the position of last ideal answer tag in final string
        ideal_answer_start = final.rfind(ideal_answer_tag)
        if ideal_answer_start != -1:
            response = final[ideal_answer_start + len(ideal_answer_tag):]
        answer, should_regen = qa_process_llm_response(response, qtype)
        if not should_regen and 'ideal' in answer and len(answer['ideal']) > 0:
            if qtype == 'summary':
                return answer, raw_answers
            else:
                if 'exact' in answer and len(answer['exact']) > 0:
                    return answer, raw_answers

        if result['finish_reason'] == 'length':
            # if the finish_reason is "length" and the answer generated so 
            # far makes sense, then continue to generate more tokens.
            next_step = "MORE_TOKEN"
            garbled,_ = qa_module.qa_resp_has_garbled_text(response)
            if garbled or should_regen:
                print("Garbled text detected, retry")
                next_step = "RESAMPLE"
            elif qtype == 'yesno' and 'Exact answer:' in response:
                # yesno answer doesn't return good exact answer, do resample
                next_step = "RESAMPLE"
            elif ideal_answer_tag in response:
                next_step = "RESAMPLE"   
        else:
            next_step = "RESAMPLE"

        print(f'Unable to parse output, retry {next_step}')
        if log_file:
            log_file.write(f"\n> {datetime.datetime.now()}\n")
            log_file.write(final)
            json.dump(req, log_file, indent=2)
            log_file.write(cur_model)
            print('write result to file')

    return answer, raw_answers

# -----------------------------------------------------------------------------
# Snippet extraction helpers
# -----------------------------------------------------------------------------
def extract_snippets(article_list, qbody, st_model):
    ''' 
        Given a list of article, a question and an embeding model, return the
        most relevant sentence in each article according to embedding simliarity. 
    '''
    snip_list = []
    for article in article_list:
        begin_loc = 0
        if article['abstract_raw']:
            # get the first sentence from the abstract
            abstract = article['abstract_raw']
            if config.SNIP_EXTRACT_MODE == 'FIRST_SENTENCE':
                end_loc = abstract.find('.')
                if end_loc == -1:
                    end_loc = len(abstract)
                snippet_str = abstract[0:end_loc]
                section = 'abstract'
            elif config.SNIP_EXTRACT_MODE == 'TRANSFORMER':
                # embedding using transformer
                abstract_list = article['abstract_list']
                all_sentences = []
                for abstract_clip in abstract_list:
                    abstract_sentences = abstract_clip.split('. ')
                    all_sentences.extend(abstract_sentences)
                # add title sentence
                if article['title']:
                    all_sentences.append(article['title'])
                if config.VERBOSE:
                    print('pmid', article['pmid'])
                sentence_rank = helper.get_top_results(st_model, qbody, all_sentences, 3)
                snippet_str = all_sentences[sentence_rank[1][0]]
                if snippet_str == '':
                    continue
                section, begin_loc, end_loc = helper.locate_snip(snippet_str, article)
            else:
                assert(not 'unrecognized extract mode')
        elif article['title']:
            title = article['title']
            end_loc = len(title)
            snippet_str = title[0:end_loc]
            section = 'title'
        else:
            continue
        snip = {}
        snip['document'] = 'http://www.ncbi.nlm.nih.gov/pubmed/' + article['pmid']
        snip['offsetInBeginSection'] = begin_loc
        snip['offsetInEndSection'] = end_loc
        snip['beginSection'] = section
        snip['endSection'] = section
        snip['text'] = snippet_str
        snip_list.append(snip)
    return snip_list


def get_context_snippet_groups(ques_obj):
    '''
        Given a question and a list of golden snippets, compose the question
        context to send to LLM for QA task
    '''
    snippets = ques_obj.snippets
    context_group = []
    context_out = ''
    curr_length = 0

    def _add_text_to_group(text):
        if text and text[-1] == '\n':
            text = text[:-1]
        if text != '':
            context_group.append(text)
    
    if ques_obj.type == 'list':
        if config.LIST_SINGLE_PROMPT:
            tmp_context = ''
            for snip in snippets:
                tmp_context += (snip['text'] + '\n')
                if len(tmp_context) >= config.MAX_CONTEXT_LEN:
                    break
            _add_text_to_group(tmp_context)
        else:
            for snip in snippets:
                _add_text_to_group(snip['text'])
    else:
        if len(snippets) > 0:
            _add_text_to_group(snippets[0]['text'])

        if len(snippets) > 1:
            for snip in snippets[1:]:
                # get max 1000 characters snip context
                if curr_length + len(snip['text']) >= config.MAX_CONTEXT_LEN:
                    _add_text_to_group(context_out)
                    context_out = ''
                    curr_length = 0        
                context_out += (snip['text'] + '\n')
                curr_length += (len(snip['text']) + 1)
            _add_text_to_group(context_out)

    if config.VERBOSE:
        print('context_group')
        print(context_group)
    if len(context_group) == 0:
        return []

    if ques_obj.type == 'list':
        selected_context_grp = context_group[:5]
    elif ques_obj.type == 'yesno':
        selected_context_grp = context_group[:3]
    elif ques_obj.type == 'factoid':
        selected_context_grp = context_group[:1]
    else:
        selected_context_grp = context_group[:1]
    return selected_context_grp

def append_log(filename, log_data):
    """Append a log entry to a JSONL file."""
    with open(filename, 'a') as file:  
        json_string = json.dumps(log_data)
        file.write(json_string + '\n')  


def run(input_filename, stage, do_validation):
    input_path = f'input/{input_filename}.json'
    cur_time = int(time.time())
    output_path = f'output/res_stage{stage}_{input_filename}_{cur_time}.json'
    submission_path = f'output/submission_stage{stage}_{input_filename}_{cur_time}.json'
    output_file = open(output_path, 'w')
    submission_file = open(submission_path, 'w')
    log_file = open('logs/log_error.txt', "a")
    incremental_log_file_path = f'logs/log_inc_{cur_time}.jsonl'
    incremental_res_file_path = f'logs/res_inc_{cur_time}.jsonl'
    collect_llm_qa = True if stage == 'B' and not config.FAKE_LLM else False
    if collect_llm_qa:
        qa_collection_file = open(f'output/qa_ans_{cur_time}.json', 'w')
        qa_results = {}

    questions = helper.parse_input(input_path)

    print(f'LLM model: {config.LLM_MODEL}, SEARCH_WORD_MODE: {config.SEARCH_WORD_MODE}, Input: {input_path}, Validate: {do_validation}')
    print(f'ADD_QA_CONTEXT: {config.ADD_QA_CONTEXT}, ENABLE_SYNONYM_GROUPING: {config.ENABLE_SYNONYM_GROUPING}, FAKE_LLM: {config.FAKE_LLM}, LIST_SINGLE_PROMPT: {config.LIST_SINGLE_PROMPT}')
    start_time = datetime.datetime.now()
    spacy_model = None
    st_model = None
    if stage == 'A':
        if config.SEARCH_WORD_MODE == 'SPACY':
            spacy_model = search_utils.load_spacy_model()
        if config.SNIP_EXTRACT_MODE == 'TRANSFORMER':
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
    output_dict = {'questions': []}
    submission_dict = {'questions': []}
    proc_rec = 0
    proc_qa = 0
    total_rec = len(questions)
    # questions = questions[:10]
    for ques_dict in questions:
        ques_obj = helper.init_ques(ques_dict, stage=stage)
        proc_rec += 1
        if config.INFO_TRACE:
            print('\n' + '-'*80 + '\n')
            print(f'({proc_rec}/{total_rec}) Processing {ques_obj.id} - {ques_obj.type} - {ques_obj.body}')
        qbody = helper.sanity_question(ques_obj.body)
        if (qbody != ques_obj.body):
            print(f'Found malformed question, rewrite it as {qbody}')

        # 1. Perform IR task
        if stage == 'A': 
            ir_log_obj = {'id': ques_obj.id, 'body': qbody, 'type': ques_obj.type, 'mode': config.SEARCH_WORD_MODE, 'outputs': []}
            if config.SEARCH_WORD_MODE == 'SPACY':
                named_entity_list = search_utils.extract_entities(spacy_model, qbody)
                keywords = [t[0] for t in named_entity_list]
                ir_log_obj['outputs'].append(keywords)
                pmid_list =  search_utils.query_by_keywords(keywords, verbose=config.VERBOSE)
            elif config.SEARCH_WORD_MODE == 'LLM':
                keywords = llm_get_keywords(config.LLM_MODEL, qbody, ques_obj.id)
                ir_log_obj['outputs'].append(keywords)
                pmid_list =  search_utils.query_by_keywords(keywords, verbose=config.VERBOSE)
            elif config.SEARCH_WORD_MODE == 'MIXTRAL_47B':
                query_term = llm.query_expansion(qbody, config.FAKE_LLM)
                ir_log_obj['outputs'].append(query_term)
                pmid_list = search_utils.get_pmids(query_term, verbose=config.VERBOSE)
            elif config.SEARCH_WORD_MODE == 'OPENAI':
                query_term = llm.query_expansion(qbody, config.FAKE_LLM, model_name='OPENAI')
                ir_log_obj['outputs'].append(query_term)
                pmid_list = search_utils.get_pmids(query_term, verbose=config.VERBOSE)
                # attempt retry when pmid_list has less than 3 results
                retry_count = 0
                while len(pmid_list) < 3:
                    if retry_count == 0:
                        new_query_term = llm.query_expansion_retry(qbody, query_term, config.FAKE_LLM)
                        ir_log_obj['outputs'].append(new_query_term)
                        print(f'Retrying with Keywords: {new_query_term}')

                        pmid_list = search_utils.get_pmids(new_query_term, verbose=config.VERBOSE)
                        query_term = new_query_term
                        retry_count += 1
                    else:
                        # last try
                        query_term = llm.query_expansion_last_try(qbody, config.FAKE_LLM)
                        ir_log_obj['outputs'].append(query_term)
                        print(f'Last try with Keywords: {query_term}')

                        pmid_list = search_utils.get_pmids(query_term, verbose=config.VERBOSE)
                        break
            else:
                assert(not 'Unrecognized search mode')

            # save log
            append_log(incremental_log_file_path, ir_log_obj)
                
            if config.INFO_TRACE:
                print('PUBMED list length:', len(pmid_list))
            article_list = search_utils.query_by_pmids(pmid_list, verbose=config.VERBOSE)

            ques_obj.documents = pmid_list
            ques_obj.snippets = extract_snippets(article_list, qbody, st_model)

        # 2. Perform QA task. In stage B, Golden Documents and snippets are already loaded from inputs file
        # elif stage == 'B' and ques_dict['type'] == 'list':
        elif stage == 'B':
            if not config.ADD_QA_CONTEXT:
                llm_ans, raw_ans_list = qa_ask_llm(config.LLM_MODEL, ques_obj, '', log_file,)
            else:
                raw_ans_list = []
                llm_pred_list = []
                context_group = get_context_snippet_groups(ques_obj)
                if len(context_group) == 0:
                    llm_ans, raw_ans_list = qa_ask_llm(config.LLM_MODEL, ques_obj, '', log_file)
                else:
                    for context in context_group:
                        if config.VERBOSE:
                            print('context')
                            print(context)
                        assert(len(context) > 0)
                        llm_partial_ans, raw_ans_partial_list = qa_ask_llm(config.LLM_MODEL, ques_obj, context, log_file)
                        raw_ans_list.extend(raw_ans_partial_list)
                        llm_pred_list.append(llm_partial_ans)
                    llm_ans = qa_module.qa_postproc_prediction(ques_obj, llm_pred_list)
            if collect_llm_qa:
                qa_results[ques_obj.id] = raw_ans_list
            ques_obj.ideal_answer = llm_ans['ideal']
            ques_obj.exact_answer = llm_ans['exact']
            proc_qa += 1
            append_log(incremental_res_file_path, ques_obj.to_submission_dict(stage))
            
        output_dict['questions'].append(ques_obj.to_dict())
        submission_dict['questions'].append(ques_obj.to_submission_dict(stage))

    # Write the output json object to a file
    json.dump(output_dict, output_file, indent=2, sort_keys=False)
    json.dump(submission_dict, submission_file, indent=2, sort_keys=False)
    if collect_llm_qa:
        json.dump(qa_results, qa_collection_file, indent=2, sort_keys=False)

    # Calculate the evaluation metrics if validate is true.
    # We expect the golden answer is stored in the input file
    if do_validation:
        if stage == 'A':
            metrics.calculate_ir_accuracy(questions, output_dict['questions'])
        else:
            metrics.calculate_qa_accuracy(questions, output_dict['questions'])

    print('\n## Output\n')
    print(f'For input file \"{input_path}\"')
    print(f'Proccessed record count: {proc_rec}, qa_rec: {proc_qa}')
    print(f'See result in output file \"{output_path}\"')
    print(f'See submission file \"{submission_path}\"')
    print(f'Start time: {start_time}')
    print(f'End time: {datetime.datetime.now()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task12b model runner')
    parser.add_argument('--input', type=str, default = 'small_question', help="The file name of the input json file")
    parser.add_argument('--stage', type=str, default = 'B', help="Stage A or B of the task")
    parser.add_argument('--validate', action='store_true', help="Run validation code to calculate the accuracy given the golden answer")
    args = parser.parse_args()
    run(args.input, args.stage, args.validate)
