import argparse
import datetime
import json
import re
import time
import torch
import search_utils
import textsynth_api as llm
import helper_utils as helper
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer, util
from prompt_utils import Question, prompt_config, keyword_extact_prompt

RUN_IR = True
RUN_QA = True
FAKE_LLM = True
ZERO_SHOT = False
SEARCH_WORD_LIST =[
    'SPACY',
    'LLM'
]
SEARCH_WORD_MODE = SEARCH_WORD_LIST[0]

LLM_LIST = [
    'mistral_7B',
    'llama2_7B',
    'mixtral_47B_instruct',
    'llama2_70B',
    'gptj_6B'
]
LLM_MODEL = LLM_LIST[0]

SNIP_EXTRACT_LIST = [
    'first_sentence',
    'transformer'
]
SNIP_EXTRACT_MODE = SNIP_EXTRACT_LIST[1]

VERBOSE = False
INFO_TRACE = True
MAX_CONTEXT_LEN = 1000

# -----------------------------------------------------------------------------
# LLM Question keyword extractor methods
# -----------------------------------------------------------------------------
def llm_get_keywords(cur_model, qbody, qid):
    if FAKE_LLM:
        if qid == '63b6de63c6c7d4d31b000039':
            return ['colon cancer', 'NGS', 'sequencing']
        elif qid == '659838b906a2ea257c000033':
            return ['DUB', 'function']
        elif qid == '658f355206a2ea257c00000b':
            return ['senolytics', 'senomorphics']
        elif qid == '658f3e7106a2ea257c000016':
            return ['light chain amyloidosis', 'AL']
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
    result = llm.textsynth_completion(cur_model, req, 'IR', FAKE_LLM)
    if VERBOSE:
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
def format_list_exact_answer(json_payload):
    '''
    Parse the response from LLM to get the exact answer for 'list' and 'factoid'
    type questions.
    '''
    out_list = []
    if len(json_payload) > 0:
        for in_list in json_payload:
            if isinstance(in_list, list):
                for entity in in_list:
                    out_list.append(entity)
            if isinstance(in_list, str):
                out_list.append(in_list)
    return out_list

def format_exact_answer(payload, qtype):
    '''
    Parse the response from LLM to get the exact answer with the correct format
    '''
    spayload = payload.strip()
    if qtype == 'yesno':
        has_yes = re.search('yes', spayload, re.IGNORECASE)
        if has_yes:
            return 'yes'
        has_no = re.search('no', spayload, re.IGNORECASE)
        if has_no:
            return 'no'
        return ''
    elif qtype == 'factoid' or qtype == 'list':
        out_list = []
        out_list_set = False
        if helper.is_json(spayload):
            json_payload = json.loads(spayload)
            if isinstance(json_payload, list):
                out_list = format_list_exact_answer(json_payload)
                out_list_set = True
        elif spayload.startswith('['):
            # Processing malformed json
            npayload = spayload
            if spayload.startswith('[[\"'):
                correct_trailing = '\"]]'
                npayload = helper.remove_trailing_characters(spayload, correct_trailing)
                npayload += correct_trailing
            elif spayload.startswith('[\"'):
                correct_trailing = '\"]'
                npayload = helper.remove_trailing_characters(spayload, correct_trailing)
                npayload += correct_trailing
            else:
                print("Unhandled output")
                print(spayload)

            if helper.is_json(npayload):
                json_payload = json.loads(npayload)
                if isinstance(json_payload, list):
                    out_list = format_list_exact_answer(json_payload)
                    out_list_set = True
            else:
                # lift all the '[]"' characters and split the names separated by comma
                npayload = spayload
                if ('[' in npayload) and (']' in npayload) and ('"' in npayload):
                    for character in '[]\"':
                        npayload = npayload.replace(character, '')
                    entities = npayload.split(', ')
                    for entity in entities:
                        if ',' in entity:
                            nested_entities = entity.split(',')
                            for nentity in nested_entities:
                                out_list.append(nentity)
                        else:
                            out_list.append(entity)
                    out_list_set = True
        if not out_list_set:
            spayload = spayload.splitlines()
            for entity in spayload:
                entity_prune = entity.strip()
                out_list.append(entity_prune)
        flatten_list = [[item] for item in out_list]
        if qtype == 'factoid' and len(flatten_list) > 5:
            return flatten_list[:5]
        if qtype == 'list' and len(flatten_list) > 100:
            return flatten_list[:100]
        return flatten_list
    else:
        return ''

def process_llm_response(text, qtype):
    ''' Parse the LLM response to fit the format of ideal and exact answers '''
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
            answer['exact'] = format_exact_answer(exact_payload, qtype)
    return answer

def ask_llm(cur_model, qbody, qtype, context, log_file=None):
    '''
    Given a question and the context (optional), return the answer from LLM with some
    post-processing.
    '''
    max_tries = 2
    question_config = prompt_config[qtype]
    if context:
        prompt = question_config['context_prompt'].format(context=context, body=qbody)
    else:
        prompt = question_config['prompt'].format(body=qbody)
    if VERBOSE:
        print("\n" + "-"*80 + "\n")
        print(f'Inspect new prompt: len={len(prompt)}')
        print(prompt)
        print("\n" + "-"*80 + "\n")
    req = { 'prompt': prompt,
            'max_tokens': question_config['max_tokens'],
            'stop': '###',
            'qtype': qtype }
    if VERBOSE:
        print(req)
        print(cur_model)
        print("\n" + "-"*80 + "\n")

    for cnt in range(max_tries):
        result = llm.textsynth_completion(cur_model, req, 'QA', FAKE_LLM)
        if VERBOSE:
            print(result)
            print("\n" + "-"*80 + "\n")
        response = result['text']
        final = prompt + response

        if INFO_TRACE:
            print(f'LLM QA cnt-{cnt+1}') 
            print(f'q: {qbody}')
            print(f'a: {response}')
            print("-"*80)

        answer = process_llm_response(response, qtype)
        if 'ideal' in answer and len(answer['ideal']) > 0:
            if qtype == 'summary':
                return answer
            else:
                if 'exact' in answer and len(answer['exact']) > 0:
                    return answer
        
        print('Unable to parse output, retry')
        if log_file:
            log_file.write(f"\n> {datetime.datetime.now()}\n")
            log_file.write(final)
            json.dump(req, log_file, indent=2)
            log_file.write(cur_model)
            print('write result to file')

    return answer

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
            if SNIP_EXTRACT_MODE == 'FIRST_SENTENCE':
                end_loc = abstract.find('.')
                if end_loc == -1:
                    end_loc = len(abstract)
                snippet_str = abstract[0:end_loc]
                section = 'abstract'
            elif SNIP_EXTRACT_MODE == 'transformer':
                # embedding using transformer
                abstract_list = article['abstract_list']
                all_sentences = []
                for abstract_clip in abstract_list:
                    abstract_sentences = abstract_clip.split('. ')
                    all_sentences.extend(abstract_sentences)
                # add title sentence
                if article['title']:
                    all_sentences.append(article['title'])
                if VERBOSE:
                    print('pmid', article['pmid'])
                sentence_rank = helper.get_top_results(st_model, qbody, all_sentences, 3)
                snippet_str = all_sentences[sentence_rank[1][0]]
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
        snip['document'] = article['pmid']
        snip['offsetInBeginSection'] = begin_loc
        snip['offsetInEndSection'] = end_loc
        snip['beginSection'] = section
        snip['endSection'] = section
        snip['text'] = snippet_str
        snip_list.append(snip)
    return snip_list

def filter_doc_from_feedback(ques_obj, feedback_dict, pmid_list):
    ''' 
        Given a list of documents, remove the ones that already exist in the
        feedback file
    '''
    out_pmid_list = []
    qid = ques_obj.id
    if qid in feedback_dict:
        for pmid in pmid_list:
            if pmid in feedback_dict[qid]['pos_docs']:
                print(f'removed {pmid} because article in positive feedback')
            elif pmid in feedback_dict[qid]['neg_docs']:
                print(f'removed {pmid} because article in negative feedback')
            else:
                out_pmid_list.append(pmid)
    return out_pmid_list

def get_context_snippet(ques_obj, feedback_dict):
    '''
        Given a question and a list of potential snippets, compose the question
        context to send to LLM for QA task
    '''
    # get max 1000 characters snip context
    context_out = ''
    cur_length = 0
    qid = ques_obj.id
    if (qid not in feedback_dict) or ('pos_snippets' not in feedback_dict[qid]):
        print(f'Feedback file doesn\'t have golden snippet for question {qid}')
        return context_out

    pos_snip_list = feedback_dict[qid]['pos_snippets']
    if len(pos_snip_list) > 0:
        ## First check the snippets from the feeback file
        for snip in pos_snip_list:
            if cur_length + len(snip) >= MAX_CONTEXT_LEN:
                break
            context_out += (snip + '\n')
            cur_length += (len(snip) + 1)
    else:
        ## Then use the snippets we found
        for snip in ques_obj.snippets:
            if cur_length + len(snip['text']) >= MAX_CONTEXT_LEN:
                break
            context_out += (snip['text'] + '\n')
            cur_length += (len(snip['text']) + 1)

    if context_out and context_out[-1] == '\n':
        context_out = context_out[:-1]
    return context_out

def run(input_filename):
    ''' Synergy model '''
    input_path = f'input/{input_filename}.json'
    output_path = f'output/res_{input_filename}_{int(time.time())}.json'
    output_file = open(output_path, 'w')
    log_file = open('logs/log_error.txt', "a")

    questions = helper.parse_input(input_path)
    feedback_snip_dict = json.loads(Path('output/feedback').read_text())

    print(f'LLM model: {LLM_MODEL}, SEARCH_WORD_MODE: {SEARCH_WORD_MODE}, ZERO_SHOT: {ZERO_SHOT}, Input: {input_path}')
    start_time = datetime.datetime.now()
    spacy_model = None
    st_model = None
    if SEARCH_WORD_MODE == 'SPACY':
        spacy_model = search_utils.load_spacy_model()
    if SNIP_EXTRACT_MODE == 'transformer':
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
    output_dict = {'questions': []}
    proc_rec = 0
    proc_qa = 0
    total_rec = len(questions)
    for ques_dict in questions:
        ques_obj = helper.init_ques(ques_dict)
        proc_rec += 1
        if INFO_TRACE:
            print(f'({proc_rec}/{total_rec}) Processing {ques_obj.id} - {ques_obj.body}')
        qbody = helper.sanity_question(ques_obj.body)
        if (qbody != ques_obj.body):
            print(f'Found malformed question, rewrite it as {qbody}')

        # 1. Perform IR task
        if RUN_IR:
            if SEARCH_WORD_MODE == 'SPACY':
                named_entity_list = search_utils.extract_entities(spacy_model, qbody)
                keywords = [t[0] for t in named_entity_list]
            elif SEARCH_WORD_MODE == 'LLM':
                keywords = llm_get_keywords(LLM_MODEL, qbody, ques_obj.id)
            else:
                assert(not 'Unrecognized search mode')
            if INFO_TRACE:
                print(f'Keywords: {keywords}')
            if keywords and len(keywords) > 0:
                pmid_list =  search_utils.query_by_keywords(keywords, verbose=VERBOSE)
                if INFO_TRACE:
                    print('Pubmed list: ')
                    pprint(pmid_list)
                filtered_pmd_list = filter_doc_from_feedback(ques_obj, feedback_snip_dict, pmid_list)
                article_list = search_utils.query_by_pmids(filtered_pmd_list, verbose=VERBOSE)
                ques_obj.documents = filtered_pmd_list
                ques_obj.snippets = extract_snippets(article_list, qbody, st_model)

        # 2. Perform QA task
        if RUN_QA and ques_obj.answer_ready:
            # Only perform QA task for questions that are 'ready to answer"
            if ZERO_SHOT:
                llm_ans = ask_llm(LLM_MODEL, ques_obj.body, ques_obj.type, '', log_file)
            else:
                llm_context = get_context_snippet(ques_obj, feedback_snip_dict)
                llm_ans = ask_llm(LLM_MODEL, ques_obj.body, ques_obj.type, llm_context, log_file)
            ques_obj.ideal_answer = llm_ans['ideal']
            ques_obj.exact_answer = llm_ans['exact']
            proc_qa += 1
        else:
            if ques_obj.type == 'yesno':
                ques_obj.exact_answer = 'yes'
        output_dict['questions'].append(ques_obj.to_dict())

    # Write the output json object to a file
    json.dump(output_dict, output_file, indent=2, sort_keys=False)
    print("-"*80)
    print(f'For input file \"{input_path}\"')
    print(f'Proccessed record count: {proc_rec}, qa_rec: {proc_qa}')
    print(f'See result in output file \"{output_path}\"')
    print(f'Start time: {start_time}')
    print(f'End time: {datetime.datetime.now()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synergy model runner')
    parser.add_argument('--input', type=str, default = 'syn24_testset_llm_small', help="The file name of the input json file")
    args = parser.parse_args()
    run(args.input)
