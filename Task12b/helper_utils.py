''' Helper methods'''

import datetime
import json
import re
import time
import torch
import search_utils
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from sentence_transformers import util
from prompt_utils import Question, prompt_config, keyword_extact_prompt

def parse_input(file_path, verbose=False):
    ''' return json object from a json file '''
    ques_list = json.loads(Path(file_path).read_text())['questions']
    if verbose:
        pprint(ques_list)
    return ques_list

def init_ques(ques, stage='A'):
    ques_obj = Question(ques['body'], ques['id'], ques['type'])
    if stage == 'B':
        if 'documents' in ques:
            ques_obj.documents = ques['documents']
        if 'snippets' in ques:
            ques_obj.snippets = ques['snippets']
    return ques_obj

def is_json(myjson):
    ''' Check if a string is json'''
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

def word_count(text):
    return len(re.findall(r'\w+', text))

def get_first_200_words(text):
    text2 = text.strip()
    if (word_count(text2)<200):
        return text2
    else:
        return text2.split()[:200]

def remove_trailing_characters(orig_str, substr):
    for c in reversed(substr):
        if orig_str[-1] == c:
            orig_str = orig_str[:-1]
    return orig_str

def combine_list_to_string(entity_list):
    ''' Combine a list of entities to a string '''
    return ';'.join(entity_list)

def divide_string_to_list(entity_str):
    ''' Divide a string to a list of entities '''
    entity_str = entity_str.strip()
    split_entity = entity_str.split(';')
    result = []
    for item in split_entity:
        item_strp = item.strip()
        if item_strp != '':
            result.append(item_strp)
    return result

def to_nested_list(entities):
    ''' Convert a list, set or dictionary of entities to a nested list '''
    result = []
    for entity in entities:
        result.append([entity])
    return result

def has_word_in_list(word, word_list):
    for w in word_list:
        if word.lower() == w.lower():
            return True
    return False

def get_word_in_list(word, word_list):
    if word in word_list:
        return word
    for w in word_list:
        if word.lower() == w.lower():
            return w
    return None

def sanity_question(qbody):
    ''' Remove unexpected question mark in the question '''
    has_ending_question_mark = (qbody[-1] == '?')
    qnew = qbody.replace('?', '')
    if has_ending_question_mark:
        qnew += '?'
    return qnew

def get_top_results(st_model, query, sentences, topk=None, verbose=False):
    '''
        Given a embedding model, a query and a list setence, calculate the
        similarity between the sentences and the query using the distance in
        the embedding space.
        Return the rank of sentences according to the embedding similarity. 
    '''
    assert(len(sentences) > 0)
    if (not topk) or (topk > len(sentences)):
        topk = len(sentences)
    query_embedding = st_model.encode(query, normalize_embeddings=True)
    corpus_embeddings = st_model.encode(sentences, normalize_embeddings=True)
    dot_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(dot_scores, k=topk)
    if verbose:
        print(top_results)
        print()
    return top_results

def locate_snip(best_snip, article):
    ''' 
        Locate the given snippet location in the article,
        return the beginning and ending index of the snippet
    '''
    assert(best_snip != '')
    snip_length = len(best_snip)
    if article['abstract_raw']:
        start_ind = article['abstract_raw'].find(best_snip)
        if start_ind != -1:
            return 'abstract', start_ind, start_ind + snip_length

    start_ind = article['title'].find(best_snip)
    if start_ind != -1:
        return 'title', start_ind, start_ind + snip_length
    assert(not 'Unable to identify snippet location')

def remove_keywords_from_phase_a_results(filename, output_filename):
    ''' Remove keywords from phase A results '''
    data = {}
    with open(filename, 'r') as f:
        data = json.load(f)
    
    res = []
    for ques in data['questions']:
        obj = {
            'type': ques['type'],
            'body': ques['body'],
            'id': ques['id'],
            'ideal_answer': ques['ideal_answer'],
            'exact_answer': ques['exact_answer'],
            'documents': ques['documents'],
            'snippets': ques['snippets']
        }
        res.append(obj)
    res_obj = {'questions': res}
    with open(output_filename, 'w') as f:
        json.dump(res_obj, f, indent=2)
    print(f'Output written to {output_filename}')
    return
