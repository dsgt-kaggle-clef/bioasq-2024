''' Helper methods'''

import datetime
import json
import re
import time
import torch
import search_utils
import textsynth_api as llm
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer, util
from prompt_utils import Question, prompt_config, keyword_extact_prompt

def parse_input(file_path, verbose=False):
    ''' return json object from a json file '''
    ques_list = json.loads(Path(file_path).read_text())['questions']
    if verbose:
        pprint(ques_list)
    return ques_list

def init_ques(ques):
    ques_obj = Question(ques['body'], ques['id'], ques['type'], ques['answerReady'])
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
    if (word_count(text)<200):
        return text
    else:
        return text.split()[:200]

def remove_trailing_characters(orig_str, substr):
    for c in reversed(substr):
        if orig_str[-1] == c:
            orig_str = orig_str[:-1]
    return orig_str

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
