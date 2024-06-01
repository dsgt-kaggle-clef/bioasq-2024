import os
import json
import re
from pathlib import Path
from pprint import pprint
import requests
import time
import datetime
from tokens import * 
import config
import prompts

def get_word_cnt(text):
    '''
    Get the word count of the text
    '''
    return len(text.split())

def match_ignore_case(text, entity_list):
    '''
    For each item in the entity_list, check if the text match any one of the item, ignoring the case
    '''
    for entity in entity_list:
        if text.lower() == entity.lower():
            return True
    return False

def get_altname(entity):
    '''
    Get the alternative name of the entity
    '''
    alt_entity = entity
    if entity[0].isupper():
        alt_entity = entity[0].lower() + entity[1:]
    if entity[0].islower():
        alt_entity = entity[0].upper() + entity[1:]
    return alt_entity

def find_entity_loc_in_text(entity, text, start_ind):
    '''
    Find the location of the entity in the text
    '''
    loc = text.find(entity, start_ind)
    if loc == -1:
        return None
    return (loc, loc + len(entity))

def find_all_loc_in_text(entity, text):
    '''
    Find all the location of the entity in the text
    '''
    loc_list = []
    allforms = [entity]
    if get_altname(entity) != entity:
        allforms.append(get_altname(entity))
    for form in allforms:
        start_ind = 0
        while start_ind < len(text):
            loc_tuple = find_entity_loc_in_text(form, text, start_ind)
            if not loc_tuple:
                break
            loc_list.append(loc_tuple)
            start_ind = loc_tuple[1]
    return loc_list

def cnt_tp_fp_fn(actual, expect):
    '''
    Calculate TP, FP and FN given a list of actual and expect named entities
    '''
    tp = 0.0
    fp = 0.0
    for a_entity in actual:
        if match_ignore_case(a_entity, expect):
            tp += 1
        else:
            fp += 1
    # calculate FN: entity exist in expect but not in actual
    fn = 0.0
    for e_entity in expect:
        if not match_ignore_case(e_entity, actual):
            fn += 1
    return tp, fp, fn

def calculate_precision_recall_f1(tp, fp, fn, verbose):
    '''
    Calculate precision, recall and F1 score given TP, FP and FN
    '''
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    if verbose:
        print(f'TP {tp}, FP {fp}, FN {fn}')
        print(f'precision {precision:.4f}\nrecall {recall:.4f}\nf1 {f1:.4f}')
        print('*'*80)
        print()
    return precision, recall, f1

def calculate_accuracy(actual, expect, verbose=False):
    '''
    Calculate precision, recall and F1 score given a list of
    expected and actual named entities.
    The entities must be exact match.
    https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    '''
    tp, fp, fn = cnt_tp_fp_fn(actual, expect)

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    if tp == 0:
        return 0.0, 0.0, 0.0

    return calculate_precision_recall_f1(tp, fp, fn, verbose)

def compare_entities(actual, expect):
    '''
    print the entities that exist in both actual and expect
    print the difference between actual and expect
    '''
    unmatch_list = []
    for a_entity in actual:
        if match_ignore_case(a_entity, expect):
            print("(match)", a_entity)
        else:
            unmatch_list.append(a_entity)
    for e_entity in expect:
        if not match_ignore_case(e_entity, actual):
            print("(not found)", e_entity)
    print("(false positives): ", unmatch_list)

def get_closest_examples(train_dict, candidate_list, selected_tag, text):
    '''
    Given a list of candidate PMIDs, return the PMIDs that is closest to the text and has selected 
    tag if possible.
    '''
    # 1. Use fixed examples
    if selected_tag == 'DEVICE':
        return ['26281196_en', '26600613_en']
    elif selected_tag == 'INJURY_POISONING':
        return ['26036067_en', '26525480_en']
    else:
        return ['25823269_en', '25842923_en']

def create_few_shot_prompt(sample_dict, id_list, tag):
    '''
    Create a few-shot prompt for the LLM model
    '''
    example_prompt = ''
    if config.ADD_INSTRUCTION:
        example_prompt += prompts.get_instruction(tag)

    for pmid in id_list:
        if tag in sample_dict[pmid]:
            tag_str = ';'.join(sample_dict[pmid][tag])
        else:
            tag_str = 'None'
        text = sample_dict[pmid]['text']
        example_prompt += f'[TEXT]: {text}\n[{tag}]: {tag_str}\n###\n'
    return example_prompt

def sanitize_llm_output(llm_resp, original_text, selected_tag):
    '''
    Given a list of named entities, remove the duplicates and the ones
    that don't exist in the original text
    '''
    non_dup_actual = set()
    # check if llm_resp is a string
    if isinstance(llm_resp, str):
        resp_list = [llm_resp]
    else:
        resp_list = llm_resp
    for one_resp in resp_list:
        raw_list = one_resp.strip().split(';')
        for name in raw_list:
            if len(name) > 0:
                if name == 'None' or name == 'none':
                    continue
                # Check if the entity exist in the original text, ignoring the case
                name1 = name.strip()
                if name1 in original_text:
                    non_dup_actual.add(name1)
                else:
                    name2 = get_altname(name1)
                    if name2 in original_text:
                        non_dup_actual.add(name2)
    non_dup_list =  list(non_dup_actual)
    print('List before filtering: ', non_dup_list)

    # Remove more than 10 words phases
    list2 = non_dup_list.copy()
    for name in list2:
        if get_word_cnt(name) >= 10:
            non_dup_list.remove(name)
            continue
        # remove entity that only has numbers
        if name.isnumeric():
            non_dup_list.remove(name)
            continue

    return non_dup_list