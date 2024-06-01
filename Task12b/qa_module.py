'''LLM Question answering module logic'''

import datetime
import json
import re
import time
import config
import prompt_utils
import helper_utils as helper
import textsynth_api as llm
from pprint import pprint

def qa_format_exact_answer(payload, qtype):
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
        # TODO: add separate entity by '\n'
        # separate the entities by ';'
        entity_list = helper.divide_string_to_list(spayload)
        entity_list2 = []
        for entity in entity_list:
            # doing some basic sanity cleaning
            # if the entity starts with number, or star, remove them
            if len(entity) < 1:
                continue
            
            if entity[0] == '*' or entity[0] == '-' :
                entity_list2.append(entity[1:].strip())
            # regex if the entity starts with 1. or 1)
            else:
                match_obj = re.match(r'^[0-9]+[.)]', entity)
                if match_obj:
                    match_span = match_obj.span()
                    truncated_entity = entity[match_span[1]:]
                    entity_list2.append(truncated_entity.strip())
                else:
                    entity_list2.append(entity)

        # remove duplicate entities
        nondup_set = set(entity_list2)

        # convert set into list of list
        out_list = []
        for set_entity in nondup_set:
            out_list.append([set_entity])

        if qtype == 'factoid' and len(out_list) > 5:
            # return at most 5 entities for factoid question
            return out_list[:5]
        if qtype == 'list' and len(out_list) > 100:
            # return at most 100 entities for list question
            return out_list[:100]
        return out_list
    else:
        return ''

def qa_resp_has_garbled_text(text):
    ''' Check if the response from LLM has garbled text 
        If we find special tokens like '##' or '```' in the response,
        we consider it as garbled text.
    '''
    first_pos = len(text) + 1
    pound_pos = text.find('##')
    if pound_pos != -1 and pound_pos < first_pos:
        first_pos = pound_pos
    backstick_pos = text.find('```')
    if backstick_pos != -1 and backstick_pos < first_pos:
        first_pos = backstick_pos
    return first_pos != len(text) + 1, first_pos

def qa_synonmy_grouping(orig_dict):
    '''Group the synonyms in the entity string together'''
    if not config.ENABLE_SYNONYM_GROUPING :
        return orig_dict

    orig_list = list(orig_dict.keys())
    tracked_entity = set()
    if len(orig_list) < 1:
        return {}

    orig_str = helper.combine_list_to_string(orig_list)

    for i in range(3):
        llm_resp = llm.group_synonym_entity(orig_str)
        text = '[GROUP1]:' + llm_resp['text'] + '\n'
        print('\nLLM grouping response')
        print(text)
        # Extract the grouped entities in mock_text into lists
        entity_list = re.findall(r'\[GROUP([0-9]+)\]:(.+)\n', text)
        print(entity_list)
        if len(entity_list) > 1:
            break

    new_dict = {}
    for ind, entity_str in enumerate(entity_list):
        if str(ind + 1) != entity_str[0]:
            print('Error: LLM generation synonmy grouping unexpected number:', entity_str[0])
        word_list = helper.divide_string_to_list(entity_str[1])
        aggr_cnt = 0
        represent_word = ''
        represent_word_cnt = 0
        for word in word_list:
            if not helper.has_word_in_list(word, orig_list):
                print('Error: LLM generation synonmy grouping has unexpected word:', word)
            elif helper.has_word_in_list(word, tracked_entity):
                print('Error: LLM generation synonmy grouping has duplicate word:', word)
            else:
                indexed_word = helper.get_word_in_list(word, orig_list)
                aggr_cnt += orig_dict[indexed_word]                    
                tracked_entity.add(word)
                if orig_dict[indexed_word] > represent_word_cnt:
                    represent_word = word
                    represent_word_cnt = orig_dict[indexed_word]
                elif orig_dict[indexed_word] == represent_word_cnt:
                    if len(word) > len(represent_word):
                        represent_word = word
        new_dict[represent_word] = aggr_cnt

    untrack_list = []
    for word in orig_list:
        if not helper.has_word_in_list(word, tracked_entity):
            untrack_list.append(word)
            new_dict[word] = orig_dict[word]

    if len(untrack_list) > 0:
        print('Error: LLM generation synonmy grouping has untracked word', untrack_list)

    return new_dict

def list_answer_basic_filter1(term_rank):
    '''
        Basic filter for list answer
        Remove single character phase,
        Remove entity with multiple comma
    '''
    new_dict = {}
    for term in term_rank:
        if len(term) < 2:
            print('BasicFilter: remove entity due to length', term)
        elif term.count(',') > 1:
            print('BasicFilter: remove entity due to comma', term)
        else:
            new_dict[term] = term_rank[term]
    return new_dict

def qa_postproc_prediction(ques_obj, llm_pred_list):
    ''' Postprocess the QA predictions from LLM '''
    if ques_obj.type == 'yesno':
        num_yes, num_no = 0, 0
        yes_list = []
        no_list = []
        for pred in llm_pred_list:
            if pred['exact'] == 'yes':
                num_yes += 1
                yes_list.append(pred)
            elif pred['exact'] == 'no':
                num_no += 1
                no_list.append(pred)
        if num_yes > num_no:
            final_ans = {'ideal': yes_list[0]['ideal'], 'exact': 'yes'}
        elif num_no > num_yes:
            final_ans = {'ideal': no_list[0]['ideal'], 'exact': 'no'}
        else:
            final_ans = llm_pred_list[0]
    elif ques_obj.type == 'list':
        # Concatenate the list of entities of the exact answers
        final_ans = llm_pred_list[0].copy()
        concate_list = []
        term_rank = {}
        secondary_rank = {} # when there is a tie, the smaller rank number takes precedence
        for sentence_id, predict_entry in enumerate(llm_pred_list):
            for exact_answer in predict_entry['exact']:
                exact_str = exact_answer[0]
                if exact_str in term_rank:
                    term_rank[exact_str] += 1
                else:
                    term_rank[exact_str] = 1
                    secondary_rank[exact_str] = sentence_id + 1
        # sort the items by the frequency of the term
        term_rank = dict(sorted(term_rank.items(), key=lambda item: item[1], reverse=True))
        print('\nRaw term rank')
        print(term_rank)
        print('\nSecondary rank')
        print(secondary_rank)

        term_rank = list_answer_basic_filter1(term_rank)
        print('\nRank after basic filter')
        print(term_rank.items())
        # We will group the synonyms of the potential entities together unless the
        # question explicitly asks for synonyms
        if ques_obj.body.lower().find('synonym') == -1:
            entity_raw_list = list(term_rank.keys())
            term_rank = qa_synonmy_grouping(term_rank)
            term_rank = dict(sorted(term_rank.items(), key=lambda item: item[1], reverse=True))
            print('\nRank after synonym grouping')
            print(term_rank)
        # Assemble all term in rank to a list of list
        term_list = list(term_rank.keys())

        # Do some sanity check.
        # First we only want 100 entities at most and each one cannot be longer than 100 characters
        round1_list = []
        for item in term_list[:100]:
            if len(item) > 100 or len(item) < 1:
                print('Error: LLM generation list answer has too long/short entity:', item)
            else:
                round1_list.append(item)
        print('round1_list')
        print(round1_list)
        # If there are more than 10 entities, we will only keep the above 10 entities if their frequency is above 1
        round2_list = []
        default_length = 10
        frequency1_items = []
        for idx, item in enumerate(round1_list):
            if term_rank[item] > 1:
                round2_list.append(item)
            else:
                frequency1_items.append(item)
        print('\nround2_list exclude frequency 1 items')
        print(round2_list)
        print('\nfrequency1_items')
        print(frequency1_items)
        item_cnt_diff = default_length - len(round2_list)
        if item_cnt_diff > 0:
            # sort the frequency1_item by the secondary rank
            tmp_dict_freq1 = {}
            for freq1_word in frequency1_items:
                alt_word = helper.get_word_in_list(freq1_word, secondary_rank)
                rank2 = secondary_rank[alt_word] if alt_word else 1000
                if alt_word not in secondary_rank:
                    print('Error: LLM generation list answer has missing secondary rank:', freq1_word)
                tmp_dict_freq1[freq1_word] = rank2
            frequency1_items = list(sorted(tmp_dict_freq1.keys(), key=lambda item: tmp_dict_freq1[item]))
            round2_list.extend(frequency1_items[:item_cnt_diff])

        print('\nround2_list final')
        print(round2_list)
        final_list = helper.to_nested_list(round2_list)
        final_ans['exact'] = final_list[:100]
    else:
        final_ans = llm_pred_list[0]
    return final_ans

def should_keep_term(term_rank, item):
    ''' Eliminate low frequency term '''
    if term_rank[item] > 1:
        return True
    if len(item.split()) > 1:
        return True
    # count number of uppercase letters in item
    if len(re.findall(r'[A-Z]', item)) < 0.5 * len(item):
        return True
    return False