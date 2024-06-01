import os
import json
import re
from pathlib import Path
from pprint import pprint
import requests
import time
import datetime
from tokens import * 
from simple_db.database import SimpleDB
import helper_utils as utils

semantic_type_dict = {
    'ANATOMY': 'A1.2',
    'DEVICE': 'A1.3.1',
    'CHEM': 'A1.4.1',
    'FINDING': 'A2.2',
    'LABPROC': 'B1.3.1.1',
    'PHYS': 'B2.2.1.1',
    'DISO': 'B2.2.1.2',
    'INJURY_POISONING': 'B2.3',
}

UMLS_URI = 'https://uts-ws.nlm.nih.gov'

def make_http_resp(url, query):
    for i in range(5):
        rsp = requests.get(url, params=query)
        try:
            rsp.json()
        except:
            print(f'Error HTTP request failed: {rsp.text}, {rsp.status_code}, {rsp}')
            time.sleep(1)
        else:
            return rsp
    print(f'Error HTTP request failed return: {rsp.text}')
    return {}

class UMLSModule:
    def __init__(self, cui_db, semantic_db, nonentity_db):
        self.cui_db = cui_db
        self.semantic_db = semantic_db
        self.nonentity_db = nonentity_db

    def umls_rules(self, entity, topk=1):
        '''
        Given an entity and a tag, return the suggested tag based on the UMLS category
        '''
        query = {'string':entity, 'apiKey':UMLS_TOKEN, 'pageSize': topk}
        nonentity_lookup = self.nonentity_db.get(entity)
        if nonentity_lookup is None:
            r1 = make_http_resp(UMLS_URI + '/search/current', query)
            r1.encoding = 'utf-8'
            res1 = r1.json()['result']['results']
            print()
            print('### entity: ', entity)
            if len(res1) == 0:
                self.nonentity_db.set(entity, '1')
                print(f'No results found for {entity}')
                return None
        else:
            print()
            print('### entity: ', entity)
            print(f'No results found for {entity}')
            return None

        cui_list = []
        concept_list = []
        for result in res1:
            concept_list.append(result['name'])
            cui_list.append(result['ui'])
        tag_list = []
        semantic_list = []
        for ind, cui in enumerate(cui_list):
            # look up the type from cui_db
            cui_lookup_res = self.cui_db.get(cui)
            if cui_lookup_res is not None:
                concept_name, semantic_name, tag_name = cui_lookup_res
                semantic_list.append(semantic_name)
                tag_list.append(tag_name)
            else:
                concept_endpoint = f'/content/current/CUI/{cui}'
                query = {'apiKey': UMLS_TOKEN}
                concpet_rsp = make_http_resp(UMLS_URI + concept_endpoint, query)
                concept_res = concpet_rsp.json()
                semantic_type = concept_res['result']['semanticTypes'][0]
                semantic_name = semantic_type['name']
                semantic_list.append(semantic_name)
                tag_name = self.semantic_db.get(semantic_name)
                if tag_name is not None:
                    tag_list.append(tag_name)
                    self.cui_db.set(cui, (concept_list[ind], semantic_name, tag_name))
                else:
                    semantic_uri = semantic_type['uri']
                    r3 = make_http_resp(semantic_uri, query)
                    res3 = r3.json()
                    semantic_treeid = res3['result']['treeNumber']
                    tag_name = 'OTHERS'
                    for stype in semantic_type_dict.keys():
                        if semantic_treeid.startswith(semantic_type_dict[stype]):
                            tag_name = stype
                            break
                        if stype == 'ANATOMY':
                            if semantic_treeid == 'A2.1.5.2': # Body location or region
                                tag_name = stype
                                break
                        if stype == 'LABPROC':
                            if semantic_treeid == 'B1.3.1.2': # Diagnostic Procedure
                                tag_name = stype
                                break
                    tag_list.append(tag_name)
                    self.cui_db.set(cui, (concept_list[ind], semantic_name, tag_name))
                    self.semantic_db.set(semantic_name, tag_name)

            if topk == 1:
                # Just check the most likely tag if we don't have a preference for the tag
                break

        print('cui_list', cui_list)
        print('concept_list', concept_list)
        print('semantic_list', semantic_list)
        print('tag_list', tag_list)
        if topk == 1:
            return tag_list[0]
        
        return [concept_list, semantic_list, tag_list]

    def get_bracket_entity(self, text):
        '''
        Get the entity inside the bracket
        '''
        bracket_list = []
        r1 = r'\(.*?\)' # match anything inside the bracket
        r2 = r'\([A-Z].*?\)' # match capital letter inside the bracket
        for match in re.finditer(r2, text):
            match_str = match.group()[1:-1]
            wrd_cnt = utils.get_word_cnt(match_str)
            if wrd_cnt < 3:
                bracket_list.append(match_str)
        # Get frequency of each word in the bracket
        tx_split = re.split(r'\W', text)
        res = {}
        bracket_list2 = bracket_list.copy()
        for word in bracket_list2:
            letter_cnt = len(word)
            cnt = tx_split.count(word)

            # remove if the acryonym is not frequent
            # if cnt < 2:
            #     bracket_list.remove(word)
            #     continue

            word_ind = text.find(f'({word})')
            # find the last x number words in text
            last_words_list = text[:word_ind].split()[-1 * letter_cnt:]
            # Check if first letter combined into acronym
            subword = ''
            for subwd in last_words_list:
                subword += subwd[0]
            if subword.lower() == word.lower():
                longform = (' ').join(last_words_list)
            else:
                longform = last_words_list[-1]
            # print(f'{word}: {cnt}, {longform}')
            res[word] = {'freq': cnt, 'longform': longform}
            # Reference: https://biotext.berkeley.edu/papers/psb03.pdf
        return bracket_list, res

    def detect_acronym_entity(self, text, get_tag=True):
        '''
        Detect the acronym entity in the text
        '''
        res = self.get_bracket_entity(text)
        entity_list = []
        for entity in res[0]:
            longname = res[1][entity]['longform']
            if get_tag:
                derived_tag = self.umls_rules(longname)
            else:
                derived_tag = None
            entity_list.append((entity, longname, derived_tag))
        return entity_list

    def get_acronym_entity_by_tag(self, text, tag):
        '''
        Get the entity inside the bracket
        '''
        res = get_bracket_entity(text)
        print(res[0])
        entity_list = []
        for entity in res[1]:
            long_form = res[1][entity]['longform']
            out_tag = self.umls_rules(long_form)
            if out_tag == tag:
                entity_list.append((entity, long_form))
        return entity_list
