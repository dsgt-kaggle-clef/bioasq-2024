import os
import json
import re
from pathlib import Path
from pprint import pprint
import requests
import time
import datetime
from tokens import * 

def preprocess_raw_data():
    split = 'dev' # train
    lang = 'en' # ru
    is_english = True if lang == 'en' else False

    nerel_path = f'~/CLEF/DATASET_BIONNE/{lang}/{split}'
    output_path = f'processed_data/bionne/{lang}/{split}.json'
    entries = os.listdir(nerel_path)
    txt_entries = [item for item in entries if not item.endswith('.ann')]
    txt_entries.sort()
    tags = ["ANATOMY", "CHEM", "DEVICE", "DISO", "FINDING", "INJURY_POISONING", "LABPROC", "PHYS"]
    res_dict = {}
    for txt_entry in txt_entries:
        txt_id = txt_entry[0:-4]
        ann_entry = txt_id + '.ann'
        txt = Path(nerel_path+f'/{txt_entry}').read_text()
        ann = Path(nerel_path+f'/{ann_entry}').read_text()
        res_dict[txt_id] = {'text': txt}
        for tag in tags:
            filtered_label=re.findall(f"(^T\w+).*({tag}).*\t(.+)$", ann, re.MULTILINE)
            if len(filtered_label) > 0:
                entry_repeat = set()
                entry_list = []
                for label in filtered_label:
                    entity = label[2]
                    lower_cased_entity = entity.lower()
                    if not (lower_cased_entity in entry_repeat):
                        entry_repeat.add(lower_cased_entity)
                        entry_list.append(entity)
                res_dict[txt_id][tag] = entry_list

    with open(output_path, "w", encoding='utf-8') as write_file:
        json.dump(res_dict, write_file, ensure_ascii=is_english, indent=2)
    print("Done writing JSON")

    with open(output_path, "r", encoding='utf-8') as read_file:
        sampleData = json.load(read_file)
        pprint(sampleData)

def preproce_test_data():
    lang = 'en'
    is_english = True if lang == 'en' else False
    bionne_path = f'~/CLEF/DATASET_BIONNE/test/{lang}'
    output_path = f'processed_data/bionne/{lang}/test.json'
    entries = os.listdir(bionne_path)
    txt_entries = [item for item in entries if not item.endswith('.ann')]
    txt_entries.sort()
    res_dict = {}
    for txt_entry in txt_entries:
        txt_id = txt_entry[0:-4]
        txt = Path(bionne_path+f'/{txt_entry}').read_text()
        res_dict[txt_id] = {'text': txt}
    with open(output_path, "w", encoding='utf-8') as write_file:
        json.dump(res_dict, write_file, ensure_ascii=is_english, indent=2)
    print("Done writing JSON")


def check_russian_dataset(category):
    ''' Findings for nerel:
    1. The multi-word entity phrase can contain comma, but no semicolon
    2. FINDING entity can contain many multi-words phrases
    3. The longest entites contains up to 18 words separated by space
    '''
    # input_path = f'processed_data/nerel-bio/{category}.json'
    input_path = f'processed_data/bionne/ru/{category}.json'
    with open(input_path, 'r', encoding='utf-8') as read_file:
        russian_json = json.load(read_file)
    has_comma = 0
    has_semicolon = 0
    max_word_cnt = 0
    cnt = 0
    for tid, tval in russian_json.items():
        cnt += 1
        for tag, en_list in tval.items():
            if tag != 'text':
                # Check no ,
                for ele in en_list:
                    if ',' in ele:
                        has_comma += 1
                        print(ele)
                    if ';' in ele:
                        has_semicolon += 1
                    word_cnt = len(ele.split())
                    max_word_cnt = max(word_cnt, max_word_cnt)
    print('Count contains comma: ', has_comma)
    print('Count contains semicolon: ', has_semicolon)
    print('Max word count: ', max_word_cnt)
    print('Total record: ', cnt)

URL = 'https://api.textsynth.com'
llm_log_file = open('logs/llm_nne_translation_history.txt', "a")
def make_textsynth_request(path, query_param):
    response = requests.post(URL + path,
        headers = { "Authorization": f"Bearer {TEXTSYNTH_TOKEN}"},
        json = query_param)
    if llm_log_file:
        llm_log_file.write(f"\n# {datetime.datetime.now()}\n")
        llm_log_file.write(f'{URL + path}\n')
        json.dump(query_param, llm_log_file, indent=2)
        llm_log_file.write('\nResponse\n')
        json.dump(response.json(), llm_log_file, indent=2)
    if response.status_code != 200:
        print("Request error:", response.text)
    return response.json()

def convert_russian_to_english():
    category = 'dev'
    # category = 'test'
    # category = 'train'
    input_path = f'processed_data/nerel-bio/{category}.json'
    output_path = f'processed_data/nerel-bio/{category}_eng.json'
    with open(input_path, "r", encoding='utf-8') as read_file:
        russian_json = json.load(read_file)
    out_json = {}
    cnt = 0
    for tid, tval in russian_json.items():
        cnt += 1
        print(tval['text'])
        query_param = {
            'text': [tval['text']],
            'source_lang': 'ru',
            'target_lang': 'en'
        }
        resp = make_textsynth_request('/v1/engines/madlad400_7B/translate', query_param)
        # text_translation = 'To directly assess the impact of smoking on causal mortality'
        text_translation = resp['translations'][0]['text']
        print(text_translation)
        out_json[tid] = {'text': text_translation}
        for tag, en_list in tval.items():
            if tag != 'text':
                tag_str = ';'.join(en_list)
                print(tag, tag_str)
                query_param['text'] = [tag_str]
                answer = make_textsynth_request('/v1/engines/madlad400_7B/translate', query_param)
                # entity_translation = 'mortality;age;life expectancy;deaths;mortality;age of smoking start;life;deaths'
                entity_translation = answer['translations'][0]['text']
                print(entity_translation)
                out_json[tid][tag] = entity_translation.split(';')
    with open(output_path, "w", encoding='utf-8') as write_file:
        json.dump(out_json, write_file, indent=2)
    print('Count', cnt)


if __name__ == '__main__':
    # Step 1: convert the dataset into json format
    # preprocess_raw_data()
    # Step 2: translate Russian into English
    # convert_russian_to_english()
    # Step 3: feed English examples for few shot learning

    # Dataset observations
    # category = ['dev', 'train']
    # for cate in category:
    #     check_russian_dataset(cate)

    preproce_test_data()
