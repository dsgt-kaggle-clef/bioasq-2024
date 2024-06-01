# TextSynth text completion.

import sys
import requests
import json
import time
import datetime
from tokens import * 

URL = 'https://api.textsynth.com'
llm_log_file = open('logs/llm_history.txt', "a")

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
        sys.exit(1)
    return response.json()

def textsynth_completion(model, query_param, task, fake_llm=True):
    if fake_llm:
        result = {}
        if task == 'QA':
            result['text'] = 'Hubris is characterized by arrogance and pride, which is seen in the person who is in the position of power, has extreme knowledge about the subject and shows narcissism and egocentrism.'
            if ('qtype' in query_param):
                if query_param['qtype'] == 'yesno':
                    result['text'] = 'Yes. Systematic NGS should be performed for patients with advanced colorectal cancer to identify actionable molecular targets for treatment.\nExact answer: yes\n'
                elif query_param['qtype'] == 'list':
                    result['text'] = 'Folate, fruit and vegetables are potential protective factors for the prevention of colorectal cancer.\nExact answer: [["folate"], ["fruit and vegetables"]]\n'
                elif query_param['qtype'] == 'factoid':
                    result['text'] =  'Detection of EWS/FLI1 is a major clinical challenge due to its heterogeneity and low expression level in clinical samples. It is mainly detected by RT-PCR, qRT-PCR, FISH and RT-nASA-PCR.\nExact answer: RT-nASA-PCR\n'
            result['reached_end'] = True
            result['input_tokens'] = 63
            result['output_tokens'] = 3
            return result
        elif task == 'IR':
            result['text'] = 'Duchenne Muscular Dystrophy, AI methods'
            result['reached_end'] = True
            result['input_tokens'] = 23
            result['output_tokens'] = 3
            return result
        else:
            assert(not 'Unrecognized task type')

    if 'qtype' in query_param:
        del query_param['qtype']
    return make_textsynth_request(f"/v1/engines/{model}/completions", query_param)
