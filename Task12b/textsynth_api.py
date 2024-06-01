# TextSynth text completion.

import sys
import requests
import json
import time
import datetime
from tokens import * 
from prompt_utils import query_expansion_prompt, synonym_grouping_prompt
import re
from openai import OpenAI
from dotenv import load_dotenv
from ast import literal_eval
import config

load_dotenv()

URL = 'https://api.textsynth.com'
PROXY_CNT = 0
QA_PROXY = {}
with open(f'cache/{config.QA_PROXY_CACHE_PATH}', "r") as proxy_cache_file:
    QA_PROXY = json.load(proxy_cache_file)

def use_llm_proxy(orig_question):
    result = {}
    qtype = orig_question.type
    qid = orig_question.id

    result['text'] = 'Hubris is characterized by arrogance and pride, which is seen in the person who is in the position of power, has extreme knowledge about the subject and shows narcissism and egocentrism.'
    if qtype == 'yesno':
        result['text'] = 'Yes. Systematic NGS should be performed for patients with advanced colorectal cancer to identify actionable molecular targets for treatment.\nExact answer: yes\n'
    elif qtype == 'list':
        result['text'] = 'Folate, fruit and vegetables are potential protective factors for the prevention of colorectal cancer.\nExact answer: folate;fruit and vegetables\n'
    elif qtype == 'factoid':
        result['text'] = 'Detection of EWS/FLI1 is a major clinical challenge due to its heterogeneity and low expression level in clinical samples. It is mainly detected by RT-PCR, qRT-PCR, FISH and RT-nASA-PCR.\nExact answer: RT-nASA-PCR\n'
    if qid in QA_PROXY:
        potential_ans_sz = len(QA_PROXY[qid])
        global PROXY_CNT
        result['text'] = QA_PROXY[qid][PROXY_CNT % potential_ans_sz]
        PROXY_CNT += 1
    result['reached_end'] = True
    result['input_tokens'] = 63
    result['output_tokens'] = 3
    result['finish_reason'] = 'length'
    return result

def make_textsynth_request(model, query_param):
    path = f"/v1/engines/{model}/completions"
    response = requests.post(URL + path,
        headers = { "Authorization": f"Bearer {TEXTSYNTH_TOKEN}"},
        json = query_param)
    
    with open('logs/llm_history.txt', "a") as llm_log_file:
        llm_log_file.write(f"\n# {datetime.datetime.now()}\n")
        llm_log_file.write(f'{URL + path}\n')
        json.dump(query_param, llm_log_file, indent=2)
        llm_log_file.write('\nResponse\n')
        json.dump(response.json(), llm_log_file, indent=2)
        
    if response.status_code != 200:
        print("Request error:", response.text)
        sys.exit(1)
    return response.json()


def textsynth_completion(model, query_param, task, fake_llm=True, orig_question=None):
    if fake_llm:
        result = {}
        if task == 'QA':
            result = use_llm_proxy(orig_question)
            return result
        elif task == 'IR':
            result['text'] = 'Duchenne Muscular Dystrophy, AI methods'
            result['reached_end'] = True
            result['input_tokens'] = 23
            result['output_tokens'] = 3
            return result
        else:
            assert(not 'Unrecognized task type')

    return make_textsynth_request(model, query_param)

def group_synonym_entity(entity_str):
    '''Group the synonyms in the entity string together'''
    query_param = {
        'prompt': synonym_grouping_prompt.format(entity_list=entity_str),
        'max_tokens': 200,
        'stop': '###'
    }
    return make_textsynth_request('mixtral_47B_instruct', query_param)

def query_expansion(question, fake_llm=True, model_name="MIXTRAL_47B"):
    if fake_llm:
        return question # We just let the model query the original question
    
    if model_name == 'MIXTRAL_47B':
        model = "mixtral_47B_instruct"
        prompt = query_expansion_prompt.format(body=question)
        query_param = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0,
            "stop": "###"
        }
        response = textsynth_completion(model, query_param, 'IR', fake_llm=False)
        
        return response['text']
    else:
        # use GPT
        client = OpenAI()
        model_name = config.OPENAI_MODEL_NAME
        params_chat = {
            "model": model_name, # model name
            "messages": [
                {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."},
                {"role": "user", "content": f"Expand this search query: '{question}' for PubMed by incorporating synonyms and additional terms that closely relate to \
                the main topic and help reduce ambiguity. Assume that phrases are not stemmed; therefore, generate useful variations. Return only the query that can \
                directly be used without any explanation text. Focus on maintaining the query's precision and relevance to the original question."}
            ], 
            "temperature": 0.0, # randomness of completion
            "frequency_penalty": 0.5, # discourage repetition of words or phrases
            "presence_penalty": 0.1, # discourage new topics or entities
        }
        response = client.chat.completions.create(
            **params_chat
        )
        
        return response.choices[0].message.content

def query_expansion_retry(question, original_query, fake_llm):
    if fake_llm:
        return question # We just let the model query the original question
    
    # use GPT
    client = OpenAI()
    model_name = config.OPENAI_MODEL_NAME

    params_chat = {
        "model": model_name, 
        "messages": [
            {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, \
            research, and information retrieval in the biomedical domain."},
            {"role": "user", "content": f"Given that the following search query for PubMed has returned\
            no documents, please generate a broader query that retains the original question's context and relevance.\
            Assume that phrases are not stemmed; therefore, generate useful variations. Return only the query that can\
            directly be used without any explanation text. Focus on maintaining the query's precision and relevance to\
            the original question. Original question: '{question}', Original query: '{original_query}'."}
        ],
        "temperature": 0.0, # randomness of completion
        "frequency_penalty": 0.6, # discourage repetition of words or phrases
        "presence_penalty": 0.2, # discourage new topics or entities
    }
    
    response = client.chat.completions.create(
        **params_chat
    )
    
    return response.choices[0].message.content

def query_expansion_last_try(question, fake_llm):
    if fake_llm:
        return question # We just let the model query the original question
    
      # use GPT
    client = OpenAI()
    model_name = config.OPENAI_MODEL_NAME

    params_chat = {
        "model": model_name, 
        "messages": [
            {"role": "user", "content": f"""Identify the medical entities in this question.
             If multiple entities are present, order the capitalized entities or entities with acronyms or entities with higher importance first.
             Do not use internet browsing tool. Reply with a list of medical entities, format your response as a valid JSON.

            Question: Concizumab is used for which diseases?
            Answer: ["Concizumab"]

            Question: {question}
            Answer: """}
        ],
        "temperature": 0.0, # randomness of completion
        "frequency_penalty": 0.6, # discourage repetition of words or phrases
        "presence_penalty": 0.2, # discourage new topics or entities
    }
    
    response = client.chat.completions.create(
        **params_chat
    )
    
    res = response.choices[0].message.content
    try:
        l = literal_eval(res)
        if isinstance(l, list):
            return ' OR '.join(l)
        # add another case checking if l is an object, if so return the value of the first attribute of that object
        elif isinstance(l, dict) and len(l) > 0:
            res = l[list(l.keys())[0]]
            if isinstance(res, list):
                return ' OR '.join(res)
            else:
                return str(res)
        else:
            # raise an exception to catch it in the except block
            return str(res)
    except:
        return str(res)
