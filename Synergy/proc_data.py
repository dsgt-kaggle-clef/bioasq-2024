''' Code to analyze golden answer and feedback files'''

import model
import json
import search_utils
import helper_utils as helper
from pprint import pprint
import torch
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

def overview_test_feedback():
    testsets = [
        ['syn24_testset_3.json', 'syn24_feedback_3.json'],
        ['syn24_testset_4.json', 'syn24_feedback_4.json']
    ]

    testset = testsets[1]
    data, feedback = testset

    print(data, feedback)
    ques_list1 = helper.parse_input(f'input/{data}')
    ques_list2 = helper.parse_input(f'input/{feedback}')
    qid_list1 = list()
    qid_list2 = list()
    answerReady_list1 = []
    answerReady_list2 = []
    for ques in ques_list1:
        qid_list1.append(ques['id'])
        if ques['answerReady']:
            answerReady_list1.append(ques['id'])

    for ques in ques_list2:
        qid_list2.append(ques['id'])
        if ques['answerReady']:
            answerReady_list2.append(ques['id'])

    list2_exc = []
    for qid in qid_list1:
        if qid not in qid_list2:
            list2_exc.append(qid)

    answerReadylist2_exc = []
    for qid in answerReady_list1:
        if qid not in answerReady_list2:
            answerReadylist2_exc.append(qid)

    list1_exc = []
    for qid in qid_list2:
        if qid not in qid_list1:
            list1_exc.append(qid)

    answerReadylist1_exc = []
    for qid in answerReady_list2:
        if qid not in answerReady_list1:
            answerReadylist1_exc.append(qid)

    print(f'Testset record: {len(qid_list1)}')
    print(f'Feedback record: {len(qid_list2)}')
    print(f'In testset, not in feedback set: {list2_exc}')
    print(f'In feedback, not in testset: {list1_exc}')
    print("\n" + "-"*80 + "\n")
    print(f'Answer ready: test {len(answerReady_list1)}, feedback {len(answerReady_list2)}')
    print(f'Answer ready In testset, not in feedback set: {answerReadylist2_exc}')
    print(f'Answer ready In feedback, not in test set: {answerReadylist1_exc}')
    print("\n" + "-"*80 + "\n")
    print(f'answer ready quesitons:')
    pprint(answerReady_list1)

def aggregate_feedback_files():
    feedback_files = ['syn24_feedback_1.json', 'syn24_feedback_2.json', 'syn24_feedback_3.json']
    # feedback_files = ['syn24_feedback_1.json']
    output_file = open('input/aggregate_feedback_123.json', 'w')
    output = {} # key=id, value = {body, document_list}
    qcnt, rqcnt = 0, 0
    dcnt = 0
    for ffile in feedback_files:
        ques_list = helper.parse_input(f'input/{ffile}')
        for ques in ques_list:
            qid = ques['id']
            if qid not in output:
                output[qid] = {}
                output[qid]['body'] = ques['body']
                output[qid]['documents'] = []
                qcnt += 1
            else:
                rqcnt += 1
            for doc in ques['documents']:
                if doc['id'] not in output[qid]['documents']:
                    dcnt += 1
                    output[qid]['documents'].append(doc['id'])
    json.dump(output, output_file, indent=2, sort_keys=False)
    print(f'Question count {qcnt}. Repeat Question count {rqcnt}. Document count {dcnt}')
    return output

def validate_output():
    path = 'experiment_data/res_syn24_testset_3_1707206195.json'
    ques_list = helper.parse_input(path)
    print(len(ques_list))

def find_repeat_question():
    testset2_list = helper.parse_input('input/syn24_testset_3.json')
    testset3_list = helper.parse_input('input/syn24_testset_4.json')
    feedback_list = helper.parse_input('input/syn24_feedback_4.json')
    output_file = open('output/repeat_ques34', 'w')
    ques_dict = {}
    common_list = []
    for ques in testset2_list:
        ques_dict[ques['id']] = {'body': ques['body'], 'testset': [2]}

    for ques in testset3_list:
        ques_id = ques['id']
        if ques_id not in ques_dict:
            ques_dict[ques_id] = {'body': ques['body'], 'testset': [3]}
        else:
            ques_dict[ques_id]['testset'].append(3)
            common_list.append(ques_id)
    print(f'There are test2_ques={len(testset2_list)} test3_ques={len(testset3_list)}. There are {len(common_list)} questions in common')

    for ques in feedback_list:
        qid = ques['id']
        if qid in ques_dict:
            ques_dict[qid]['pos_docs'] = []
            ques_dict[qid]['neg_docs'] = []
            for doc in ques['documents']:
                if doc['golden']:
                    ques_dict[qid]['pos_docs'].append(doc['id'])
                else:
                    ques_dict[qid]['neg_docs'].append(doc['id'])

            ques_dict[qid]['pos_snippets'] = []
            ques_dict[qid]['neg_snippets'] = []
            for snip in ques['snippets']:
                if snip['golden']:
                    ques_dict[qid]['pos_snippets'].append(snip['text'])
                else:
                    ques_dict[qid]['neg_snippets'].append(snip['text'])

    no_pos_list = []
    for k in ques_dict:
        if ('pos_snippets' not in ques_dict[k]) or (len(ques_dict[k]['pos_snippets']) == 0):
            no_pos_list.append(k)
    print('no_pos_list', no_pos_list)
    json.dump(ques_dict, output_file, indent=2, sort_keys=False)

def load_sentence_transformer():
    '''
    Usage: 
        st_model = load_sentence_transformer()
        embed_snippets(st_model)
        embed_golden_doc(st_model)
    '''
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_golden_doc(st_model):
    print(f'max sequence: {st_model.max_seq_length}')
    path = 'input/syn23_golden_1.json'
    ques_list = helper.parse_input(path)
    for ques in ques_list[:1]:
        print(ques['body'])
        for snip in ques['snippets']:
            doc_id = snip['document']
            print(snip['document'])
            print(snip['text'])
            print("-"*80)
            out = search_utils.query_by_pmids([doc_id], verbose=False)
            abstract = out[0]['abstract_raw']
            sentences = abstract.split(". ")
            pprint(sentences)
            print("-"*80)
            helper.get_top_results(st_model, ques['body'], sentences)

def embed_and_rank_snip(st_model, ques_body, pos_snip, neg_snip):
    print("-"*80)
    print('Positive snippet ranking')
    helper.get_top_results(st_model, ques_body, pos_snip)
    print("-"*80)
    print('Negitive snippet ranking')
    helper.get_top_results(st_model, ques_body, neg_snip)

def embed_snippets(st_model):
    inpath = 'output/repeat_ques23'
    ques_dict = json.loads(Path(inpath).read_text())
    for k in ques_dict:
        qbody = ques_dict[k]['body']
        print("-"*80)
        print(f'Question: {qbody}')
        embed_and_rank_snip(st_model, ques_dict[k]['body'], ques_dict[k]['pos_snippets'], ques_dict[k]['neg_snippets'])

if __name__ == '__main__':
    overview_test_feedback()
    print("\n" + "-"*80 + "\n")
    find_repeat_question()
