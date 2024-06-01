''' Metrics Calculation Code'''

import datetime
import json
import re
import time
from pathlib import Path
from pprint import pprint
from rouge import Rouge # Run: "pip install rouge". The version used is rouge-1.0.1
import helper_utils as helper

def cal_prec_recall_f1(tp, fp, fn):
    ''' Calculate precision, recall and f1 given tp, fp, fn'''
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def calculate_rouge(golden_text, ref_text):
    ''' Calculate the ROUGE-2 score for ideal answer'''
    rouge = Rouge(metrics=['rouge-2'])
    scores = rouge.get_scores(golden_text, ref_text)[0]['rouge-2']
    return scores

def calculate_ir_accuracy(golden_ques, output_ques):
    ''' Calculate the accuracy metrics of the IR task '''

    def _get_pmid_from_url(url):
        return url.split('/')[-1]

    assert(len(golden_ques) == len(output_ques))
    total_ques = len(golden_ques)
    doc_tt_prec, doc_tt_recall, doc_tt_f1 = 0.0, 0.0, 0.0
    snip_tt_prec, snip_tt_recall, snip_tt_f1 = 0.0, 0.0, 0.0

    for i in range (total_ques):
        assert(golden_ques[i]['id'] == output_ques[i]['id'])
        # 1. Calculate the accuracy of documents list
        golden_pmids = []
        predict_pmids = []
        for url in golden_ques[i]['documents']:
            golden_pmids.append(_get_pmid_from_url(url))
        for url in output_ques[i]['documents']:
            predict_pmids.append(_get_pmid_from_url(url))
        doc_fp, doc_fn, doc_tp = 0.0, 0.0, 0.0
        for pmid in golden_pmids:
            if pmid in predict_pmids:
                doc_tp += 1
            else:
                doc_fn += 1
        for pmid in predict_pmids:
            if pmid not in golden_pmids:
                doc_fp += 1
        # Unordered retrival measures (Precision, Recall, F1)
        doc_precision, doc_recall, doc_f1 = cal_prec_recall_f1(doc_tp, doc_fp, doc_fn)
        doc_tt_prec += doc_precision
        doc_tt_recall += doc_recall
        doc_tt_f1 += doc_f1
        # print('QID:', golden_ques[i]['id'], 'Doc Precision:', doc_precision, 'Doc Recall:', doc_recall, 'Doc F1:', doc_f1)

        # TODO: Document ordered retrival measures (MAP and GMAP)

        # 2. Calculate the accuracy of snippet list
        pred_snippet_cnt = len(output_ques[i]['snippets'])
        golden_snippet_cnt = len(golden_ques[i]['snippets'])
        overlap_cnt = 0.0
        for pred_snip in output_ques[i]['snippets']:
            snip_doc = _get_pmid_from_url(pred_snip['document'])
            # search for overlapping snippets in the golden doc
            for golden_snip in golden_ques[i]['snippets']:
                if (snip_doc == _get_pmid_from_url(golden_snip['document'])) and    \
                    (pred_snip['beginSection'] == golden_snip['beginSection']) and  \
                    (pred_snip['endSection'] == golden_snip['endSection']):
                    if (pred_snip['text'] in golden_snip['text']) or (golden_snip['text'] in pred_snip['text']):
                        overlap_cnt += 1
                        break
                    elif (pred_snip['offsetInBeginSection'] >= golden_snip['offsetInBeginSection'] and  \
                        pred_snip['offsetInBeginSection'] <= golden_snip['offsetInEndSection']) or      \
                        (pred_snip['offsetInEndSection'] >= golden_snip['offsetInBeginSection'] and     \
                        pred_snip['offsetInEndSection'] <= golden_snip['offsetInEndSection']):
                        overlap_cnt += 1
                        break
        snip_prec = overlap_cnt / pred_snippet_cnt if pred_snippet_cnt > 0 else 0
        snip_recall = overlap_cnt / golden_snippet_cnt if golden_snippet_cnt > 0 else 0
        snip_f1 = 2 * snip_prec * snip_recall / (snip_prec + snip_recall) if (snip_prec + snip_recall) > 0 else 0
        # print('QID:', golden_ques[i]['id'], 'Snippet Precision:', snip_prec, 'Snippet Recall:', snip_recall, 'Snippet F1:', snip_f1)
        # TODO: Snippet ordered retrival measures (MAP and GMAP)

        snip_tt_prec += snip_prec
        snip_tt_recall += snip_recall
        snip_tt_f1 += snip_f1

    print('\n## IR Metrics Calculation')

    doc_tt_prec /= total_ques
    doc_tt_recall /= total_ques
    doc_tt_f1 /= total_ques
    print(f'\nDocument mean precision: {doc_tt_prec:.4f}')
    print(f'Document recall: {doc_tt_recall:.4f}')
    print(f'Document f1: {doc_tt_f1:.4f}')

    snip_tt_prec /= total_ques
    snip_tt_recall /= total_ques
    snip_tt_f1 /= total_ques
    print(f'\nSnippet mean precision: {snip_tt_prec:.4f}')
    print(f'Snippet recall: {snip_tt_recall:.4f}')
    print(f'Snippet f1: {snip_tt_f1:.4f}')


def calculate_qa_accuracy(golden_ques, output_ques):
    ''' Calculate the accuracy of the QA task '''
    assert(len(golden_ques) == len(output_ques))

    yesno_tt, yesno_correct, yes_tp, yes_fn, yes_fp, no_tp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    factoid_tt, factoid_strict, factoid_lenient, factoid_mrr = 0.0, 0.0, 0.0, 0.0
    list_tt, list_precision, list_recall, list_f1 = 0.0, 0.0, 0.0, 0.0
    ideal_tt, ideal_precision, ideal_recall, ideal_f1 = 0.0, 0.0, 0.0, 0.0
    summary_tt, summary_precision, summary_recall, summary_f1 = 0.0, 0.0, 0.0, 0.0
    for i in range (len(golden_ques)):
        qid = golden_ques[i]['id']
        assert(golden_ques[i]['id'] == output_ques[i]['id'])
        qtype = golden_ques[i]['type']
        golden_exact = golden_ques[i]['exact_answer'] if 'exact_answer' in golden_ques[i] else ''
        golden_ideal = golden_ques[i]['ideal_answer'][0]
        output_exact = output_ques[i]['exact_answer']
        output_ideal = output_ques[i]['ideal_answer']
        if qtype == 'yesno':
            yesno_tt += 1
            yesno_correct += 1 if golden_exact == output_exact else 0
            # if golden_exact != output_exact:
                # print('QID:', golden_ques[i]['id'], 'Golden:', golden_exact, 'Output:', output_exact)
            # F1 yes and F1 no
            if golden_exact == 'yes':
                if output_exact == 'yes':
                    yes_tp += 1
                else:
                    yes_fn += 1
            else:
                if output_exact == 'yes':
                    yes_fp += 1
                else:
                    no_tp += 1
        elif qtype == 'factoid':
            factoid_tt += 1
            position = -1
            for idx, entry in enumerate(output_exact[:5]):
                for gold_syn in golden_exact[0]:
                    if entry[0].lower() == gold_syn.lower():
                        factoid_lenient += 1
                        if idx == 0:
                            factoid_strict += 1
                        position = idx
                        break
                if position != -1:
                    break
            factoid_mrr += 1.0/(position+1) if position != -1 else 0
            # print('QID:', golden_ques[i]['id'], 'Position:', position)
            # print('\tGolden:', golden_exact[0], 'Output:', output_exact)
        elif qtype == 'list':
            list_tt += 1
            # Pairing each output set with the golden set index
            golden_len = len(golden_exact)
            output_flatten = [item for sublist in output_exact for item in sublist]
            golden_flatten = [item for sublist in golden_exact for item in sublist]
            golden_selected_set = set()
            for idx, golden_list in enumerate(golden_exact):
                for golden_item in golden_list:
                    if helper.has_word_in_list(golden_item, output_flatten):
                        golden_selected_set.add(idx)
            lfp = 0
            list_fp_items = []
            list_fn_items = []
            list_tp_items = []
            for item in output_flatten:
                if not helper.has_word_in_list(item, golden_flatten):
                    lfp += 1
                    list_fp_items.append(item)
                else:
                    list_tp_items.append(item)
            
            for gold_ind in range(golden_len):
                if gold_ind not in golden_selected_set:
                    list_fn_items.append(golden_exact[gold_ind][0])

            ltp = len(golden_selected_set)
            lfn = golden_len - ltp
            lprecision, lrecall, lf1 = cal_prec_recall_f1(ltp, lfp, lfn)
            list_precision += lprecision 
            list_recall +=  lrecall
            list_f1 += lf1
            # print('\nQID:', qid, 'Precision:', lprecision, 'Recall:', lrecall, 'F1:', lf1)
            # print('Golden:', golden_flatten)
            # print('Output:', output_flatten)
            # print('TP:', list_tp_items)
            # print('FP:', list_fp_items)
            # print('FN:', list_fn_items)

        if output_ideal == '':
            ideal_rouge = {'p': 0.0, 'r': 0.0, 'f': 0.0}
        else:
            ideal_rouge = calculate_rouge(golden_ideal, output_ideal)
        ideal_tt += 1
        ideal_precision += ideal_rouge['p']
        ideal_recall += ideal_rouge['r']
        ideal_f1 += ideal_rouge['f']

        if qtype == 'summary':
            summary_tt += 1
            summary_precision += ideal_rouge['p']
            summary_recall += ideal_rouge['r']
            summary_f1 += ideal_rouge['f']

        # print()
        # print(golden_ideal)
        # print(output_ideal)
        # print('Ideal rouge2: ', ideal_rouge)

    print('\n## QA Metrics Calculation')

    if yesno_tt > 0:
        # Yesno metrics
        yesno_acc = yesno_correct/yesno_tt
        yesno_f1yes = cal_prec_recall_f1(yes_tp, yes_fp, yes_fn)
        yesno_f1no = cal_prec_recall_f1(no_tp, yes_fn, yes_fp)
        yesno_macro_f1 = (yesno_f1yes[2] + yesno_f1no[2]) / 2
        print()
        print(f'Yesno accuracy: {yesno_acc:.4f}')
        print(f'Yesno yes precision, recall, f1: {yesno_f1yes[0]:.4f}, {yesno_f1yes[1]:.4f}, {yesno_f1yes[2]:.4f}')
        print(f'Yesno no precision, recall, f1: {yesno_f1no[0]:.4f}, {yesno_f1no[1]:.4f}, {yesno_f1no[2]:.4f}')
        print(f'Yesno macro-average f1: {yesno_macro_f1:.4f}')

    if factoid_tt > 0:
        factoid_strict_acc = factoid_strict/factoid_tt
        factoid_lenient_acc = factoid_lenient/factoid_tt
        MRR = factoid_mrr/factoid_tt
        print()
        print(f'Factoid strict accuracy: {factoid_strict_acc:.4f}')
        print(f'Factoid lenient accuracy: {factoid_lenient_acc:.4f}')
        print(f'Factoid MRR: {MRR:.4f}')
    
    if list_tt > 0:
        list_precision /= list_tt
        list_recall /= list_tt
        list_f1 /= list_tt
        print()
        print(f'List precision: {list_precision:.4f}')
        print(f'List recall: {list_recall:.4f}')
        print(f'List f1: {list_f1:.4f}')
    
    if summary_tt > 0:
        summary_precision /= summary_tt
        summary_recall /= summary_tt
        summary_f1 /= summary_tt
        print()
        print(f'Summary precision: {summary_precision:.4f}')
        print(f'Summary recall: {summary_recall:.4f}')
        print(f'Summary f1: {summary_f1:.4f}')

    if ideal_tt > 0:
        ideal_precision /= ideal_tt
        ideal_recall /= ideal_tt
        ideal_f1 /= ideal_tt
        print()
        print(f'Ideal precision: {ideal_precision:.4f}')
        print(f'Ideal recall: {ideal_recall:.4f}')
        print(f'Ideal f1: {ideal_f1:.4f}')