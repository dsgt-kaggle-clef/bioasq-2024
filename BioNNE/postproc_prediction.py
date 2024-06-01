import json
def convert_jsonl_to_single_entity_json(jsonl_file, output_file):
    '''
    Convert a jsonl file to a single entity json file
    '''
    with open(jsonl_file, 'r') as f:
        data = f.readlines()
    with open(output_file, 'w') as f:
        out_dict = {}
        for line in data:
            json_data = json.loads(line)
            out_dict[json_data['id']] = {
                'text': json_data['text']
            }
            for entity in json_data['entity']:
                out_entity_text = json_data['text'][entity[0]:entity[1]]
                out_entity_tag = entity[2]
                if entity[2] not in out_dict[json_data['id']]:
                    out_dict[json_data['id']][entity[2]] = []
                if out_entity_text not in out_dict[json_data['id']][entity[2]]:
                    out_dict[json_data['id']][entity[2]].append(out_entity_text)
        json.dump(out_dict, f, indent=2)

import helper_utils
ENTITY_LIST = ['DISO', 'FINDING', 'ANATOMY', 'PHYS', 'CHEM', 'LABPROC', 'INJURY_POISONING', 'DEVICE']
def calculate_outputfile_accuracy(golden_file, prediction_file):
    with open(golden_file, 'r') as f:
        golden_data = json.load(f)
    with open(prediction_file, 'r') as f:
        prediction_data = json.load(f)
    # Track tp, fp, fn of each entity type
    entity_type_dict = {}
    for entity_type in ENTITY_LIST:
            entity_type_dict[entity_type] = {'tp': 0, 'fp': 0, 'fn': 0}

    for pmid in prediction_data.keys():
        golden_ans = golden_data[pmid]
        prediction_ans = prediction_data[pmid]
        for entity_type in ENTITY_LIST:
            if entity_type not in golden_ans and entity_type not in prediction_ans:
                continue
            elif entity_type not in golden_ans and entity_type in prediction_ans:
                entity_type_dict[entity_type]['fp'] += len(prediction_ans[entity_type])
            elif entity_type in golden_ans and entity_type not in prediction_ans:
                entity_type_dict[entity_type]['fn'] += len(golden_ans[entity_type])
            else:
                tp, fp, fn = helper_utils.cnt_tp_fp_fn(golden_ans[entity_type], prediction_ans[entity_type])
                entity_type_dict[entity_type]['tp'] += tp
                entity_type_dict[entity_type]['fp'] += fp
                entity_type_dict[entity_type]['fn'] += fn
    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0
    for entity_type in ENTITY_LIST:
        prec, recall, f1 = helper_utils.calculate_precision_recall_f1(entity_type_dict[entity_type]['tp'], entity_type_dict[entity_type]['fp'], entity_type_dict[entity_type]['fn'], False)
        print(f'{entity_type} precision: {prec:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
        entity_type_dict[entity_type]['precision'] = prec
        entity_type_dict[entity_type]['recall'] = recall
        entity_type_dict[entity_type]['f1'] = f1
        total_precision += prec
        total_recall += recall
        total_f1 += f1

    # Calculate overall precision, recall, f1 by average the F1 score of each entity type
    total_precision /= len(ENTITY_LIST)
    total_recall /= len(ENTITY_LIST)
    total_f1 /= len(ENTITY_LIST)
    print(f'Average precision: {total_precision:.4f}, recall: {total_recall:.4f}, f1: {total_f1:.4f}')

def convert_llm_answer_to_resp(input_path, output_path):
    # Open the input jsonl file and assemble the output json dict with key as the id value of each row
    with open(input_path, 'r') as f:
        data = f.readlines()
    out_dict = {}
    for line in data:
        json_data = json.loads(line)
        out_dict[json_data['id']] = {}
        for entity in ENTITY_LIST:
            if entity in json_data:
                out_dict[json_data['id']][entity] = json_data[entity]
    # Write the output json dict to a json file
    with open(output_path, 'w') as f:
        json.dump(out_dict, f, indent=2)

def word_statistics():
    with open('processed_data/bionne/en/train.json', 'r') as f:
        data1 = json.load(f)
    with open('processed_data/bionne/en/dev.json', 'r') as f:
        data2 = json.load(f)
    word_dict = {}
    word_category_dict = {}
    for data in [data1]:
        for pmid in data.keys():
            for tag in ENTITY_LIST:
                if tag in data[pmid]:
                    for word in data[pmid][tag]:
                        if word not in word_dict:
                            word_dict[word] = 0
                            word_category_dict[word] = set()
                        word_dict[word] += 1
                        word_category_dict[word].add(tag)

    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    for k, v in word_dict.items():
        if v > 1:
            print(f'{k}: {v}, {word_category_dict[k]}')

if __name__ == '__main__':
    # convert_jsonl_to_single_entity_json('output/mixtral47_train_rule.jsonl', 'output/mixtral47_train_rule.json')
    # calculate_outputfile_accuracy('processed_data/bionne/en/train.json', 'output/mixtral47_train_rule.json')
    # convert_llm_answer_to_resp('output/mixtral47_train_rule_llm_answer.jsonl', 'mocks/mixtral47_train_rule_resp.json')
    word_statistics()