import json

# extract question from final answer
def build_question_from_answer(input_file='small_answer.json', output_file='small_question.json'):
    with open(input_file, 'r') as f:
        file_content = f.read()
    all_answers = json.loads(file_content)
    all_answers = all_answers["questions"]
    res = []
    for ans_obj in all_answers:
        obj = {
                "type": ans_obj["type"],
                "body": ans_obj["body"],
                "id": ans_obj["id"],
                "answerReady": True
               }
        res.append(obj)
    res = {"questions": res}
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=2)

    print("Answer file created")
    return

# extract document and snippet from final answer
def build_snippet_from_answer(input_file='small_answer.json', output_file='small_snippet.json'):
    with open(input_file, 'r') as f:
        file_content = f.read()
    all_answers = json.loads(file_content)
    all_answers = all_answers["questions"]
    res = []
    for ans_obj in all_answers:
        obj = {
                "type": ans_obj["type"],
                "body": ans_obj["body"],
                "id": ans_obj["id"],
                "answerReady": True,
                "documents": ans_obj["documents"],
                "snippets": ans_obj["snippets"]
               }
        res.append(obj)
    res = {"questions": res}
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=2)

    print("Snippet file created")
    return

QUES_TYPES = ['factoid', 'list', 'yesno', 'summary']
def check11b():
    type_cnt = {}
    reserved_data = [
        '54e25eaaae9738404b000017', '51757bbb8ed59a060a00002e',
        '55031181e9bde69634000014', '532624ae600967d132000005',
        '513ce3c8bee46bd34c000008', '52bf202003868f1b06000018',
        '552fac4fbc4f83e828000006', '56bc751eac7ad10019000013']
    for qtype in QUES_TYPES:
        type_cnt[qtype] = 0
    for i in range(1, 5):
        for qtype in QUES_TYPES:
            type_cnt[qtype] = 0
        input_file = f'11B{i}_golden.json'
        with open(input_file, 'r') as f:
            file_content = f.read()
        all_answers = json.loads(file_content)["questions"]
        for ans_obj in all_answers:
            # print(ans_obj["id"])
            if ans_obj["id"] in reserved_data:
                print('found reserved data', ans_obj["id"])
                print(ans_obj["type"])
            type_cnt[ans_obj["type"]] += 1
        print('-'*80 + '\n')
        print(input_file)
        print(type_cnt)

if __name__ == '__main__':
    build_snippet_from_answer()
    build_question_from_answer()
    # check11b()