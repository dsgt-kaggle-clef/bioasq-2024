''' Unit tests for helper methods '''

import model
from pprint import pprint
import search_utils

def test_exact_answer_regex():
    tests = ['    yes', 'YES,', 'Yes', 'yes', 'No', 'no', 'no, ', 'Nobody knows','', 'random stuff']
    result = ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', '', '']
    for idx, test in enumerate(tests):
        res = model.format_exact_answer(test, 'yesno')
        exp_res = result[idx]
        assert res == result[idx], f'expect {exp_res} for {test}, but get {res}'

def test_factoid_answer_parser():
    tests = [' [["folate"], ["fruit and vegetables"]]\n',
            '[["folate"]]',
             'fruit and vegetables',
"""1. apple
2. pears
3. pineapple
"""
             ]
    expected_result = [
        [["folate"], ["fruit and vegetables"]],
        [['folate']],
        [['fruit and vegetables']],
        [['1. apple'], ['2. pears'], ['3. pineapple']]
    ]
    for idx, test in enumerate(tests):
        res = model.format_exact_answer(test, 'factoid')
        print('Test:', test)
        print('Res:', res)
        exp_res = expected_result[idx]
        assert res == exp_res, f'expect {exp_res} for {test}, but get {res}'

def process_malform_json():
    expect_jsonout = [["Dasatinib"], ["Quercetin"], ["Fisetin"]]
    json_in1 = "[[\"Dasatinib\"], [\"Quercetin\"], [\"Fisetin"
    json_out1 = model.format_exact_answer(json_in1, 'list')
    assert(expect_jsonout == json_out1)

    json_in2 = "[\"Dasatinib\", \"Quercetin\", \"Fisetin\""
    json_out2 = model.format_exact_answer(json_in2, 'list')
    assert(expect_jsonout == json_out2)

    json_in3 = "[[\"Dasatinib\"], [\"Quercetin\"], [\"Fisetin\"]"
    json_out3 = model.format_exact_answer(json_in3, 'list')
    assert(expect_jsonout == json_out3)

    llm_resp ="""
a) Senolytics cause cell death of senescent cells by selectively stimulating the apoptosis and necrosis pathways in senescent cells and inhibiting their survival pathways.
b) Senomorphics suppress the senescence phenotypes but don’t kill senescent cells.
c) Some senolytics have been FDA approved for other diseases in the past.
d) A number of senomorphics are in clinical trials to treat age-related diseases.
e) Senomorphics show great promise for the treatment of age-related diseases, but their mechanism of action is less clear.
Exact answer: [["senolytics", ["senolytics kill SCs selectively"], ["senomorphics", ["suppress the senescence phenotypes"], ["senolytics approved for other diseases"], ["senomorphics", ["number of senomorphics in clinical trials"], ["senomorphics mechanism of
"""
    llm_out = model.process_llm_response(llm_resp, 'list')
    assert (llm_out['exact'] == [['senolytics'], ['senolytics kill SCs selectively'], ['senomorphics'], ['suppress the senescence phenotypes'], ['senolytics approved for other diseases'], ['senomorphics'], ['number of senomorphics in clinical trials'], ['senomorphics mechanism of']])

def test_xml_parser():
    output = search_utils.query_by_pmids(['38340534'], verbose=True)
    assert('BACKGROUND: Few population-based data sources fully recognise' in output[0]['abstract_raw'])
    # TODO: unable to correctly parse
    # https://pubmed.ncbi.nlm.nih.gov/38355787/

def test_pubmed_api():
    # id = ['11965461']
    # id = ['31375923']
    # id = ['38353645']
    pmid = ['24286700']
    sentence = 'The algorithm provides excellent discrimination of PD patients from PSP patients at an individual level, thus encouraging the application of computer-based diagnosis in clinical practice.'
    length = len(sentence)
    out = search_utils.query_by_pmids(pmid, verbose=True)
    extract = out[0]['abstract_raw']
    start_ind = extract.find(sentence)
    end_ind = start_ind + length
    assert(start_ind == 1588)
    assert(end_ind == 1775)
    assert(extract.find('The FDA awarded conditional approval to etiplirsen, an') == -1)

def execute_all_tests():
    test_exact_answer_regex()
    test_factoid_answer_parser()
    process_malform_json()
    test_xml_parser()
    test_pubmed_api()

if __name__ == '__main__':
    execute_all_tests()
    
