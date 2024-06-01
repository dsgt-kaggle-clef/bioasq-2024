import metrics
import helper_utils
import qa_module

def test_calculate_qa_accuracy_yesno():
    golden_ques = [
        {'id': 1, 'type': 'yesno', 'exact_answer': 'yes', 'ideal_answer': ['yes, it is']},
        {'id': 2, 'type': 'yesno', 'exact_answer': 'yes', 'ideal_answer': ['yes it is']},
        {'id': 3, 'type': 'yesno', 'exact_answer': 'no', 'ideal_answer': ['no it is not']}
    ]
    output_ques = [
        {'id': 1, 'type': 'yesno', 'exact_answer': 'yes', 'ideal_answer': 'it is yes'},
        {'id': 2, 'type': 'yesno', 'exact_answer': 'no', 'ideal_answer': 'yes it is'},
        {'id': 3, 'type': 'yesno', 'exact_answer': 'yes', 'ideal_answer': 'it is not'}
    ]
    metrics.calculate_qa_accuracy(golden_ques, output_ques)

def test_calculate_qa_accuracy_factoid():
    golden_ques = [
        {'id': 1, 'type': 'factoid', 'exact_answer': [['a']],
            'ideal_answer': ["Mitapivat, an oral activator of pyruvate kinase in red blood cells"]},
        {'id': 2, 'type': 'factoid', 'exact_answer': [['a']],
            'ideal_answer': ["Mitapivat, an oral activator of pyruvate kinase in red blood cells"]},
        {'id': 3, 'type': 'factoid', 'exact_answer': [['d']],
            'ideal_answer': ["Mitapivat, an oral activator of pyruvate kinase in red blood cells"]}
    ]
    output_ques = [
        {'id': 1, 'type': 'factoid', 'exact_answer': [['a'],['c']], 'ideal_answer': "Mitapivat, an oral activator of pyruvate kinase in red blood cells"},
        {'id': 2, 'type': 'factoid', 'exact_answer': [['b'],['a']], 'ideal_answer': "Mitapivat, an oral activator of pyruvate kinase in red blood cells"},
        {'id': 3, 'type': 'factoid', 'exact_answer': [['d']], 'ideal_answer': 'oral activator of pyruvate kinase'}
    ]
    metrics.calculate_qa_accuracy(golden_ques, output_ques)

def test_calculate_qa_accuracy_list():
    golden_ques = [
        {'id': 1, 'type': 'list', 'exact_answer': [['a'],['c']], 'ideal_answer': ['hello yes are you there']},
        {'id': 2, 'type': 'list', 'exact_answer': [['a'],['b']], 'ideal_answer': ['hello yes']},
        {'id': 3, 'type': 'list', 'exact_answer': [['d']], 'ideal_answer': ['yes']}
    ]
    output_ques1 = [
        {'id': 1, 'type': 'list', 'exact_answer': [['a'],['c']], 'ideal_answer': 'hello yes are you there'},
        {'id': 2, 'type': 'list', 'exact_answer': [['b'],['a']], 'ideal_answer': 'yes hello'},
        {'id': 3, 'type': 'list', 'exact_answer': [['d']], 'ideal_answer': 'yes'}
    ]
    output_ques2 = [
        {'id': 1, 'type': 'list', 'exact_answer': [['c']], 'ideal_answer': 'yes'},
        {'id': 2, 'type': 'list', 'exact_answer': [['b'],['a']], 'ideal_answer': 'yes'},
        {'id': 3, 'type': 'list', 'exact_answer': [], 'ideal_answer': 'yes'}
    ]
    metrics.calculate_qa_accuracy(golden_ques, output_ques1)
    metrics.calculate_qa_accuracy(golden_ques, output_ques2)

def test_ir_accuracy():
    golden_ques = [
        {
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/29610180",
                "http://www.ncbi.nlm.nih.gov/pubmed/1818237"
            ],
            "snippets": [
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 477,
                "text": "Blue diaper syndrome (BDS) (Online Mendelian Inheritance in Man number 211000) is an extremely rare disorder that was first described in 1964. The characteristic finding is a bluish discoloration of urine spots in the diapers of affected infants. Additional clinical features of the first described patients included diarrhea, inadequate weight gain, hypercalcemia, and nephrocalcinosis. An intestinal defect of tryptophan absorption was postulated as the underlying pathology.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/29610180"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 120,
                "text": "We describe the ocular abnormalities seen in a new metabolic disease which is deficient in the transport of tryptophan. ",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/1818237"
                }
            ],
            "body": "Which amino acid in implicated in the Blue diaper syndrome?",
            "type": "factoid",
            "id": "63f57d9b33942b094c000004",
        },
        {
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/35850241",
                "http://www.ncbi.nlm.nih.gov/pubmed/16581313",
                "http://www.ncbi.nlm.nih.gov/pubmed/16344344",
                "http://www.ncbi.nlm.nih.gov/pubmed/25831023",
                "http://www.ncbi.nlm.nih.gov/pubmed/9630233",
                "http://www.ncbi.nlm.nih.gov/pubmed/24152405",
                "http://www.ncbi.nlm.nih.gov/pubmed/35038030"
            ],
            "snippets": [
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 232,
                "text": "Friedreich\u0027s ataxia (FRDA) is an autosomal recessive neurodegenerative disorder caused by a triplet guanine-adenine-adenine (GAA) repeat expansion in intron 1 of the FXN gene, which leads to decreased levels of the frataxin protein.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/35850241"
                },
                {
                "offsetInBeginSection": 225,
                "offsetInEndSection": 403,
                "text": "The classic form of autosomal recessive ataxia, Friedreich\u0027s ataxia (FA), is now known to be due to an intronic expansion of a guanine-adenine-adenine (GAA)-trinucleotide repeat.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/9630233"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 152,
                "text": "Friedreich\u0027s Ataxia (FA) is the commonest genetic cause of ataxia and is associated with the expansion of a GAA repeat in intron 1 of the frataxin gene.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/16581313"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 110,
                "text": "Reduced expression of the mitochondrial protein Frataxin (FXN) is the underlying cause of Friedreich\u0027s ataxia.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/25831023"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 161,
                "text": "Friedreich Ataxia (FA) is the most common hereditary ataxia, caused by abnormal expansion of the GAA triplet of the first intron of the X25 gene on chromosome 9.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/24152405"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 162,
                "text": "BACKGROUND: Friedreich ataxia (FA), the most common hereditary ataxia, is caused by pathological expansion of GAA repeats in the first intron of the X25 gene on c",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/16344344"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 113,
                "text": "Friedreich Ataxia (FA) is a rare neuro-cardiodegenerative disease caused by mutations in the frataxin (FXN) gene.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/35038030"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 125,
                "text": "Friedreich\u0027s ataxia (FA) is an inherited neurodegenerative disorder caused by decreased expression of frataxin (FXN) protein.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/35289725"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 173,
                "text": "BACKGROUND: Friedreich ataxia (FA), the most common hereditary ataxia, is caused by pathological expansion of GAA repeats in the first intron of the X25 gene on chromosome 9",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/16344344"
                }
            ],
            "body": "What is the cause of Friedreich\u0027s Ataxia (FA)?",
            "type": "factoid",
            "id": "6450ec0c57b1c7a31500008f"
        }
    ]
    output_ques = [
        {
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/29610180",
            ],
            "snippets": [
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 477,
                "text": "Blue diaper syndrome (BDS) (Online Mendelian Inheritance in Man number 211000) is an extremely rare disorder that was first described in 1964. The characteristic finding is a bluish discoloration of urine spots in the diapers of affected infants. Additional clinical features of the first described patients included diarrhea, inadequate weight gain, hypercalcemia, and nephrocalcinosis. An intestinal defect of tryptophan absorption was postulated as the underlying pathology.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/29610180"
                }
            ],
            "body": "Which amino acid in implicated in the Blue diaper syndrome?",
            "type": "factoid",
            "id": "63f57d9b33942b094c000004",
        },
        {
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/16344344",
                "http://www.ncbi.nlm.nih.gov/pubmed/25831023",
                "http://www.ncbi.nlm.nih.gov/pubmed/9630233",
                "http://www.ncbi.nlm.nih.gov/pubmed/24152405",
                "http://www.ncbi.nlm.nih.gov/pubmed/35038030"
            ],
            "snippets": [
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 232,
                "text": "Friedreich\u0027s ataxia (FRDA) is an autosomal recessive neurodegenerative disorder caused by a triplet guanine-adenine-adenine (GAA) repeat expansion in intron 1 of the FXN gene, which leads to decreased levels of the frataxin protein.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/35850241"
                },
                {
                "offsetInBeginSection": 225,
                "offsetInEndSection": 403,
                "text": "The classic form of autosomal recessive ataxia, Friedreich\u0027s ataxia (FA), is now known to be due to an intronic expansion of a guanine-adenine-adenine (GAA)-trinucleotide repeat.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/9630233"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 152,
                "text": "Friedreich\u0027s Ataxia (FA) is the commonest genetic cause of ataxia and is associated with the expansion of a GAA repeat in intron 1 of the frataxin gene.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/16581313"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 110,
                "text": "Reduced expression of the mitochondrial protein Frataxin (FXN) is the underlying cause of Friedreich\u0027s ataxia.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/25831023"
                },
                {
                "offsetInBeginSection": 0,
                "offsetInEndSection": 161,
                "text": "Friedreich Ataxia (FA) is the most common hereditary ataxia, caused by abnormal expansion of the GAA triplet of the first intron of the X25 gene on chromosome 9.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/24152405"
                }
            ],
            "body": "What is the cause of Friedreich\u0027s Ataxia (FA)?",
            "type": "factoid",
            "id": "6450ec0c57b1c7a31500008f"
        }
    ]
    metrics.calculate_ir_accuracy(golden_ques, output_ques)
    metrics.calculate_ir_accuracy(golden_ques, golden_ques)

def test_postproc_predict():
    from prompt_utils import Question
    ques_obj = Question('question1', '1', 'list')
    llm_pred_list = [
        {'exact': [['serum biomarkers'],['complement component C3']], 'ideal': 'There are a variety of plasma and CSF biomarkers that have been'}, 
        {'exact': [['NfL'],['GFAP'],['complement component C3']], 'ideal': 'The level of serum NfL has been reported to be elevated in'}
    ]
    ans = qa_module.qa_postproc_prediction(ques_obj, llm_pred_list)
    print('\nList answer')
    print(ans)

test_calculate_qa_accuracy_yesno()
test_calculate_qa_accuracy_factoid()
test_calculate_qa_accuracy_list()
test_ir_accuracy()
test_postproc_predict()