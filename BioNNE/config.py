VERBOSE = 0
DEBUG = 1
FAKE_LLM = True
LLM_MODEL_LIST = [
    'mistral_7B',
    'mixtral_47B_instruct'
]

LLM_MODEL = LLM_MODEL_LIST[1]
LLM_SEED = 0
MOCK_PATH = 'mocks/llm_resp.json'
ADD_INSTRUCTION = True
ENTITY_LIST = ['DISO', 'FINDING', 'ANATOMY', 'PHYS', 'CHEM', 'LABPROC', 'INJURY_POISONING', 'DEVICE']
RUN_MODE = 'validate' # train, validate, test
ENABLE_SECOND_LLM_RUN = False