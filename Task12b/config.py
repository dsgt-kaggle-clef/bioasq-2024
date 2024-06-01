FAKE_LLM = False
ADD_QA_CONTEXT = True
MAX_CONTEXT_LEN = 1000

SEARCH_WORD_LIST =[
    'SPACY',            # 0
    'LLM',              # 1
    'MIXTRAL_47B',      # 2
    'OPENAI'            # 3
]
SEARCH_WORD_MODE = 'MIXTRAL_47B'

OPENAI_MODEL_NAME = 'gpt-4'#'gpt-3.5-turbo'

LLM_LIST = [
    'mistral_7B',           # 0
    'llama2_7B',            # 1
    'mixtral_47B_instruct', # 2
    'llama2_70B',           # 3 
    'gptj_6B'               # 4
]
LLM_MODEL = LLM_LIST[2]

SNIP_EXTRACT_LIST = [
    'FIRST_SENTENCE',
    'TRANSFORMER'
]
SNIP_EXTRACT_MODE = SNIP_EXTRACT_LIST[1]

# Logging
VERBOSE = False
INFO_TRACE = True

# Pubmed
PUBMED_MAX_LENGTH = 30

# QA
QA_PROXY_CACHE_PATH = 'qa_proxy.json'
ENABLE_SYNONYM_GROUPING = False
LIST_SINGLE_PROMPT = True