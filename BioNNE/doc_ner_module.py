'''
A class to check entities in a document
'''
import config
import json
import textsynth_api as llm

global COUNTER
COUNTER = {
    'exact_match': 0,
    'match_custom_findings': 0,
    'match_custom_injury': 0,
    'unmatched_freq_entity': 0,
    'matched_nested': 0,
    'semantic_search_succeed': 0,
    'semantic_search_failed': 0,
}

class DocNER:
    def __init__(self, text):
        self.abstract = text
        self.name_table = {}
        self.tag_table = {}
        for tag in config.ENTITY_LIST:
            self.tag_table[tag] = set()

    def print_name_table(self):
        '''
        Print the name table
        '''
        print()
        print('-'*80)
        for word in self.name_table:
            print(self.name_table[word])
        print()
        print('-'*80)

    def print_tag_table(self):
        '''
        Print the tag table
        '''
        print()
        for tag in self.tag_table:
            print(f'{tag}: {self.tag_table[tag]}')
    
    def add_word(self, doc_entity):
        '''
        Add a word to the name table
        '''
        self.name_table[doc_entity.text] = doc_entity
    
    def append_llm_ner_tag(self, word, new_tag, mentions):
        '''
        Append a new tag to the word
        '''
        if word in self.name_table:
            self.name_table[word].llm_ner_set.add(new_tag)
            self.name_table[word].llm_ner_mentions += mentions
        else:
            print(f'Error: Word {word} not found in the name table')

    def append_spy_ner_tag(self, word, new_tag):
        '''
        Append a new tag to the word
        '''
        if word in self.name_table:
            self.name_table[word].spy_ner_set.add(new_tag)
        else:
            print(f'Error: Word {word} not found in the name table')

    def populate_umls_tags(self, umls):
        '''
        Populate the UMLS tag for each word in the name_tables
        '''
        for word in self.name_table:
            doc_ent = self.name_table[word]
            if doc_ent.hardcode_tag is not None:
                continue
            if (not doc_ent.is_abbr) and len(doc_ent.umls_tags) == 0:
                umls_res = umls.umls_rules(doc_ent.text, topk=5)
                if umls_res is None:
                    print(f'No UMLS results found for {doc_ent.text}')
                    self.name_table[word].umls_tags = ['Unknown']
                else:
                    # May add check to see how concept match the original word
                    concept, semantic, tag = umls_res
                    self.name_table[word].umls_tags = tag
                    self.name_table[word].umls_concepts = concept
                    self.name_table[word].umls_semantics = semantic
                    self.name_table[word].umls_first_match_exact = (concept[0].lower() == doc_ent.text.lower())
                    if self.name_table[word].umls_first_match_exact:
                        print(f'Exact match found, {doc_ent.text}, type: {tag[0]}')

    def set_hardcode_tag(self, word, tag):
        '''
        Set hardcode tags for specific words
        '''
        if word not in self.name_table:
            print(f'Error: set_hardcode_tags word {word} not found in the name table')
            return
        self.name_table[word].hardcode_tag = tag

    def ask_llm_top3_tag_category(self, word):
        '''
        Distinguish between Anatomy, Physiology and Disease
        '''
        if self.name_table[word].umls_final_tag is not None:
            return
        options_sentence = ''
        ascii_lowercase = 'abcde'
        sem_options = ['Anatomy', 'Physiology', 'Disease', 'Chemical', 'Other']
        for index, semantic in enumerate(sem_options):
            char = ascii_lowercase[index]
            options_sentence += f'{char}. {semantic}\n'

        # Get sentence that contain the word span
        orig_sentence = self.name_table[word].sentence.strip()
        if orig_sentence != word:
            prompt = f'Identify term "{word}" in sentence "{orig_sentence}" refer to which of the following option\n'  + options_sentence + '\nEnter the option:\n'
            llm_query = {
                'prompt': prompt,
                'max_tokens': 30 + len(word.split()),
                'n': 2,
                'seed': config.LLM_SEED
            }
            print(f'\nSemantic Question: {prompt}')
            resp = llm.make_textsynth_request(
                f'/v1/engines/{config.LLM_MODEL}/completions',
                llm_query)['text']
            print(f'Semantic response: {resp}')
            found = False
            cate_set = set()
            for one_resp in resp:
                lower_resp = one_resp.lower()
                physio_loc = lower_resp.find('physiology')
                anatomy_loc = lower_resp.find('anatomy')
                disease_loc = lower_resp.find('disease')
                chemical_loc = lower_resp.find('chemical')
                other_loc = lower_resp.find('other')
                loc_dict = {physio_loc: 'PHYS', anatomy_loc: 'ANATOMY', disease_loc: 'DISO', chemical_loc: 'CHEM', other_loc: 'OTHER'}
                min_loc = float("inf")
                
                category = 'OTHER'
                for loc in [physio_loc, anatomy_loc, disease_loc, chemical_loc, other_loc]:
                    if loc >=0 and loc < min_loc:
                        min_loc = loc
                        category = loc_dict[loc]
                        found = True
                print(f'Chosen category: {category}')
                cate_set.add(category)

            if found:
                if len(cate_set) == 1:
                    self.name_table[word].umls_final_tag = cate_set.pop()
                    COUNTER['semantic_search_succeed'] += 1
                elif len(cate_set) > 1:
                    print('Error: LLM semantic multiple category found')
                    self.name_table[word].umls_final_tag = cate_set.pop()
                COUNTER['semantic_search_succeed'] += 1

            if not found:
                print('Error: LLM semantic no option found')
                self.name_table[word].umls_final_tag = 'Other'
                COUNTER['semantic_search_failed'] += 1

    def ask_llm_semantic_type(self, word):
        '''
        Given UMLS semantic type, ask LLM to predict the semantic type given the sentence
        '''
        if self.name_table[word].umls_final_tag is not None:
            return
        options_sentence = ''
        ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
        sem_options = self.name_table[word].umls_semantics
        sem_options.append('Other')
        for index, semantic in enumerate(sem_options):
            char = ascii_lowercase[index]
            options_sentence += f'{char}. {semantic}\n'
        # Get sentence that contain the word span
        orig_sentence = self.name_table[word].sentence
        if orig_sentence:
            prompt = f'"{word}" in sentence: "{orig_sentence}" refer to which of the following option\n'  + options_sentence + 'Enter the option:\n'
            llm_query = {
                'prompt': prompt,
                'max_tokens': 30 + len(word.split()),
                'n': 2,
                'seed': config.LLM_SEED
            }
            print(f'Semantic Question: {prompt}')
            resp = llm.make_textsynth_request(
                f'/v1/engines/{config.LLM_MODEL}/completions',
                llm_query)['text']
            print(f'Semantic response: {resp}')
            found = False
            for one_resp in resp:
                # find "option x" in the response and get the x
                for ind, sem_option in enumerate(sem_options):
                    char = ascii_lowercase[ind]
                    if (f'option {char}' in one_resp) or (one_resp.startswith(char + '.')) or one_resp.startswith(char + '\n') or one_resp.startswith(f'Answer: {char}.'):
                        print(f'Chosen option: {sem_option}')
                        if ind >= len(sem_options) - 1:
                            self.name_table[word].umls_final_tag = 'Other'
                        else:
                            self.name_table[word].umls_final_tag = self.name_table[word].umls_tags[ind]
                        found = True
                        COUNTER['semantic_search_succeed'] += 1
                        break
            if not found:
                print('Error: LLM semantic no option found')
                self.name_table[word].umls_final_tag = 'Other'
                COUNTER['semantic_search_failed'] += 1

    def finalize_word_helper(self, doc_ent, tag):
        '''
        Finalize the word with the given tag
        '''
        self.name_table[doc_ent.text].final_tag = tag
        self.tag_table[tag].add(doc_ent.text)
        # check if the word has shortform
        if doc_ent.abrv_shortform is not None:
            shortform = doc_ent.abrv_shortform
            if shortform in self.name_table and self.name_table[shortform].final_tag is None:
                self.name_table[shortform].final_tag = tag
                self.tag_table[tag].add(shortform)
    
    def pass_custome_finding_rule(self, doc_ent):
        '''
        Return true if the word can be categorized as finding
        '''
        if doc_ent.is_like_num or doc_ent.has_num:
            return False
        if doc_ent.llm_ner_mentions < 2:
            return False
        token_size = len(doc_ent.text.split())
        if token_size >= 2 and token_size <= 10:
            return True
        return False

    def pass_custome_injury_rule(self, doc_ent):
        '''
        Return true if the word can be categorized as injury
        '''
        if doc_ent.is_like_num or doc_ent.has_num:
            return False
        token_size = len(doc_ent.text.split())
        if token_size >= 2 and token_size <= 5:
            return True
        return False

    def populate_final_tag(self):
        '''
        Populate the final tag for each word in the name_tables
        '''
        high_freq_nested_entity = []
        for word in self.name_table:
            doc_ent = self.name_table[word]
            if doc_ent.final_tag is not None:
                continue

            if doc_ent.hardcode_tag is not None:
                self.name_table[word].final_tag = doc_ent.hardcode_tag
                if doc_ent.hardcode_tag in config.ENTITY_LIST:
                    self.tag_table[doc_ent.hardcode_tag].add(word)
                continue
            
            found_tag = False

            # check for exact match
            if (self.name_table[word].umls_first_match_exact):
                my_tag = self.name_table[word].umls_tags[0]
                self.name_table[word].final_tag = my_tag
                if my_tag in config.ENTITY_LIST:
                    self.finalize_word_helper(doc_ent, my_tag)
                COUNTER['exact_match'] += 1
                continue

            # Checking Device and Injury before DISO, as injury is a subset of DISO
            for tag in ['INJURY_POISONING']:
                if (tag in doc_ent.llm_ner_set) and (tag in doc_ent.umls_tags):
                    self.finalize_word_helper(doc_ent, tag)
                    found_tag = True
                    break
            if found_tag:
                continue

            # Check if the word is a DISO
            if 'DISEASE' in doc_ent.spy_ner_set:
                if ('DISO' in doc_ent.llm_ner_set) or ('DISO' in doc_ent.umls_tags):
                    self.finalize_word_helper(doc_ent, 'DISO')
                    continue
            
            # Check if the word is a CHEM
            if 'CHEM' in doc_ent.spy_ner_set:
                if ('CHEM' in doc_ent.llm_ner_set) or ('CHEM' in doc_ent.umls_tags):
                    self.finalize_word_helper(doc_ent, 'CHEM')
                    continue

            potential_lists = []
            for tag in ['FINDING', 'DISO','ANATOMY', 'PHYS', 'CHEM', 'LABPROC', 'DEVICE']:
                if (tag in doc_ent.llm_ner_set) and (tag in doc_ent.umls_tags):
                    potential_lists.append(tag)
            
            if len(potential_lists) == 1:
                self.finalize_word_helper(doc_ent, potential_lists[0])
                continue
            elif len(potential_lists) > 1:
                # Check the order and frequency in UMLS.
                tag_order = {}
                for tag in potential_lists:
                    pos_tag = doc_ent.umls_tags.index(tag)
                    tag_order[pos_tag] = tag
                # Get the tag with the lowest index
                min_index = min(tag_order.keys())
                self.finalize_word_helper(doc_ent, tag_order[min_index])
                continue

            if config.ENABLE_SECOND_LLM_RUN and self.name_table[word].has_threebig_llm_ner_tag():
                self.ask_llm_top3_tag_category(word)
                my_tag = self.name_table[word].umls_final_tag
                if (my_tag is not None) and (my_tag in ['ANATOMY', 'PHYS', 'CHEM']):
                    self.finalize_word_helper(doc_ent, my_tag)
                    continue

            # Accept findings that are >= 2 tokens <= 10 tokens and do not contain numbers
            if self.name_table[word].has_ner_finding_tag() and self.pass_custome_finding_rule(doc_ent):
                self.finalize_word_helper(doc_ent, 'FINDING')
                COUNTER['match_custom_findings'] += 1
                continue
            
            if (self.name_table[word].llm_ner_mentions >= 6) and (not self.name_table[word].is_abbr):
                print(f'{word} has more than 6 mentions, mentiond times: {self.name_table[word].llm_ner_mentions}')
                print(f'LLM NER: {self.name_table[word].llm_ner_set} , UMLS: {self.name_table[word].umls_tags}')
                COUNTER['unmatched_freq_entity'] += 1
                if (not self.name_table[word].is_abbr) and (not self.name_table[word].has_num) and ('Unknown' in self.name_table[word].umls_tags):
                    high_freq_nested_entity.append(word)
        
        for word in high_freq_nested_entity:
            # Find the longest substring that is a valid final tag which is also part of the llm_ner_set of this entity
            # If found, set the final tag to the longest substring
            max_matching_length = 0
            matching_entity = None
            matching_tag = None
            word_split = word.lower().split()
            for entity in self.name_table:
                entity_split = entity.lower().split()
                if not (self.name_table[entity].is_abbr) and (self.name_table[entity].final_tag is not None):
                    if (entity.lower() in word.lower()) and (len(entity) > max_matching_length):
                        if self.name_table[entity].final_tag in self.name_table[word].llm_ner_set:
                            if len(list(set(word_split) & set(entity_split))) > 0:
                                max_matching_length = len(entity)
                                matching_entity = entity
                                matching_tag = self.name_table[entity].final_tag
            if matching_entity is not None:
                self.finalize_word_helper(self.name_table[word], matching_tag)
                print(f'High freq nested entity: {word} matched with {matching_entity} with tag {matching_tag}')
                COUNTER['matched_nested'] += 1

    def get_tagged_entities(self, tag):
        '''
        Get the tagged entities for a specific tag
        '''
        return self.tag_table[tag]
    
    def get_order_entity_by_mentions(self):
        '''
        Get the order of entities based on the number of mentions
        '''
        res = sorted(self.name_table.values(), key=lambda x: x.llm_ner_mentions, reverse=True)
        # print the result with the entity name and mention count
        print()
        print('Order by Mentions:')
        for item in res:
            print(f'{item.text}: {item.llm_ner_mentions}')

class DocEntity:
    def __init__(self, text, span, sentence):
        self.text = text
        self.span = span
        self.sentence = sentence
        self.is_abbr = False
        self.is_like_num = False
        self.has_num = False
        self.abrv_longform = None
        self.abrv_shortform = None
        self.spy_ner_set = set()
        self.umls_tags = []
        self.umls_concepts = []
        self.umls_semantics = []
        self.umls_first_match_exact = False
        self.umls_final_tag = None
        self.llm_ner_set = set()
        self.llm_ner_mentions = 0
        self.hardcode_tag = None
        self.final_tag = None

        # check sentence contain text
        if (not self.is_abbr) and (text.lower() not in sentence.lower()):
            print(f'Error: {text} not found in sentence {sentence}')
    
    def __str__(self):
        return f'{self.text}: span=({self.span.start}, {self.span.end}), umls_tags={self.umls_tags}, llm_ner_set={self.llm_ner_set}, spy_ner_set={self.spy_ner_set}, ' \
            f'like_num={self.is_like_num}, has_num={self.has_num}, abbr={self.is_abbr}, abrv_longform={self.abrv_longform}, abrv_shortform={self.abrv_shortform}, ' \
            f'hardcode_tag = {self.hardcode_tag}, final_tag={self.final_tag}'
    
    def has_valid_umls_tag(self):
        '''
        Check if the entity has a valid UMLS tag
        '''
        if self.is_abbr:
            return False
        if len(self.umls_tags) <= 0:
            return False
        for tag in self.umls_tags:
            if tag in config.ENTITY_LIST:
                return True
        return False
    
    def has_threebig_llm_ner_tag(self):
        '''
        Check if the entity has a valid llm ner tag
        '''
        if self.is_abbr:
            return False
        word_length = len(self.text.split())
        if word_length < 2 or word_length > 5:
            return False
        if self.has_num:
            return False
        if len(self.llm_ner_set) <= 0:
            return False
        for tag in self.llm_ner_set:
            if tag in ['PHYS', 'ANATOMY', 'CHEM']:
                return True
        return False  

    def has_ner_finding_tag(self):
        '''
        Check if the entity has a valid llm ner tag
        '''
        if self.is_abbr:
            return False
        if len(self.llm_ner_set) <= 0:
            return False
        if 'FINDING' in self.llm_ner_set:
            return True
        return False

    def has_freq_ner_injury_tag(self):
        '''
        Check if the entity has a valid llm ner tag
        '''
        if self.is_abbr:
            return False
        if self.llm_ner_mentions < 1:
            return False
        if 'INJURY_POISONING' in self.llm_ner_set:
            return True
        return False

    def has_unanimous_umls_tag(self, tag):
        '''
        Check if the entity has a unanimous UMLS tag
        '''
        if len(self.umls_tags) <= 0:
            return False
        umls_set = set(self.umls_tags)
        if tag in umls_set and len(umls_set) == 1:
            return True
        return False