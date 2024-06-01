'''
Language processing using spacy modules and pipelines
'''
import spacy
import re
from scispacy.abbreviation import AbbreviationDetector
from doc_ner_module import DocNER, DocEntity
import helper_utils as helper
from spacy.matcher import Matcher
from pprint import pprint

class SpacyLMModule:
    def __init__(self, abstract, umls):
        self.abstract = abstract
        self.gen_nlp = spacy.load('en_core_web_trf') # general purpose NLP
        self.gen_nlp.add_pipe("abbreviation_detector") # TODO: does the abbreviation underlying NLP model matter?
        self.gen_doc = self.gen_nlp(abstract)

        self.nlp_bc5 = spacy.load("en_ner_bc5cdr_md") # biomedical NLP
        self.nlp_bc5_doc = self.nlp_bc5(abstract) # Entity types: DISEASE, CHEMICAL

        self.umls_module = umls

    def get_sentence(self, word, span):
        '''
        Get the sentence that contains the span
        '''
        for sent in self.gen_doc.sents:
            if span.start >= sent.start and span.end <= sent.end:
                return sent.text
        print('Error: sentence not found for span')
        return word

    def get_abbrev(self):
        '''
        Get the abbreviation and their long forms
        '''
        abrv_set = set()
        abrv_list = []
        for abrv in self.gen_doc._.abbreviations:
            if abrv.text in abrv_set:
                continue
            
            abrv_span = self.get_span(abrv.text)
            # Get span of long form
            longform_span = self.get_span(abrv._.long_form.text)
            longform_sent = self.get_sentence(abrv._.long_form.text, longform_span)

            abrv_entity = DocEntity(abrv.text, abrv_span, self.get_sentence(abrv.text, abrv_span))
            abrv_entity.is_abbr = True
            abrv_entity.abrv_longform = abrv._.long_form.text
            abrv_list.append(abrv_entity)
            long_entity = DocEntity(abrv._.long_form.text, longform_span, longform_sent)
            long_entity.abrv_shortform = abrv.text
            abrv_list.append(long_entity)
            abrv_set.add(abrv.text)
    
        umls_abbrvs = self.umls_module.detect_acronym_entity(self.abstract, get_tag=False)
        for abrv in umls_abbrvs:
            if abrv[0] in abrv_set:
                continue
            print('Self detected abrv:', abrv) # Add counter to track how many self detected abrvs
            short, long, derived_tag = abrv
            short_span = self.get_span(short)
            long_span = self.get_span(long)
            abrv_entity = DocEntity(short, short_span, self.get_sentence(short, short_span))
            abrv_entity.is_abbr = True
            abrv_entity.abrv_longform = long
            abrv_list.append(abrv_entity)
            long_entity = DocEntity(long, long_span, self.get_sentence(long, long_span))
            long_entity.abrv_shortform = short
            abrv_list.append(long_entity)
            abrv_set.add(short)

        return abrv_list

    def get_spy_ner_entities(self):
        '''
        Get the entities recognized by nlp_bc5 and nlp_bio13 model 
        '''
        spy_ner_set = set()
        result_set = {}
        print('Spacy BioMed NER entities:')
        for ent in self.nlp_bc5_doc.ents:
            if ent.text in spy_ner_set:
                continue
            print(f'"{ent.text}"', ent.start_char, ent.end_char, ent.label_)
            spy_ner_set.add(ent.text)
            if ent.label_ == 'DISEASE':
                result_set[ent.text] = {
                    'span': self.get_span(ent.text),
                    'label': 'DISEASE'
                }
            if ent.label_ == 'CHEMICAL':
                result_set[ent.text] = {
                    'span': self.get_span(ent.text),
                    'label': 'CHEM'
                }
        return result_set

    def get_span(self, text):
        '''
        Get the span of the text
        '''
        # TODO: check if there is exact match
        start = self.abstract.find(text)
        end = start + len(text)
        return self.gen_doc.char_span(start, end, alignment_mode='expand')

    def get_diso_entities(self):
        '''
        Get disease entities
        '''
        diso_set = set()
        for ent in self.nlp_bc5_doc.ents:
            if ent.label_ == 'DISEASE':
                diso_set.add(ent.text)
        return diso_set
    
    def get_numeric_entities(self):
        '''
        Get numeric token and span   
        '''
        numerics = []
        for ent in self.gen_doc.ents:
            if ent.label_ == 'CARDINAL':
                numerics.append((ent.text, ent.start_char, ent.end_char))
        return numerics

    def process_llm_output(self, llm_output):
        # check if llm_output is a string
        if isinstance(llm_output, str):
            resp_list = [llm_output]
        else:
            resp_list = llm_output
        ent_list1 = {}

        def insert_entity(word, index):
            if word in ent_list1:
                if index not in ent_list1[word]:
                    ent_list1[word].append(index)
            else:
                ent_list1[word] = [index]

        for idx, one_resp in enumerate(resp_list):
            raw_entities = one_resp.strip().split(';')
            for entity in raw_entities:
                if len(entity) > 1:
                    if entity == 'None' or entity == 'none' or (len(entity) <= 2 and entity.islower()):
                        continue
                    # Check if the entity exist in the original text, ignoring the case
                    entity1 = entity.strip()
                    if entity1 in self.abstract:
                        insert_entity(entity1, idx)
                    else:
                        entity2 = helper.get_altname(entity1)
                        if entity2 in self.abstract:
                            insert_entity(entity2, idx)

        checked_names = set()
        result = {}
        for ent in ent_list1:
            if ent in checked_names:
                continue
            found_span = False
            try:
                # Use regex to find all matches of ent in the text
                exact_match_offset = re.finditer(ent, self.abstract)
            except:
                print(f'Error: regex error for finding "{ent}"')
                continue
            match_offset_list = []
            for match in re.finditer(ent, self.abstract):
                # get offset of match
                start = match.start()
                end = match.end()
                # get the span of the match
                span = self.gen_doc.char_span(start, end, alignment_mode='strict')
                if span:
                    found_span = True
                    result[span.text] = { 'span': span, 'mentions': len(ent_list1[ent])}
                    checked_names.add(ent)
                    break
                match_offset_list.append((start, end))
            if not found_span:
                for start, end in match_offset_list:
                    span = self.gen_doc.char_span(start, end, alignment_mode='expand')
                    if span:
                        found_span = True
                        result[span.text] = { 'span': span, 'mentions': len(ent_list1[ent])}
                        checked_names.add(ent)
                        break
        # the result is a dictionary with the entity string as key and the span as value
        # pprint(result)
        return result

    def process_hardcode_entry(self, word):
        if word.islower():
            pattern = [[{'LOWER': word}], [{'LOWER': f'{word}s'}]]
        else:
            pattern = [[{'TEXT': word}]]
        matcher = Matcher(self.gen_nlp.vocab)
        matcher.add(word, pattern)
        matches = matcher(self.gen_doc)
        res = {}
        if len(matches) == 0:
            return res
        for match in matches:
            match_id, start, end = match
            span = self.gen_doc[start:end]
            res[span.text] = span
        return res

    def initialize_basic_doc_ent(self, word, span):
        '''
        Initialize a DocEntity object
        '''
        doc_entity = DocEntity(word, span, self.get_sentence(word, span))
        # populate is_acronym 
        # if the word contains only capital letters, it is an acronym
        if word.isupper():
            doc_entity.is_abbr = True
        # if the word contains only digits, it is a number
        if word.isdigit():
            doc_entity.is_like_num = True
        # if the span contains only one token, and the token is a number, it is a number
        if len(span) == 1:
            if span[0].is_digit or span[0].like_num:
                doc_entity.is_like_num = True
        
        for token in span:
            if token.pos_ == 'NUM' or token.is_digit or token.like_num:
                doc_entity.has_num = True
                break
        # TODO: More rules
        # 1. detect dates like 30th day as like date. For example detect if half of the span is a CARDINAL
        # 2. detect captical letter and - like  ICD-10 asa acryomn
        return doc_entity

    def match_original_word(self, input_list):
        '''
        (Unused)
        Given a list of entity phrases, match the original word in the text
        The word should be composed of full tokens
        '''
        result_span = []
        for ent in input_list:
            ent_start = self.abstract.find(ent)
            if ent_start == -1:
                print(f'Error: not found {ent} in original text')
                result_span.append((ent, '', ''))
            # match exact token if possible
            # get strict else get expanded token
            

            ent_end = ent_start + len(ent)
            # get the token index that contains the entity
            token_start_ind = 0
            for token in self.gen_doc:
                if token.idx >= ent_start:
                    break
                token_start_ind += 1
            token_end_ind = token_start_ind
            for token in self.gen_doc[token_start_ind:]:
                if token.idx >= ent_end:
                    break
                token_end_ind += 1
            # get the original word span
            original_span = self.gen_doc[token_start_ind:token_end_ind]
            original_word = original_span.text
            result_span.append((ent, original_word, original_span))
        return result_span

    def filtering_entity(self, phrase_span_dict):
        '''
        (Unused)
        filtering out entity that contains number or date POS token. Exclude from DISO type
        '''
        filtered_dict = {}
        for phrase, span in phrase_span_dict.items():
            is_valid = True
            for token in span:
                if token.pos_ in ['NUM']:
                    is_valid = False
                    break
                if token.is_digit or token.like_num:
                    is_valid = False
                    break
            if is_valid:
                filtered_dict[phrase] = span
        return filtered_dict

    def pos_categorized_entities(self, input_list):
        '''
        (Unused)
        Get the categorized entities
        '''
        pos_dict = {}
        for ent in self.gen_doc.ents:
            pos_dict[ent.text] = ent.label_
        return pos_dict