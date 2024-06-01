# Copy from search/search-utils.py
import spacy
import requests
import xml.etree.ElementTree as ET
import config

MINDATE = '2000/01/01' # TODO: find a better starting date
MAXDATE = '2024/01/01' # TODO: BioASQ 12b requires us to use PubMed 2024 annual baseline version.
                       # Setting the end date to 2024 is a good approximation?

def extract_entities(model, sentence):
    """Given a spacy model and a sentence, it extract medical entities and return as list of tuple [(text_extraction, entity), ...]"""

    doc = model(sentence)

    return [(ent.text, ent.label_) for ent in doc.ents]

def get_pmids(query_term, max_length=config.PUBMED_MAX_LENGTH, verbose=True):  
    """Given a search term, query for relevant pubmed articles"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    db_param = "db=pubmed"
    retmax_param = f"retmax={max_length}"
    term_param = f"term={query_term}"
    date_range_param = f'mindate={MINDATE}&maxdate={MAXDATE}'
    full_url = f"{base_url}?{db_param}&{term_param}&{retmax_param}&{date_range_param}"
  
    if verbose:
        print('Querying from ', full_url)
    if config.INFO_TRACE:
        print('Query term: ', query_term)

    # Send the request
    response = requests.get(full_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the XML response
        root = ET.fromstring(response.content)

        # Extract PubMed IDs (PMIDs)
        pmids = [id_elem.text for id_elem in root.findall('.//IdList/Id')]
        
        # Display the PMIDs
        if verbose:
            for pmid in pmids:
                print("PMID:", pmid)
        return pmids
    
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
        return []
    
def formulate_query(kws, qbody=''):
    """Given a list of keywords, formulate a query term"""
    if not kws or len(kws) == 0:
        return qbody #fallback to qbody
    
    return ' AND '.join(keyword.replace(' ', '+') for keyword in kws)

def query_by_keywords(kws, max_length=10, verbose=True):
    """Given a list of keywords, query for relevant pubmed articles"""
    # query_term = ' AND '.join(keyword.replace(' ', '+') for keyword in kws)
    query_term = formulate_query(kws)
    return get_pmids(query_term, max_length, verbose)

def query_by_pmids(pmids, verbose=True):
    """Given a list of pmid, return a dictionary of the corresponding article title and abstract"""
    # Construct the EFetch query
    efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"

    # Send the request
    response = requests.get(efetch_url)
    # print(response.content)

    res = []
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the XML response
        root = ET.fromstring(response.content)
        # Iterate through each article in the response
        for article in root.findall('.//PubmedArticle'):
            item = {}
            # Extract and display the PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ''
            item['pmid'] = pmid
            if verbose:
                print("PMID:", pmid)

            # Extract and display the article title
            article_title_elem = article.find('.//ArticleTitle')
            article_title = article_title_elem.text if article_title_elem is not None else ''
            item['title'] = article_title
            if verbose:
                print("Title:", article_title)

            # Extract and display the abstract
            abstract_elem = article.find('.//Abstract')
            abstract_full_text = ''
            abstract_list = []
            if abstract_elem:
                for abs_nested_ele in abstract_elem:
                    if abs_nested_ele.tag == 'AbstractText':
                        # NLM uses all uppercase letters followed by a colon and space for the labels
                        # that appear in structured abstracts in MEDLINE/PubMed® citations 
                        if abs_nested_ele.attrib and ('Label' in abs_nested_ele.attrib):
                            abstract_full_text += abs_nested_ele.attrib['Label'] + ': '
                        if abs_nested_ele.text:
                            abstract_full_text += (abs_nested_ele.text)
                            abstract_list.append(abs_nested_ele.text)
                        else:
                            for ele_next in abs_nested_ele.itertext():
                                abstract_full_text += ele_next
                                abstract_list.append(ele_next)

            item['abstract_raw'] = abstract_full_text
            item['abstract_list'] = abstract_list
            if verbose:
                print("Abstract:", abstract_full_text)
                print("\n" + "-"*80 + "\n")

            res.append(item)
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
    return res

def load_spacy_model():
    return spacy.load("en_ner_bc5cdr_md")

if __name__ == '__main__':
    model = load_spacy_model()
    sentence = 'What are the latest recommendations for bone health and osteoporosis management in patients with Duchenne Muscular Dystrophy?'
    res = extract_entities(model, sentence)
    print('Sentence:', sentence)
    print('Extracted entities:', res)

    pmids = query_by_keywords([t[0] for t in res])
    res = query_by_pmids(pmids)


