class Question:
    ''' class that capture the attributes related to a Synergy question'''
    def __init__(self, body, id, type, answerReady):
        self.body = body
        self.id = id
        self.type = type
        self.documents = []
        self.snippets = []
        self.answer_ready = answerReady
        self.ideal_answer = ""
        self.exact_answer = []

    def __str__(self):
        str_out = f'id={self.id}, type={self.type}, body={self.body}\n'
        if len(self.documents) > 0:
            str_out += f'article={self.documents}\n'
        if len(self.snippets) > 0:
            str_out +=  f'snippet={self.snippets}\n'
        if len(self.ideal_answer) > 0 or len(self.exact_answer) > 0:
            str_out += f'exact={self.exact_answer}; ideal={self.ideal_answer}\n'
        return str_out

    def to_json(self):
        return json.dumps(self.__dict__)

    def to_dict(self):
        return self.__dict__

# -----------------------------------------------------------------------------
#  The prompt template and LLM requst parameter for each type of question
# -----------------------------------------------------------------------------
yesno_tmplate = '''Question: Is the protein Papilin secreted?
Ideal answer: Yes, papilin is a secreted protein
Exact answer: yes
###
Question: Are long non coding RNAs as conserved in sequence as protein coding genes?
Ideal answer: No. Most long non coding RNAs are under lower sequence constraints than protein-coding genes.
Exact answer: no
###
Question: {body}
Ideal answer: '''

yesno_context_tmplate = '''Context: Papilins are homologous, secreted extracellular matrix proteins which share a common order of protein domains.
Question: Is the protein Papilin secreted?
Ideal answer: Yes, papilin is a secreted protein
Exact answer: yes
###
Context:
Question: Are long non coding RNAs as conserved in sequence as protein coding genes?
Ideal answer: No. Most long non coding RNAs are under lower sequence constraints than protein-coding genes.
Exact answer: no
###
Context: {context}
Question: {body}
Ideal answer: '''

summary_tmplate = '''Question: Is Hirschsprung disease a mendelian or a multifactorial disorder?
Answer: Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model.
###
Question:What are the outcomes of Renal sympathetic denervation?
Answer: Significant decreases and progressively higher reductions of systolic and diastolic blood pressure were observed after RSD. The complication rate was minimal.\nRenal sympathetic denervation also reduces heart rate, which is a surrogate marker of cardiovascular risk.
###
Question: {body}
Answer: '''

summary_context_tmplate = '''Context: Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes.
Question: Is Hirschsprung disease a mendelian or a multifactorial disorder?
Answer: Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model.
###
Context: Significant decreases and progressively higher reductions of systolic and diastolic blood pressure were observed after RSD.
The RSD presents itself as an effective and safe approach to resistant hypertension.
Question: What are the outcomes of Renal sympathetic denervation?
Answer: Renal sympathetic denervation reduces heart rate, systolic and diastolic blood pressure.
###
Context: {context}
Question: {body}
Answer: '''

list_tmplate = '''Question: Which human genes are more commonly related to craniosynostosis?
Ideal answer: The genes that are most commonly linked to craniosynostoses are the members of the Fibroblast Growth Factor Receptor family FGFR3 and to a lesser extent FGFR1 and FGFR2. Some variants of the disease have been associated with the triplication of the MSX2 gene and mutations in NELL-1. NELL-1 is being regulated bu RUNX2, which has also been associated to cases of craniosynostosis. Other genes reported to have a role in the development of the disease are RECQL4, TWIST, SOX6 and GNAS.
Exact answer: [["FGFR3"], ["FGFR2"], ["FGFR1"], ["MSX2"], ["NELL1"],["RUNX2"],["RECQL4"],["TWIST"], ["SOX6"], ["GNAS"]]
###
Question: What are the main indications of lacosamide?
Ideal answer: Lacosamide is an anti-epileptic drug, licensed for refractory partial-onset seizures. In addition to this, it has demonstrated analgesic activity in various animal models. Apart from this, LCM has demonstrated potent effects in animal models for a variety of CNS disorders like schizophrenia and stress induced anxiety.
Exact answer: [["refractory epilepsy"],["analgesic"],["CNS disorders"]]
###
Question: {body}
Ideal answer: '''

list_context_tmplate = '''Context: The FGFR3 P250R mutation was the single largest contributor (24%) to the genetic group
Syndromic craniosynostosis due to complex chromosome 5 rearrangement and MSX2 gene triplication
Question: Which human genes are more commonly related to craniosynostosis?
Ideal answer: The genes that are most commonly linked to craniosynostoses are the members of the Fibroblast Growth Factor Receptor family FGFR3 and to a lesser extent FGFR1 and FGFR2. Some variants of the disease have been associated with the triplication of the MSX2 gene and mutations in NELL-1. NELL-1 is being regulated bu RUNX2, which has also been associated to cases of craniosynostosis. Other genes reported to have a role in the development of the disease are RECQL4, TWIST, SOX6 and GNAS.
Exact answer: [["FGFR3"], ["FGFR2"], ["FGFR1"], ["MSX2"], ["NELL1"],["RUNX2"],["RECQL4"],["TWIST"], ["SOX6"], ["GNAS"]]
###
Context: 
Question: What are the main indications of lacosamide?
Ideal answer: Lacosamide is an anti-epileptic drug, licensed for refractory partial-onset seizures. In addition to this, it has demonstrated analgesic activity in various animal models. Apart from this, LCM has demonstrated potent effects in animal models for a variety of CNS disorders like schizophrenia and stress induced anxiety.
Exact answer: [["refractory epilepsy"],["analgesic"],["CNS disorders"]]
###
Context: {context}
Question: {body}
Ideal answer: '''

factoid_tmplate = '''Question: Which fusion protein is involved in the development of Ewing sarcoma?
Ideal answer: Ewing sarcoma is the second most common bone malignancy in children and young adults. In almost 95% of the cases, it is driven by oncogenic fusion protein EWS/FLI1, which acts as an aberrant transcription factor, that upregulates or downregulates target genes, leading to cellular transformation.
Exact answer: [["EWS"], ["FLI1"]]
###
Question: Name synonym of Acrokeratosis paraneoplastica.
Ideal answer: Acrokeratosis paraneoplastic (Bazex syndrome) is a rare, but distinctive paraneoplastic dermatosis characterized by erythematosquamous lesions located at the acral sites and is most commonly associated with carcinomas of the upper aerodigestive tract.
Exact answer: [["Bazex syndrome"]]
###
Question: {body}
Ideal answer: '''

factoid_context_tmplate = '''Context: Ewing sarcoma is the second most common bone malignancy in children and young adults. It is driven by oncogenic fusion proteins (i.e. EWS/FLI1) acting as aberrant transcription factors that upregulate and downregulate target genes, leading to cellular transformation
Ewing sarcoma/primitive neuroectodermal tumors (EWS/PNET) are characterized by specific chromosomal translocations most often generating a chimeric EWS/FLI-1 gene
Question: Which fusion protein is involved in the development of Ewing sarcoma?
Ideal answer: Ewing sarcoma is the second most common bone malignancy in children and young adults. In almost 95% of the cases, it is driven by oncogenic fusion protein EWS/FLI1, which acts as an aberrant transcription factor, that upregulates or downregulates target genes, leading to cellular transformation.
Exact answer: [["EWS"], ["FLI1"]]
###
Context: Acrokeratosis paraneoplastica of Bazex is a rare but important paraneoplastic dermatosis, usually manifesting as psoriasiform rashes over the acral sites
Bazex syndrome (acrokeratosis paraneoplastica): persistence of cutaneous lesions after successful treatment of an associated oropharyngeal neoplasm.
Question: Name synonym of Acrokeratosis paraneoplastica.
Ideal answer: Acrokeratosis paraneoplastic (Bazex syndrome) is a rare, but distinctive paraneoplastic dermatosis characterized by erythematosquamous lesions located at the acral sites and is most commonly associated with carcinomas of the upper aerodigestive tract.
Exact answer: [["Bazex syndrome"]]
###
Context: {context}
Question: {body}
Ideal answer: '''

prompt_config = {}
prompt_config['yesno'] = {
    'prompt': yesno_tmplate,
    'context_prompt': yesno_context_tmplate,
    'max_tokens': 200
}
prompt_config['summary'] = {
    'prompt': summary_tmplate,
    'context_prompt': summary_context_tmplate,
    'max_tokens': 200
}
prompt_config['list'] = {
    'prompt': list_tmplate,
    'context_prompt': list_context_tmplate,
    'max_tokens': 200
}
prompt_config['factoid'] = {
    'prompt': factoid_tmplate,
    'context_prompt': factoid_context_tmplate,
    'max_tokens': 200
}

# -----------------------------------------------------------------------------
#  The prompt template for question keyword extraction
# -----------------------------------------------------------------------------
keyword_extact_prompt ="""Q: What is the mode of action of Molnupiravir?
Keywords: Molnupiravir, action
###
Q: Is dapagliflozin effective for COVID-19?
Keywords: dapagliflozin, COVID-19
###
Q: {body}
Keywords: """