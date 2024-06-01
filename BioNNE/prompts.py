ANATOMY_PROMPT = 'Return phrases or entities that comprise organs, body part, cells and cell components, body substances in TEXT'
CHEM_PROMPT = 'Return chemicals, including legal and illegal drugs, and biological molcules in TEXT'
DEVICE_PROMPT = 'Return manufactured objects used for medical purposes in TEXT'
DISO_PROMPT = 'Return diseases, symptoms, dyfunctions, abnormality of organ, excluding injuries or poisoning in TEXT'
FINDING_PROMPT = 'Return phrases or entities that convey the result of scientific study, experiments described in TEXT'
INJURY_POISONING_PROMPT = 'Return injuries on the body as a result of external force including poisoning in TEXT'
LABPROC_PROMPT = 'Return testing body substances and other diagnostic procedures in TEXT'
PHYS_PROMPT = 'Return biological function or process in organism including organism attribute (such as temperature) and excluding mental processes described in TEXT'

def get_instruction(tag):
    '''
    Get the instruction for the few-shot prompt
    '''
    
    str1 = 'Instruction: '
    if tag == 'FINDING':
        str1 += FINDING_PROMPT
    elif tag == 'DISO':
        str1 += DISO_PROMPT
    elif tag == 'ANATOMY':
        str1 += ANATOMY_PROMPT
    elif tag == 'DEVICE':
        str1 += DEVICE_PROMPT
    elif tag == 'CHEM':
        str1 += CHEM_PROMPT
    elif tag == 'LABPROC':
        str1 += LABPROC_PROMPT
    elif tag == 'PHYS':
        str1 += PHYS_PROMPT
    elif tag == 'INJURY_POISONING':
        str1 += INJURY_POISONING_PROMPT
    else:
        print('Error Invalid tag')
        return ''
    str1 += f', in the {tag} concatenated by \';\'\n\n'
    return str1

