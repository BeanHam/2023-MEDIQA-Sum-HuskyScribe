import json
import pandas as pd
from model_constants import topic_ontology, SECTION2FULL, topic_ontology_canonical, topic_ontology_new

# def convert_dialogue_chunks2t5input_topic(dialogue_chunks, ontology = topic_ontology):
   
#     t5_data = []
#     for turn in dialogue_chunks:
#         t5_input = f" {turn} Question: what is the section topic among categories below? {topic_ontology} "
#         #t5_input = f" {turn} Question: what is the current section topic ? and Is the current section a different topic from the previous section? {ontology}"

#         t5_output = "Answer: DISPOSITION "
#         t5_data.append({'t5_input':  t5_input,
#                          't5_output': t5_output })
#     return t5_data

def convert_dialogue_chunks2t5input_seq(dialogue_chunks, ontology = topic_ontology):
   
    t5_data = []
    for tid, turn in enumerate(dialogue_chunks):
        x = {}
        x['sentence_AA'] = dialogue_chunks[tid-2] if tid-2 >= 0 else ''
        x['sentence_A'] = dialogue_chunks[tid-1] if tid-1 >= 0 else ''

        x['sentence_B'] = dialogue_chunks[tid] 
        x['sentence_BB'] = dialogue_chunks[tid + 1] if tid+1 < len(dialogue_chunks) else ''

        x['sentence_BBB'] = dialogue_chunks[tid + 2]  if tid+2 < len(dialogue_chunks) else ''
        x['section_header_full'] = 'DISPOSITION'
        x['new_section'] = False


        t5_input = f"previous section:  {x['sentence_AA']} {x['sentence_A']}\ncurrent section: {x['sentence_B']}\nnext section: {x['sentence_BB']} {x['sentence_BBB']} Question: what is the current section topic ? and Is the current section a different topic from the previous section? {ontology} " 
        t5_output =  f" Topic: {x['section_header_full']} | new topic: {   'yes' if x['new_section'] else  'no'}" 


        t5_data.append({'t5_input':  t5_input,
                         't5_output': t5_output })
    return t5_data


def write_dialogue_chunk_list(list_of_dialogue_chunks, output_file, ontology_version = 'original'):
    '''
    Overview
    --------
    Write dialogue chunk list into T5 input format for predict
    Parameters
    ----------
    list_of_dialogue_chunks : list of document snippet list
    '''
    # load ontology by version
    ontology_dict = {'original': topic_ontology,
                     'original_new': topic_ontology_new,
                     'canonical': topic_ontology_canonical,

                     }
    ontology = ontology_dict[ontology_version]
    # convert list of dialogue_chunks into T5 input format
    t5_data = []
    for dialogue_chunks in list_of_dialogue_chunks:
        # convert dialogue_chunks into T5 input format
        #t5_data_t  = convert_dialogue_chunks2t5input(dialogue_chunks)
        t5_data_t  = convert_dialogue_chunks2t5input_seq(dialogue_chunks, ontology = ontology)
        
        t5_data += t5_data_t
    
    #print(json.dumps(t5_data, indent=4))
    df = pd.DataFrame(t5_data)
    print('save to ', output_file)
    df.to_csv(output_file, index=False)
    print(output_file)
    return t5_data
