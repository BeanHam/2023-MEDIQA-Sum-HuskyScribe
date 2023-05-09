import numpy as np
import pandas as pd
import argparse
import shutil
import nltk
nltk.download('punkt')
from transformers import pipeline
from model_utils import *
from model_constants import *

#--------------------------------------------------------
# Supplementary Functions
#--------------------------------------------------------
def chunker(dialogue):
    '''
    Overview
    --------
    Heuristically splits the input dialogue into chunks, by assuming that 
    each doctor turn is the start of a new chunk

    Parameters
    ----------
    dialogue : str, entire multi-turn dialgue

    Ouputs
    ------
    chunks : list of str, seqeunece of dialogue chunks 


    '''

    dialogue_cleaned = dialogue.replace('[doctor]','Doctor:').replace('[patient]','Patient:')
    sep = '\nDoctor:'
    chunks_cleaned  =  [d if did==0 else  'Doctor:'+d   for did, d in enumerate(dialogue_cleaned.split(sep)) ]

    sep = '\n[doctor]'
    chunks  =  [d if did==0 else  '[doctor]'+d   for did, d in enumerate(dialogue.split(sep)) ]
    
    return np.array(chunks_cleaned), np.array(chunks)

def section_formatter(subsection_norm, section_map=SUBSECTION_MAP):
    '''

    Overview
    --------
    Maps the input section heading to the expanded, properly formatted heading
    
    Parameters
    ----------
    subsection_norm : str, truncated, normalized section heading

    Ouputs
    ------
    subsection_expanded : str, expanded section name

    '''

    return section_map[subsection_norm]


def aggregator(dialogue, summarizer, tokenizer_kwargs):
    
    ''' Concat dialogue chunk, then summarize '''
    t5_pred_outputs = {}
    
    # Split dialogue into chunks
    dialogue_chunks_cleaned, dialogue_chunks = chunker(dialogue)
    
    # log intermediate
    t5_pred_outputs['dialogue_chunks'] = dialogue_chunks_cleaned.tolist()
    # Determin subsection for each dialogue chunk
    #dialogue_labels = t5_subsection_classfier([dialogue_chunks_cleaned.tolist()])
    dialogue_labels = t5_subsection_classfier([dialogue_chunks_cleaned.tolist()], ontology_version = 'original_new')

    dialogue_labels_canonical = t5_canonical_section_classfier([dialogue_chunks.tolist()])

    #print('dialogue_labels', dialogue_labels)
    #print('dialogue_labels_canonical',dialogue_labels_canonical)

    # log intermediate  
    t5_pred_outputs['t5_subsection_pred'] = dialogue_labels
    t5_pred_outputs['dialogue_labels_canonical'] = dialogue_labels_canonical

    dialogue_labels = np.array([label[7:].strip().upper() for label in dialogue_labels])
    index = np.array([label in CLASSIFICATION_MAP.keys() for label in dialogue_labels])
    dialogue_labels[~index] = 'OTHER'

    # task A label: subsection
    dialogue_labels = np.array([CLASSIFICATION_MAP[label] for label in dialogue_labels])
    
    # task C label: alignment annotation
    dialogue_labels_canonical = np.array([[tt.strip() if tt.strip() in CANONICAL_CLASSES else 'OTHER' for tt in label[7:].strip().upper().split(',')] for label in dialogue_labels_canonical])

    # log intermediate
    t5_pred_outputs['postprocessed_subsection_pred'] = dialogue_labels.tolist()
    t5_pred_outputs['postprocessed_canonical_pred'] = dialogue_labels_canonical.tolist()

    # Generated note 
    note = []
    
    # Iterate over first-level sections
    for section in SECTIONS:
            
        # Iterate over second-level subsections
        subsections = SECTIONS[section]
        for subsection in subsections:
    
            # Append note heading
            subsection = section_formatter(subsection)
            note.append(subsection+' :')

            # Extract subsection texts
            index = np.where(dialogue_labels == subsection)
            if len(index[0]) == 0:
                note.append('None\n')
                continue
            subsection_chunks = dialogue_chunks[index]
            subsection_text = '\n'.join(subsection_chunks)
            
            # Get summary 
            summary = summarizer(subsection_text, **tokenizer_kwargs)
    
            # Append note text
            note.append(summary[0]['summary_text']+'\n')
    
    # Note as string
    note = '\n'.join(note)
    
    return note, t5_pred_outputs


def main():
    
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser(
        prog='evaluate_summarization',
        description='This runs basic evaluation for both snippet (taskA) and full note summarization (taskB).'
    )
    parser.add_argument('--data', required=True, help='filename of test data')
    args = parser.parse_args()
    
    #-------------------------
    # make directories
    #-------------------------
    print('Make Directories...')
    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')
    else:
        shutil.rmtree('tmp/')
        os.makedirs('tmp/')
        
    #-------------------------
    # load data
    #-------------------------
    print('Load Data...')
    data = pd.read_csv(args.data)
    dialogues = data['dialogue']
    
    #-------------------------
    # load summerizer
    #-------------------------
    print('Load Summarizer...')
    #summarizer = pipeline("summarization", model="beanham/mediqa-samsum-dialoguesum")
    summarizer = pipeline("summarization", model='philschmid/bart-large-cnn-samsum')
    tokenizer_kwargs = {'truncation':True}
    
    #-------------------------
    # generate notes
    #-------------------------
    print('Generating Full Notes...')
    summaries = []
    section_pred_outputs_all = []
    for i, dialogue in enumerate(dialogues):
        summarized_note, section_pred_outputs = aggregator(
                dialogue, 
                summarizer, 
                tokenizer_kwargs
            )
        summaries.append(
            summarized_note
        )
        section_pred_outputs_all.append(
            {**section_pred_outputs, **{'dialogue_idx':i}}
        )

        
    #-------------------------
    # output results
    #-------------------------   
    data['SystemOutput'] = summaries
    col_names_to_keep = ['encounter_id','SystemOutput']
    col_names_to_del = [x for x in data.columns if x not in col_names_to_keep]
    data = data.drop(col_names_to_del, axis=1)
    data.to_csv('taskC_HuskyScribe_run1.csv', index=False)
    

    df_section_pred_all = pd.DataFrame(section_pred_outputs_all)
    df_section_pred_all.to_csv('taskC_HuskyScribe_run1_sectino_pred_by_chunk.csv', index=False)


if __name__ == "__main__":
    main()
    
