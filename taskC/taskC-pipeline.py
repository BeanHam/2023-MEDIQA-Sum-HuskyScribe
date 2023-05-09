import numpy as np
import pandas as pd
import argparse
import shutil
import nltk
nltk.download('punkt')
from simplet5 import SimpleT5
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


def aggregator(dialogue, summarizer):
    
    ''' Concat dialogue chunk, then summarize '''
    t5_pred_outputs = {}
    
    # Split dialogue into chunks
    dialogue_chunks_cleaned, dialogue_chunks = chunker(dialogue)
    
    # taskA headers
    dialogue_labels = t5_subsection_classfier([dialogue_chunks_cleaned.tolist()], ontology_version = 'original_new')
    dialogue_labels = np.array([label[7:].strip().upper() for label in dialogue_labels])
    index = np.array([label in TASKA_TO_CANONICAL.keys() for label in dialogue_labels])
    dialogue_labels[~index] = 'OTHER'
    dialogue_labels = np.array([TASKA_TO_CANONICAL[label] for label in dialogue_labels])
        
    # taskB canonical headers
    dialogue_labels_canonical = t5_canonical_section_classfier([dialogue_chunks.tolist()])
    dialogue_labels_canonical = np.array([[tt.strip() if tt.strip() in CANONICAL_CLASSES else 'OTHER' for tt in label[7:].strip().upper().split(',')] for label in dialogue_labels_canonical])

    # print('dialogue_labels_canonical',dialogue_labels_canonical)
    # print('dialogue_labels', dialogue_labels)
    # Generated note
    note = []
    note_sections = []

    # Iterate over first-level sections:
    # SUBJECTIVE, OBJECTIVE_EXAM, OBJECTIVE_RESULTS, AP
    for section in SECTIONS:

        # exrtact second-level subsections
        subsections = SECTIONS[section]
        section_text = []

        # Iterate over second-level subsections
        for subsection in subsections:

            # Append note heading
            note.append(subsection+':')

            # Extract subsection texts
            dialogue_index = np.where(dialogue_labels == subsection)[0]
            dialogue_canonical_index = np.where(dialogue_labels_canonical == subsection)[0]
            all_index = np.unique(np.concatenate([dialogue_index, dialogue_canonical_index]))

            if len(all_index) == 0:
                note.append('None\n')
                continue
            subsection_chunks = dialogue_chunks[all_index]
            subsection_text = '\n'.join(subsection_chunks)

            # Get summary
            summary = summarizer.predict(subsection_text)[0]

            # Append note text
            note.append(summary+'\n')
            section_text.append(summary+'\n')

        # section text
        note_sections.append('\n'.join(section_text))

    # Note as string
    note = '\n'.join(note)

    return note, note_sections


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
    summarizer = SimpleT5()
    summarizer.load_model("t5","beanham/t5-large", use_gpu=True)
    
    #-------------------------
    # generate notes
    #-------------------------
    print('Generating Full Notes...')
    summaries = []
    subjective = []
    objective_exam = []
    objective_result = []
    ap = []
    section_pred_outputs_all = []
    for i, dialogue in enumerate(dialogues):
        summarized_note, section_note = aggregator(dialogue, summarizer)
        summaries.append(summarized_note)
        subjective.append(section_note[0])
        objective_exam.append(section_note[1])
        objective_result.append(section_note[2])
        ap.append(section_note[3])
    #np.save('summaries.npy', summaries)
    #np.save('subjective.npy', subjective)
    #np.save('objective_exam.npy', objective_exam)
    #np.save('objective_result.npy', objective_result)
    #np.save('ap.npy', ap)

    #-------------------------
    # output results
    #-------------------------
    data['note'] = summaries
    data['subjective'] = subjective
    data['objective_exam'] = objective_exam
    data['objective_results'] = objective_result
    data['assessment_and_plan'] = ap
    col_names_to_keep = ['encounter_id','note', 'subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
    col_names_to_del = [x for x in data.columns if x not in col_names_to_keep]
    data = data.drop(col_names_to_del, axis=1)
    data.to_csv('taskC_HuskyScribe_run1_mediqaSum.csv', index=False)

if __name__ == "__main__":
    main()
    
