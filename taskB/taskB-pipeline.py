import numpy as np
import pandas as pd
import argparse
import shutil
import json
import torch
from tqdm import tqdm
from simplet5 import SimpleT5

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
    # load data
    #-------------------------
    print('Load Data...')
    data = pd.read_csv(args.data)
    dialogues = data['dialogue']
    
    #-------------------------
    # load summerizer
    #-------------------------
    print('Load Summarizer...')
    model = SimpleT5()
    model.load_model("t5","beanham/t5-large", use_gpu=True)
    summaries = [model.predict(dialogue)[0] for dialogue in tqdm(dialogues)]

    #-------------------------
    # output results
    #-------------------------
    data['SystemOutput'] = summaries
    col_names_to_keep = ['ID','SystemOutput']
    col_names_to_del = [x for x in data.columns if x not in col_names_to_keep]
    data = data.drop(col_names_to_del, axis=1)
    data = data.rename({'ID': 'TestID'}, axis=1)
    data.to_csv('taskB_HuskyScribe_run1_mediqaSum.csv', index=False)

if __name__ == "__main__":
    main()
    
