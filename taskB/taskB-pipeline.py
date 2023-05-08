import numpy as np
import pandas as pd
import argparse
import shutil
import json
import torch
from tqdm import tqdm
from transformers import pipeline
from model_utils import *
from model_constants import *

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
    summarizer = pipeline("summarization", model="beanham/bart-large-finetune")
    tokenizer_kwargs = {'truncation':True}
    summaries = [summarizer(dialogue)[0]['summary_text'] for dialogue in tqdm(dialogues)]
        
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
    
