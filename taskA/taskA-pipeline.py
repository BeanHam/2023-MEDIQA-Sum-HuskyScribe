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
    summarizer = pipeline("summarization", model="beanham/mediqa-samsum-dialoguesum")
    tokenizer_kwargs = {'truncation':True}
    summaries = [summarizer(dialogue)[0]['summary_text'] for dialogue in tqdm(dialogues)]
    
    #-------------------------
    # load section predictor
    #-------------------------
    print('Load Section Header Model...')
    cmd = f"""python convert_taskA_topic_input_to_t5_input.py with \
              in_fn='{args.data}' \
              out_fn='tmp/taskA_test_input.csv'"""
    out = os.system(cmd)
    
    cmd_clinical_t5 = f"""python train_t5_with_trainer.py with \
              do_predict=True \
              do_train=False \
              task_name='tmp' \
              train_csv='tmp/taskA_test_input.csv' \
              val_csv='tmp/taskA_test_input.csv' \
              model_name='../models/Clinical-T5-Large'\
              model_short_name='clinical-t5-large' \
              load_weights_from='../models/Clinical-T5-Large' \
              pretrain_model_path='../models/Clinical-T5-Large' \
              test_pred_fn='tmp/taskA_test_output.csv'"""
    
    cmd_t5 = f"""python train_t5_with_trainer.py with do_predict=True do_train=False \
            task_name='topic_whole_update_ed_predict_step2000' \
            train_csv='tmp/taskA_test_input.csv' \
            val_csv='tmp/taskA_test_input.csv' \
            model_short_name=t5-large \
            load_weights_from='' \
            pretrain_model_path=sitongz/medqa_taskA_t5-large_topic_whole_update_ed-checkpoint-2000 \
            test_pred_fn='tmp/taskA_test_output.csv'"""
    cmd = cmd_t5
    out = os.system(cmd)
    
    f = open('tmp/taskA_test_output.csv')
    headers = json.load(f)
    pred_headers = [header['converted_output'].upper() for header in headers]
    f.close()
    
    #-------------------------
    # output results
    #-------------------------
    data['SystemOutput1'] = pred_headers
    data['SystemOutput2'] = summaries
    col_names_to_keep = ['ID','SystemOutput1', 'SystemOutput2']
    col_names_to_del = [x for x in data.columns if x not in col_names_to_keep]
    data = data.drop(col_names_to_del, axis=1)
    data = data.rename({'ID': 'TestID'}, axis=1)
    data.to_csv('taskA_HuskyScribe_run1.csv', index=False)
    
if __name__ == "__main__":
    main()
    
