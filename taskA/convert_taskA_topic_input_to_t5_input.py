import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq
import transformers
import os
from model_constants import  topic_ontology_taskA_topic
from sacred import Experiment

ex = Experiment()


@ex.config
def my_config():
    in_fn=None
    out_fn=None

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
from tqdm import tqdm 
from sacred import Experiment
import json



@ex.automain

def main(in_fn, out_fn):

    data_dir = '/home/sitongz/medqa/notebooks/data'

    import pandas as pd



    df = pd.read_csv(in_fn)


    df['t5_input'] = df.apply(lambda x: f" {x['dialogue']} Question: what is the section topic among categories below? {topic_ontology_taskA_topic.lower()} " ,  axis = 1)

    df['t5_output'] = 'empty'
    print(f'save to {out_fn}')

    df[['t5_input','t5_output']].to_csv(out_fn)