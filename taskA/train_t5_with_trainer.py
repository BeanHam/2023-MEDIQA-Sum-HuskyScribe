import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq
import transformers
import os
from model_utils import convert_full2section

from model_constants import  CLINICAL_T5_SCRATCH_MODEL_NAME, CLINICAL_T5_LARGE_MODEL_NAME, CLINICAL_T5_SCRATCH_MODEL_NAME_SAFE
# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'
from tqdm import tqdm 
from sacred import Experiment
import json

import nltk
nltk.download('punkt')


ex = Experiment()


# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
def preprocess_function(examples, max_input_length=1024, max_target_length=128, tokenizer = None):
    inputs = [doc for doc in examples["t5_input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():

        labels = tokenizer(examples["t5_output"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
import nltk
import numpy as np
metric = load_metric("rouge")

def compute_metrics_any_tokenizer(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    #print('decoded_preds',decoded_preds)
    #print('decoded_labels',decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


@ex.config
def my_config():
    foo = 42
    bar = 'baz'
    
    train_bs = 2   
    val_bs = 2   
    train_ep = 1     
    gradient_accumulation_steps = 1
    fast_run = False
    val_ep = 1 

    seed = 42            
    max_input_length = 1024
    max_target_length = 150 
    do_train = False
    do_predict = False
    load_weights_from = None


    model_short_name = 'clinical-t5-scratch'
    model_name = None # OUTDATED
    pretrain_model_path = None

    max_epochs = 1
    max_steps = None

    task_name = 'topic'
    train_csv = None
    val_csv = None
    destination = f'exp/{model_short_name}_{task_name}'
    if fast_run:
        destination = f'exp/{model_short_name}_{task_name}_fast'
    model_output_dir = f'{destination}/model/'
    os.makedirs(destination, exist_ok=True)
    test_metric_fn = f'{destination}/metrics_test.json'
    test_pred_fn = f'{destination}/predictions_test.json'



@ex.automain
def main(train_csv,val_csv, model_short_name, seed,fast_run,
    train_bs,val_bs,gradient_accumulation_steps,train_ep, val_ep, max_input_length, max_target_length,max_epochs,max_steps,
     do_train, do_predict, load_weights_from ,pretrain_model_path,
     test_metric_fn, test_pred_fn, model_output_dir):


    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(seed) # pytorch random seed
    np.random.seed(seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    model_short_name2pretrain_path = {
        'clinical-t5-scratch': CLINICAL_T5_SCRATCH_MODEL_NAME,
        'clinical-t5-large': CLINICAL_T5_LARGE_MODEL_NAME ,
        't5-base': 't5-base',
        't5-large': 't5-large',
        'clinical-t5-large-old': 'luqh/ClinicalT5-large',

    }

    # tokenzier for encoding the text
    if pretrain_model_path:
        # load on clinical t5 with credential data from  MIMIC4
        print('tokenizer load from ',pretrain_model_path)

        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, use_fast=True)
    elif model_short_name in model_short_name2pretrain_path:
        model_name = model_short_name2pretrain_path[model_short_name]
        print('tokenizer load from ',model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f'Unknown model_short_name: {model_short_name}')

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 


    '''
    Loading dataset
    Requirement:
                dataet should be in csv format
                csv should have two columns: t5_input, t5_output
    '''
    train_dataset  = load_dataset("csv", data_files={'train':train_csv})
    val_dataset  = load_dataset("csv", data_files={'test':val_csv})
    if fast_run:
        train_dataset['train'] = train_dataset['train'].select([0, 10, 20, 30, 40, 50])
        val_dataset['test'] = val_dataset['test'].select([0, 10, 20, 30, 40, 50])


    print("Reading training data...")
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    tokenized_val_set = val_dataset.map(lambda x: preprocess_function(x,  max_input_length=max_input_length, max_target_length=max_target_length, tokenizer=tokenizer), batched=True)
    tokenized_train_set = train_dataset.map(lambda x: preprocess_function(x,  max_input_length=max_input_length, max_target_length=max_target_length,tokenizer=tokenizer), batched=True)



    # load train args

    batch_size = 2
    if max_steps is None:
        max_steps = -1
    if max_epochs is None:
        max_epochs = 1
    training_args = Seq2SeqTrainingArguments(
        model_output_dir,
        evaluation_strategy = "steps",
        save_strategy = "steps",
        learning_rate=2e-5,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=val_bs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=max_epochs,
        eval_steps = 1000,
        save_steps = 1000,
        predict_with_generate=True,
        max_steps = max_steps,
         load_best_model_at_end=True,
         metric_for_best_model = 'eval_rougeLsum',
    )



    '''
    Loading pretrain model
    if pretrain_model_path is given, we load there
    otherwise we load model by mapping from short_name to the pretrained checkpoint name

    '''


    if pretrain_model_path is not None:

        print(f'Loading model {pretrain_model_path}')
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path)

    elif model_short_name  in model_short_name2pretrain_path:
        pretrain_model_path = model_short_name2pretrain_path[model_short_name]
        if model_short_name == 'clinical-t5-large-old':
            model = T5ForConditionalGeneration.from_pretrained( model_short_name2pretrain_path[model_short_name], from_flax=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path )
        print(pretrain_model_path)
    else:
        raise Exception(f'model_short_name { model_short_name} not found')



    model = model.to(device)



    '''
    Prepare trainer
    '''    
    print('Initiating Fine-Tuning for the model on our dataset')
    data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)
    train_dataset = tokenized_train_set['train']
    if do_predict:
        train_dataset = None

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset=tokenized_val_set['test'],
        data_collator=data_collator,
        compute_metrics=lambda x:compute_metrics_any_tokenizer(x, tokenizer) ,

    )


    '''
    load_weights_from
    '''  
    if load_weights_from:
        print(f'load from {load_weights_from}')
        state_dict =  torch.load(os.path.join(load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)


    '''
    If Train
    ''' 
    if do_train:
       trainer.train()

    '''
    If Predict
    ''' 
    if do_predict:
        predictions,_, test_metrics= trainer.predict(tokenized_val_set['test'])
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print(f'save to {test_metric_fn}')
        json.dump([{'output':p,
                    'converted_output': convert_full2section(p.split('Answer:')[-1].strip())} for p in decoded_preds], open(test_pred_fn  , 'w'))
        print(f'save to {test_pred_fn}')
        json.dump(test_metrics, open(test_metric_fn, 'w'))




   
