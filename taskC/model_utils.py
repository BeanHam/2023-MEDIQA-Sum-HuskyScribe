import os
from t5_io_utils import *
import json
from model_constants import  SECTION2FULL
import transformers

def convert_full2section(pred):
    full2section = { v.lower() :k for k,v in SECTION2FULL.items() }
    return full2section.get(pred.lower(), pred.lower()).lower()

def download_url_to_file(url, file):
    if os.path.exists(file):
        print(f'model already downloaded to {file}')
    else:
        print(f'downloading model from {url} to {file}')


def t5_subsection_classfier(list_of_dialogue_chunks):

    # write labels_in_batch in file as test_file
    # run python script to predict
    # python trian_t5 eval model_ckp, 


    '''
    Overview
    --------
    Apply section classifier model to list of document snippet list as 
    multi-label classification problem in batch
    Parameters
    ----------
    list_of_dialogue_chunks : list of document snippet list
    Ouputs
    ------
    labels_in_batch : list of normalized section name str list
    '''
    
    #-----------------------------------
    # urls 
    #-----------------------------------
    write_dialogue_chunk_list(list_of_dialogue_chunks, 'tmp/t5_test.csv')
    model_ckp = 'finetuned-model'
    model_short_name = 'clinical-t5-scratch'
    pretrain_path = f"../models/Clinical-T5-Scratch"
    ckp_path = f"../models/{model_ckp.strip('/').replace('/','_')}/{transformers.WEIGHTS_NAME}"
    ckp_dir = os.path.dirname(ckp_path)
    
    #-----------------------------------
    # remove existing predictions 
    #-----------------------------------
    #pred_fn = f'{model_short_name}_tmp/predictions_test.json'
    pred_fn = 'tmp/taskB_predictions_test.json'
    if os.path.exists(pred_fn):
        print(f'delete existing prediction file {pred_fn}')
        os.remove(pred_fn)    
    
    #-----------------------------------
    # prediction 
    #-----------------------------------
    print('Running T5 model to predict labels for list of dialogue chunk list')

    # T5 clinical
    cmd_clinical = f"""python train_t5_with_trainer.py with \
              do_predict=True do_train=False do_predict=True \
              load_weights_from='{ckp_dir}' \
              task_name='tmp' max_epochs=0 \
              train_csv=tmp/t5_test.csv \
              val_csv=tmp/t5_test.csv \
              model_short_name={model_short_name} \
              pretrain_model_path={pretrain_path} \
              test_pred_fn={pred_fn} """
    

    # T5 base
    model_short_name = 't5-base'
    pretrain_path='sitongz/medqa_taskB_t5-base_seq_synthetic_onl-checkpoint-11000'
    cmd_t5 = f"""python train_t5_with_trainer.py with \
            do_predict=True do_train=False do_predict=True \
            load_weights_from='' \
            task_name='tmp' max_epochs=0 \
            train_csv=tmp/t5_test.csv \
            val_csv=tmp/t5_test.csv \
            model_short_name={model_short_name} \
            pretrain_model_path={pretrain_path} \
            test_pred_fn={pred_fn}"""
    cmd = cmd_t5
    print("run",cmd)

    out = os.system(cmd)
    print('finish section prediction')

    if out!=0:
        print("run",cmd)
        print('Error in running T5 model')
        quit()
    
    #-----------------------------------
    # save output
    #-----------------------------------
    #pred_fn = f'{model_short_name}_tmp/predictions_test.json'
    preds = json.loads(open(pred_fn).read())
    preds = [pred['output'].split('|')[0].strip().replace('topic:','') for pred in preds]
    preds = [convert_full2section(p) for p in preds]
    return preds



def t5_canonical_section_classfier(list_of_dialogue_chunks):

    # write labels_in_batch in file as test_file
    # run python script to predict
    # python trian_t5 eval model_ckp, 


    '''
    Overview
    --------
    Apply section classifier model to list of document snippet list as 
    multi-label classification problem in batch
    Parameters
    ----------
    list_of_dialogue_chunks : list of document snippet list
    Ouputs
    ------
    labels_in_batch : list of normalized section name str list
    '''
    
    #-----------------------------------
    # urls 
    #-----------------------------------
    write_dialogue_chunk_list(list_of_dialogue_chunks, 'tmp/t5_test_canonical.csv', ontology_version = 'canonical')
    #-----------------------------------
    # remove existing predictions 
    #-----------------------------------
    pred_fn = 'tmp/taskB_predictions_test_canonical.json'
    if os.path.exists(pred_fn):
        print(f'delete existing prediction file {pred_fn}')
        os.remove(pred_fn)    
    
    #-----------------------------------
    # prediction 
    #-----------------------------------
    print('Running T5 model to predict labels for list of dialogue chunk list')

    # T5 base

    #val_csv='/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_filter30_dev.csv'
    model_short_name = 't5-base'
    pretrain_path='sitongz/medqa_taskB_t5-base_seq_synthetic_onl-checkpoint-11000'
    tmp_fn = 'tmp/t5_test_canonical.csv'
    #load_weights_from='sitongz/medqa_taskB_t5-base_seq_synthetic_onl-checkpoint-11000'
    load_weights_from='/home/sitongz/2023-MEDIQA-shared-tasks/taskB/exp/t5-base_train_multilabel_filter30/model/checkpoint-198'
    # assume run in taskB
    load_weights_from='exp/t5-base_train_multilabel_filter30/model/checkpoint-198'

    cmd_t5 = f"""python train_t5_with_trainer.py with \
            do_predict=True do_train=False do_predict=True \
            load_weights_from={load_weights_from} \
            task_name='tmp' max_epochs=0 \
            train_csv={tmp_fn} \
            val_csv={tmp_fn} \
            model_short_name={model_short_name} \
            pretrain_model_path={pretrain_path} \
            test_pred_fn={pred_fn}"""
    cmd = cmd_t5
    print("run",cmd)

    out = os.system(cmd)
    if out!=0:
        print("run",cmd)
        print('Error in running T5 model')
        quit()
    
    #-----------------------------------
    # save output
    #-----------------------------------
    preds = json.loads(open(pred_fn).read())
    preds = [pred['output'].split('|')[0].strip().replace('topic:','') for pred in preds]
    preds = [convert_full2section(p) for p in preds]
    return preds
