# Task A: Section Classifier

The Task A model is a section classifier that identifies the topic of a section of a medical dialogue. The model is trained using a T5 model and is based on a question-answering format. 
- The T5 input is constructed using the format:

```
{dialogue} Question: what is the section topic among categories below? topic categories: general_history | medications | chief_complaints | past_medical_history | allergy | family_and_social_history | past_surgical | other_history | assessment | review_of_system | disposition | exam | plan | diagnosis | emergency_department | immunizations | labs | imaging | procedures | gynecological_history

```
- The T5 output is constructed using the format:

```
Topic: {gold section of the current dialogue} 
```
The implementation details of the model are as follows:

The model is initialized with T5-Large or Clinical-T5-Large, which can be found at https://physionet.org/content/clinical-t5/1.0.0/.
The batch size is set to 16 (2 per step * 8 accumulated step / batch).
The step is set to 2000 (epoch 13.3).

- The T5 large model can be found in huggingface
```sitongz/medqa_taskA_t5-large_topic_whole_update_ed-checkpoint-2000```

- The  clinical T5 large model require physient credential.
  * tokenizer :  can be downloaded from the following URL: 
  https://drive.google.com/drive/folders/1EySXjUEZsBdYJ-547ewdutF_x-DAxD94?usp=share_link
  * The model checkpoint can be downloaded from the following URL: 
https://drive.google.com/file/d/1bnhFtrHMQK1-LselKfqAQMGw07pI2uDN/view?usp=share_link

To run the model, you can use the following commands:

```
# Option 1: clinical T5 large
pretrain_model_path=download_models/Clinical-T5-Large load_weights_from=download_models/clinical-t5-large_topic_whole_update_ed_model_checkpoint-2000
model_short_name='clinical-t5-large' 

# Option 2: T5 large (huggingface)
pretrain_model_path=sitongz/medqa_taskA_t5-large_topic_whole_update_ed-checkpoint-2000
load_weights_from=''
model_short_name='t5-large' 

python convert_taskA_topic_input_to_t5_input.py with in_fn=/home/sitongz/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv \
out_fn=tmp/taskA_test_input.csv

cmd="""python train_t5_with_trainer.py with do_predict=True do_train=False task_name='topic_whole_update_ed_predict_step2000'   train_csv='tmp/taskA_test_input.csv' val_csv='tmp/taskA_test_input.csv' model_short_name=${model_short_name} load_weights_from=${load_weights_from} \
pretrain_model_path=${pretrain_model_path} \
test_pred_fn='tmp/taskA_test_output.csv'"""
echo ${cmd}
To score the model, you can use the following command:


python scoring_taskA_topic.py with gold_fn='/home/sitongz/medqa/notebooks/data/medqa_topic_whole_taskA_Validation_full_t5_update.csv' pred_fn='clinical-t5-large_topic_whole_update_ed_predict_step2000/predictions_test.json'
Task B: Section Summarization
```


The Task B model is a section summarization model that identifies the section topic of a dialogue turn, given the context of the previous and next dialogue turns. The model is trained using a BERT-based



# Task B summarization


We format task B section classification into question answering tasks for T5 modeling. The task involves identifying the section topic of a dialogue turn, given the context of the previous and next dialogue turns.

- dialogue chunker:
The script uses data from a task A dataset and chunks the dialogues into doctor-patient turns by identifying the speaker as 'Doctor:' or '[Doctor]:'. 

The T5 input is then constructed by concatenating the previous two doctor-patient turns, the current doctor-patient turn, and the next doctor-patient turn. The T5 output is the topic of the current section, and a binary classification of whether the current dialogue turn is a new topic or not. We provide the full topic category list to provide schema




## clinical T5 scratch trained on randomly shuffled data: 
`absoluate_path=/home/sitongz/medqa/clinical-t5-scratch_seq_30_10k/model/checkpoint-4500`


- The T5 input is constructed using the format:
```previous section: {previous two doctor-patient turns} current section: {current doctor-patient turn} next section: {next doctor-patient turn} Question: what is the current section topic? and Is the current section a different topic from the previous section? Topic categories: GENERAL HISTORY | MEDICATIONS | CHIEF COMPLAINTS | PAST MEDICAL HISTORY | ALLERGY | FAMILY AND SOCIAL HISTORY | PAST SURGICAL | OTHER_HISTORY | ASSESSMENT | REVIEW OF SYSTEM | DISPOSITION | EXAM | PLAN | DIAGNOSIS | ED COURSE | IMMUNIZATIONS | LABS | IMAGING | PROCEDURES | GYNECOLOGICAL HISTORY "```

- The T5 output is constructed using the format:

```Topic: {topic of the current section} | new topic: {yes or no depending on whether the current section is a new topic or not}```

- training data:
randomly shuffle taskA training data by section headers to 30 times larger. Limit each section header category examples within 10K.

- implementation details:
        * initialized with Clinical-T5-Scratch in https://physionet.org/content/clinical-t5/1.0.0/
          require to be credentialed user on Physionet
        * hyperparamter:
        * epoch 0.24

## Clinical-T5-scratch on  synthetic data sampled with similar section order as task B
`/home/sitongz/medqa/clinical-t5-scratch_seq_synthetic_only/model/checkpoint-5000`

- The T5 input is constructed using the format:
```previous section: {previous two doctor-patient turns} current section: {current doctor-patient turn} next section: {next doctor-patient turn} Question: what is the current section topic? and Is the current section a different topic from the previous section? Topic categories: GENERAL_HISTORY | MEDICATIONS | CHIEF_COMPLAINTS | PAST_MEDICAL_HISTORY | ALLERGY | FAMILY_AND_SOCIAL_HISTORY | PAST_SURGICAL | OTHER_HISTORY | ASSESSMENT | REVIEW_OF_SYSTEM | DISPOSITION | EXAM | PLAN | DIAGNOSIS | EMERGENCY_DEPARTMENT | IMMUNIZATIONS | LABS | IMAGING | PROCEDURES | GYNECOLOGICAL_HISTORY ",```
- T5 output: 

```Topic: CHIEF_COMPLAINTS | new topic: no"```


- training data:
 sampled taskA training data by section headers to 8 times larger, using similar order as observed  in task B dialogue.  


- implementation details:
        * initialized with Clinical-T5-Scratch in https://physionet.org/content/clinical-t5/1.0.0/
                require to be credentialed user on Physionet

        * epoch 0.25



## T5-base on  synthetic data sampled with similar section order as task B

- huggingface location:
`sitongz/medqa_taskB_t5-base_seq_synthetic_onl-checkpoint-11000`
- training data: 
same as above
- implementation details:
initialized with T5-large. epoch: 2.16







