# MEDIQA-shared-tasks
This is our repo for MEDIQA-Sum shared tasks, 2023 (**Website**: https://www.imageclef.org/2023/medical/mediqa)

## Tasks
- **Subtask A - Dialogue2Topic Classification**.  Given a conversation snippet between a doctor and patient, participants are tasked with identifying the topic (associated section header). Topics/Section headers will be one of twenty normalized common section labels (e.g. Assessment, Diagnosis, Exam, Medications, Past Medical History).
- **Subtask B - Dialogue2Note Summarization**. Given a conversation snippet between a doctor and patient and a section header, participants are tasked with producing a clinical note section text summarizing the conversation.
- **Subtask C - Full-Encounter Dialogue2Note Summarization**. Given a full encounter conversation between a doctor and patient, participants are tasked with producing a full clinical note summarizing the conversation

## Inference 
To run our model on the test dataset, do the following:

1. To set up the environment, clone this repo and do the following cmd lines:
```
## install.sh creates the environment & install all packages
source ./install.sh
## activate.sh activates the environment
source ./activate.sh
``` 

2. To run Task A, do the following cmd lines:
```
cd taskA
source ./decode_taskA_run1.sh
```
The output will be a .csv file: "taskA_HuskyScribe_run1.csv"

3. To run Task B, do the following cmd lines:
```
cd taskB
source ./decode_taskB_run1.sh
```
The output will be a .csv file: "taskB_HuskyScribe_run1.csv"
