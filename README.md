# MEDIQA-shared-tasks
This is our repo for MEDIQA-Sum shared tasks, 2023 (**Website**: https://www.imageclef.org/2023/medical/mediqa)

## Tasks
- **Task A - Short Dialogue2Note Summarization**: generating a section summary (section header and content) associated with the short input conversation. Section header will be one of twenty normalized section labels provided with the training data. 
- **Task B - Full Dialogue2Note Summarization**: generating a clinical note from the full input conversation. The note should include all relevant sections. Accepted first-level section headers are: "HISTORY OF PRESENT ILLNESS", "PHYSICAL EXAM", "RESULTS", "ASSESSMENT AND PLAN". 

We only attempted Task A and Task B.

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
