pretrain_path='sitongz/medqa_taskB_t5-base_seq_synthetic_onl-checkpoint-11000'
model_short_name='t5-base'
train_csv='/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_train.csv' 
val_csv='/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_dev.csv'
task_name=train_multilabel


train_csv='/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_filter30_train.csv' 
val_csv='/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_filter30_dev.csv'
task_name=train_multilabel_filter30
cmd="""python train_t5_with_trainer.py with \
            do_predict=True do_train=True  \
            load_weights_from='' \
            task_name=${task_name} max_epochs=10 \
            train_csv=${train_csv} \
            val_csv=${val_csv} \
            model_short_name=${model_short_name} \
            pretrain_model_path=${pretrain_path}"""
echo $cmd

#python train_t5_with_trainer.py with do_predict=True do_train=True load_weights_from='' task_name=train_multilabel max_epochs=5 train_csv=/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_train.csv val_csv=/home/sitongz/2023-MEDIQA-shared-tasks/taskB/t5_multilable_data_dev.csv model_short_name=t5-base pretrain_model_path=sitongz/medqa_taskB_t5-base_seq_synthetic_onl-checkpoint-11000

