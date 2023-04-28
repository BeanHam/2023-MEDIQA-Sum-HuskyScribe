import numpy as np
import torch
import wandb
import nltk
import evaluate
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, concatenate_datasets, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

#-----------------------
# Main Function
#-----------------------
def main():
    
    wandb.login()
    
    #-----------------------
    # Complementary Function
    #-----------------------
    def preprocess_function(sample,padding="max_length"):
    
        # add prefix to the input for t5
        max_source_length=512
        max_target_length=512
        inputs = ["summarize: " + item for item in sample["dialogue"]]
    
        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["section_text"], max_length=max_target_length, padding=padding, truncation=True)
    
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # helper function to postprocess text
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
    
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
    
        return preds, labels
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        #result = {k: round(v * 100, 4) for k, v in result.items()}
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    
    #-----------------------
    # Load Tokenizer & Data
    #-----------------------
    model_checkpoint = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = load_dataset('csv', 
                    data_files={'train': ['../mediqa-data/TaskA/TaskA-TrainingSet.csv',
                                          '../mediqa-data/external/samsum.csv',
                                          '../mediqa-data/external/dialoguesum.csv',
                                         ],
                                'val': '../mediqa-data/TaskA/TaskA-ValidationSet.csv'
                               }
                   )
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "section_text"])
    nltk.download('punkt')
    metric = load_metric("rouge")
    
    #-----------------------
    # Load Model
    #-----------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
        
    #-----------------------
    # Configuration
    #-----------------------
    args = Seq2SeqTrainingArguments(
        output_dir="mediqa-flan-t5",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        fp16=False,
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_strategy="steps",
        logging_steps=5000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False
    )
    
    #-----------------------
    # Trainer
    #-----------------------
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    #-----------------------
    # Evaluate
    #-----------------------
    trainer.evaluate()
    
    #-----------------------
    # Finalize
    #-----------------------
    wandb.finish()
    
if __name__ == "__main__":
    main()
