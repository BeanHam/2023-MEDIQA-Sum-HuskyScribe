import numpy as np
import torch
import wandb
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

#-----------------------
# Evaluation Metrics
#-----------------------
def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

#-----------------------
# Main Function
#-----------------------
def main():
    
    wandb.login()
    
    #-----------------------
    # Preprocessing Function
    #-----------------------
    def preprocess_function(examples,
                            max_input_length = 512,
                            max_target_length = 128):
        inputs = ['\n'.join(doc.split('\r\n')) for doc in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["section_text"], max_length=max_target_length, truncation=True)
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    
    #-----------------------
    # Load Tokenizer & Data
    #-----------------------
    model_checkpoint = "facebook/bart-large-xsum"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data = load_dataset('csv', 
                    data_files={'train': 'mediqa-data/TaskA/TaskA-TrainingSet.csv',
                                'val': 'mediqa-data/TaskA/TaskA-ValidationSet.csv'
                               }
                   )
    tokenized_datasets = data.map(preprocess_function, batched=True)
    nltk.download('punkt')
    
    #-----------------------
    # Load Model
    #-----------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    #-----------------------
    # Configuration
    #-----------------------
    args = Seq2SeqTrainingArguments(
        "val-dialogue-summarization",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
        report_to="wandb",  # enable logging to W&B
        run_name="mediqa-dialogue-summarization"  # name of the W&B run (optional)
    )
    
    #-----------------------
    # Trainer
    #-----------------------
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
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
