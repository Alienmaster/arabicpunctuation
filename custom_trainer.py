# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb
import json
# from pathlib import Path


from datasets import load_dataset, Dataset, tqdm
import evaluate
from peft import get_peft_model, LoraConfig, PeftModel
from termcolor import colored
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline, DataCollatorWithPadding
# from trl import SFTConfig, SFTTrainer

from preprocessing import create_dataset, remove_characters, write_dataset

train = True

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "inceptionai/jais-family-2p7b"
model_name = "inceptionai/jais-adapted-7b"

# Configuration
rank = 512
alpha = 512
dropout = 0.05
quant = False
# lr = 0.

new_model = f'{model_name}_{rank}_{alpha}_{dropout}{"_quant" if quant else ""}_ownLoss'.replace('/', '_') 

bnb_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    load_in_4bit=True,
    # torch_dtype=torch.float16
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=4096)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
# print(dir(tokenizer))
# exit()
if quant:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", quantization_config=bnb_config)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
# model, tokenizer = setup_chat_format(model, tokenizer)

ds = create_dataset('original', tokenizer)
ds_train = ds["train"]
ds_test = ds["test"]
# write_dataset(ds_train)

# with grad 78gb
# wo grad 58gb
for param in model.parameters():
    param.requires_grad=False

lora_config = LoraConfig(
    lora_alpha=alpha,
    lora_dropout=dropout,
    r=rank,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=["c_attn"]
    target_modules=['k_proj', 'q_proj', 'v_proj']
)
model = get_peft_model(model, lora_config)


# run = 

training_arguments = TrainingArguments(
    output_dir="new_model_punct",
    logging_steps=1,
    per_device_train_batch_size=1,
    report_to="wandb",
    num_train_epochs=3,
    do_train=True,
    gradient_accumulation_steps=8,
    
)

def nll_loss(logits, labels):
    return torch.nn.NLLLoss(logits, labels)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # reference, prediction, word_mask, punctuation_mask, alpha=0.1):
        # Custom loss function to penalize word-errors more than punctuation errors.

        # Parameters:
        # - reference: Tensor of reference (ground truth) token IDs.
        # - prediction: Tensor of predicted token IDs.
        # - word_mask: Boolean mask where True indicates a word, False indicates punctuation.
        # - punctuation_mask: Boolean mask where True indicates punctuation, False indicates a word.
        # - alpha: Weight for the punctuation errors. Should be 0 < alpha < 1.

        # Returns:
        # - loss: The computed loss value.
        # - prediction: The prediction tensor (for further use if needed).

        # Compute Cross-Entropy loss for the entire batch.
        # Note: Adjust this based on your token vocabulary size.
        reference = inputs.pop('labels')
        outputs = model(**inputs)
        prediction = outputs.logits
        vocab_size = tokenizer.vocab_size
        shift_logits = prediction[..., :-1, :].contiguous()
        shift_labels = reference[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels)
        
        # Word errors - penalized normally
        # word_loss = loss * word_mask.view(-1)
        # word_loss = word_loss.sum() / word_mask.sum().float()
        
        # Punctuation errors - penalized less
        # punctuation_loss = loss * punctuation_mask.view(-1)
        # punctuation_loss = (alpha * punctuation_loss.sum()) / punctuation_mask.sum().float()
        
        # Total loss is a combination of both
        total_loss = loss
        # total_loss = word_loss + punctuation_loss
        
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model = model,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    # peft_config=lora_config,
    # dataset_text_field="input_ids",
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
)

def predict(ds, model, tokenizer):
    y_pred = []
    tqdm_obj = tqdm(ds)
    for sample in tqdm_obj:
        text = sample['text']
        reference = sample['reference']
        # print(dir(model))
        input_ids = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length').input_ids
        inputs = input_ids.to('cuda')
        # text = sample.pop("text")
        # inputs = sample.pop('input_ids').to('cuda')
        input_len = inputs.shape[-1]
        generate_ids = model.generate(inputs,
                                 min_length= input_len + 4,
                                 top_p=0.9,
                                temperature=0.3,
                                # max_length=200-input_len,
                                # min_length=input_len + 4,
                                repetition_penalty=1.2,
                                do_sample=True,
                                max_length=4096)
        # print(f'{generate_ids=}')
        response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
        print(response)
        print(reference)
        # print(f'{reference=}')
        # print(f'{text=}')
        # print(f'{response=}')
        y_pred.append({"prediction":response,"reference":reference})
    return y_pred

def compute_metrics(y_pred):
    wer = evaluate.load("wer")
    predictions = [remove_characters(x["prediction"]) for x in y_pred]
    references = [remove_characters(x["reference"]) for x in y_pred]
    wer_score = wer.compute(predictions=predictions, references=references)
    print(wer_score)

if train:
    trainer.train()
    trainer.save_model(new_model)
    tokenizer.save_pretrained(new_model, save_embedding_layers=True)
# exit()

# Evaluate
model =  PeftModel.from_pretrained(
        model=model,
        model_id=new_model,
        device_map="auto",
        trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(new_model, model_max_length=4095)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# ds_test = ds_test.train_test_split(test_size=0.1)
# ds_test = ds_test["train"]
results = predict(ds_test, model, tokenizer)

with open("results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False)

compute_metrics(results)


# TODO: Eval correct text

# class WeightedCELossTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         # Get model's predictions
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # Compute custom loss
#         loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([class_weights["neg"], class_weights["pos"], class_weights["neu"]], device=model.device, dtype=logits.dtype))
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss
