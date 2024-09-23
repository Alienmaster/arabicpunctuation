import torch
import transformers
from transformers import Conv1D, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig
from trl import SFTConfig, SFTTrainer, setup_chat_format
model_name = 'inceptionai/jais-adapted-7b'
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        print(name)
        print(type(module))
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names

print(list(set(get_specific_layer_names(model))))