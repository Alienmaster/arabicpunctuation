from pathlib import Path
import re
from datasets import Dataset

def _read_files(folder):
    p = Path(folder).glob("*")
    files = [x for x in p if x.is_file()]
    return files

def remove_characters(text, regex=r"[~$&+,،:;=?@#|'<>.^*()%!\-\[\]]"):
    new_text = re.sub(regex, " ", text)
    new_text = " ".join(new_text.split())
    return new_text

def _read_data(files, tokenizer):
    def prepro(example):
        # print(tokenizer(example["input_ids"], truncation=True, padding=True))
        # exit()
        # print(f'{tokenizer(example["labels"], truncation=True)=}')
        # print(f'{tokenizer(example["input_ids"], truncation=True)=}')
        example['reference'] = example['labels']
        example["input_ids"] = tokenizer(example["text"], padding='max_length')['input_ids']
        example["labels"] = tokenizer(example["labels"], padding='max_length')['input_ids']
        return example
    # def create_conv(sample):
    #     return {"prompt":sample["input_ids"], "completion":sample["labels"]}

    cleaned_data = []
    for file in files:
        with open(file) as f:
            data = f.read()
        data = data.split("\n")
        for line in data:
            if len(line) < 1:
                continue
            line = re.sub(r"[a-zA-Z]+[a-zA-Z0-9]+", #Remove PageV01B02 marks
                        " ",
                        line)
            line = re.sub(r"\|+", "|", line)
            line_w_punctuation = remove_characters(line, r"[~$&+=?@#|'<>^*()%!\-\[\]]") #ToDo: Dont remove %~ Erhalten: ;,.،
            line_wo_punctuation = remove_characters(line)
            if len(line) <= 2:
                continue
            if len(line_wo_punctuation) <= 2:
                continue
            cleaned_data.append({"text":line_wo_punctuation, "labels": line_w_punctuation})
    ds = Dataset.from_list(cleaned_data)
    # print(ds[10])
    ds = ds.map(prepro)
    ds.set_format('torch')
    # ds = ds.map(create_conv, remove_columns=ds.features, batched=False)
    return ds

def create_dataset(folder, tokenizer):
    files = _read_files(folder)
    ds = _read_data(files, tokenizer)
    ds = ds.train_test_split(test_size=0.1, seed=8)
    return ds

def write_dataset(ds=Dataset):
    ds.to_json('dataset.json')