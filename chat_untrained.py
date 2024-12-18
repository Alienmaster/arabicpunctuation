# -*- coding: utf-8 -*-
import json

import evaluate
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor

from preprocessing import create_dataset, remove_characters, write_dataset


model_path = "inceptionai/jais-adapted-13b-chat"

prompt_eng = "### Instruction: Please reconstruct the punctuation for the following Arabic text. Your task is to accurately place all necessary punctuation marks, including commas, periods, question marks, and quotation marks. The goal is to enhance the clarity and readability of the text without altering the original words or their order. Below is the Arabic text that needs punctuation:\n### Input: أخبرنا عبد الأول قال أخبرنا الداودي قال أخبرنا يحيى بن زكريا قال أخبرنا الليث عن عقيل قال قال ابن شهاب أخبرني عروة عن عائشة قالت لبث رسول الله صلى الله عليه وسلم في بني عمرو بن عوف بضع عشرة ليلة وأسس المسجد الذي أسس على التقوى وصلى فيه رسول الله صلى الله عليه وسلم ثم ركب راحلته فسار يمشي معه الناس حتى بركت عند مسجد رسول الله صلى الله عليه وسلم بالمدينة وهو يصلي فيه رجال من المسلمين وكان مربدا للتمر لسهل وسهيل غلامين يتيمين في حجر أسعد بن زرارة فقال رسول الله صلى الله عليه وسلم حين بركت «هذا إن شاء الله المنزل» ثم دعا الغلامين فساومهما بالمربد ليتخذه مسجدا فقالا بل نهبه لك يا رسول الله ثم بناه مسجدا وطفق ينقل معهم اللبن في بنيانه ويقول\n[|AI|]\n### Response : أخبرنا عبد الأول قال: أخبرنا الداودي قال: أخبرنا يحيى بن زكريا قال: أخبرنا الليث، عن عقيل قال: قال ابن شهاب: أخبرني عروة، عن عائشة قالت: لبث رسول الله، صلى الله عليه وسلم، في بني عمرو بن عوف بضع عشرة ليلة، وأسس المسجد الذي أسس على التقوى، وصلى فيه رسول الله، صلى الله عليه وسلم ، ثم ركب راحلته، فسار يمشي معه الناس حتى بركت عند مسجد رسول الله، صلى الله عليه وسلم، بالمدينة وهو يصلي فيه رجال من المسلمين، وكان مربدا للتمر لسهل وسهيل غلامين يتيمين في حجر أسعد بن زرارة، فقال رسول الله، صلى الله عليه وسلم، حين بركت: «هذا إن شاء الله. المنزل». ثم دعا الغلامين فساومهما بالمربد ليتخذه مسجدا. فقالا: بل نهبه لك يا رسول الله، ثم بناه مسجدا، وطفق ينقل معهم اللبن في بنيانه ويقول:\n[|AI|] ### Input: {text}\n[|AI|]\n### Response : "
# prompt_ar = "### Instruction:اسمك \"جيس\" وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception في الإمارات. أنت مساعد مفيد ومحترم وصادق. أجب دائمًا بأكبر قدر ممكن من المساعدة، مع الحفاظ على البقاء أمناً. أكمل المحادثة بين [|Human|] و[|AI|] :\n### Input:[|Human|] {Question}\n[|AI|]\n### Response :"
# أخبرنا عبد الأول قال: أخبرنا الداودي قال: أخبرنا يحيى بن زكريا قال: أخبرنا الليث، عن عقيل قال: قال ابن شهاب: أخبرني عروة، عن عائشة قالت: لبث رسول الله، صلى الله عليه وسلم، في بني عمرو بن عوف بضع عشرة ليلة، وأسس المسجد الذي أسس على التقوى، وصلى فيه رسول الله، صلى الله عليه وسلم ، ثم ركب راحلته، فسار يمشي معه الناس حتى بركت عند مسجد رسول الله، صلى الله عليه وسلم، بالمدينة وهو يصلي فيه رجال من المسلمين، وكان مربدا للتمر لسهل وسهيل غلامين يتيمين في حجر أسعد بن زرارة، فقال رسول الله، صلى الله عليه وسلم، حين بركت: «هذا إن شاء الله. المنزل». ثم دعا الغلامين فساومهما بالمربد ليتخذه مسجدا. فقالا: بل نهبه لك يا رسول الله، ثم بناه مسجدا، وطفق ينقل معهم اللبن في بنيانه ويقول:
# أخبرنا عبد الأول قال أخبرنا الداودي قال أخبرنا يحيى بن زكريا قال أخبرنا الليث عن عقيل قال قال ابن شهاب أخبرني عروة عن عائشة قالت لبث رسول الله صلى الله عليه وسلم في بني عمرو بن عوف بضع عشرة ليلة وأسس المسجد الذي أسس على التقوى وصلى فيه رسول الله صلى الله عليه وسلم ثم ركب راحلته فسار يمشي معه الناس حتى بركت عند مسجد رسول الله صلى الله عليه وسلم بالمدينة وهو يصلي فيه رجال من المسلمين وكان مربدا للتمر لسهل وسهيل غلامين يتيمين في حجر أسعد بن زرارة فقال رسول الله صلى الله عليه وسلم حين بركت «هذا إن شاء الله المنزل» ثم دعا الغلامين فساومهما بالمربد ليتخذه مسجدا فقالا بل نهبه لك يا رسول الله ثم بناه مسجدا وطفق ينقل معهم اللبن في بنيانه ويقول
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "Please reconstruct the punctuation for the following Arabic text. Your task is to accurately place all necessary punctuation marks, including commas, periods, question marks, and quotation marks. The goal is to enhance the clarity and readability of the text without altering the original words or their order. Below is the Arabic text that needs punctuation:\n"
few_shot = [
    {'role': 'user',
         "content": f"{prompt}أخبرنا عبد الأول قال أخبرنا الداودي قال أخبرنا يحيى بن زكريا قال أخبرنا الليث عن عقيل قال قال ابن شهاب أخبرني عروة عن عائشة قالت لبث رسول الله صلى الله عليه وسلم في بني عمرو بن عوف بضع عشرة ليلة وأسس المسجد الذي أسس على التقوى وصلى فيه رسول الله صلى الله عليه وسلم ثم ركب راحلته فسار يمشي معه الناس حتى بركت عند مسجد رسول الله صلى الله عليه وسلم بالمدينة وهو يصلي فيه رجال من المسلمين وكان مربدا للتمر لسهل وسهيل غلامين يتيمين في حجر أسعد بن زرارة فقال رسول الله صلى الله عليه وسلم حين بركت «هذا إن شاء الله المنزل» ثم دعا الغلامين فساومهما بالمربد ليتخذه مسجدا فقالا بل نهبه لك يا رسول الله ثم بناه مسجدا وطفق ينقل معهم اللبن في بنيانه ويقول"},
    {'role': 'assistant',
         "content": "أخبرنا عبد الأول قال: أخبرنا الداودي قال: أخبرنا يحيى بن زكريا قال: أخبرنا الليث، عن عقيل قال: قال ابن شهاب: أخبرني عروة، عن عائشة قالت: لبث رسول الله، صلى الله عليه وسلم، في بني عمرو بن عوف بضع عشرة ليلة، وأسس المسجد الذي أسس على التقوى، وصلى فيه رسول الله، صلى الله عليه وسلم ، ثم ركب راحلته، فسار يمشي معه الناس حتى بركت عند مسجد رسول الله، صلى الله عليه وسلم، بالمدينة وهو يصلي فيه رجال من المسلمين، وكان مربدا للتمر لسهل وسهيل غلامين يتيمين في حجر أسعد بن زرارة، فقال رسول الله، صلى الله عليه وسلم، حين بركت: «هذا إن شاء الله. المنزل». ثم دعا الغلامين فساومهما بالمربد ليتخذه مسجدا. فقالا: بل نهبه لك يا رسول الله، ثم بناه مسجدا، وطفق ينقل معهم اللبن في بنيانه ويقول:"}
]

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)



if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class PunctuationProcessor(LogitsProcessor):
    def __init__(self, tokenizer, text):
        self.tokenizer = tokenizer
        self.to_consume = text
        self.tokens = tokenizer.tokenize(text)
        print(f'{self.tokens=}')
        all_tokens = [tokenizer(x, add_special_tokens=False)['input_ids'] for x in self.tokens]
        self.to_consume_ids = [y for sub in all_tokens for y in sub if y != 29871]
        print(self.to_consume_ids)
        self.last_id = None
        self.consumed_text = []

        # possible delimiter
        self.punct = {k: v for k, v in tokenizer.vocab.items() if k.endswith('.')}
        self.punct = {"." : 29889}
        self.comma = {k: v for k, v in tokenizer.vocab.items() if k.endswith(',')}
        self.comma = {"," : 29892}
        self.comma2 = {k: v for k, v in tokenizer.vocab.items() if k.endswith('،')}
        # self.arabicquestion = {k: v for k, v in tokenizer.vocab.items() if k.endswith('؟')}
        self.question = {k: v for k, v in tokenizer.vocab.items() if k.endswith('?')}
        print(self.question)
        # self.colon = {k: v for k, v in tokenizer.vocab.items() if k.endswith(':')}
        self.colon = {":" : 29901}
        self.exclamationmark = {k: v for k, v in tokenizer.vocab.items() if k.endswith('!')}
        self.exclamationmark = {"!" : 29991}
        self.underscore = {'▁' : 29871} # _ (29918), ▁ (29871)
        self.interpunct = self.punct | self.comma | self.comma2 | self.exclamationmark | self.colon # | self.underscore # | self.arabicquestion
        print(f'{self.interpunct.values()}')
        self.previous_interpunct = False

    def check_prev(self, generated):
        previous_round = generated[0,-1]
        if self.last_id:
            if previous_round == self.last_id:
                print('Last token was correct! Lets get back on track!')
                self.consumed_text.append(self.last_id)
                self.last_id = None
                self.previous_interpunct = False
            elif previous_round in self.interpunct.values():
                print(f'Last round was an interpunctuation!')
                print(f'Last generated ID: {previous_round} Token: {self.tokenizer.decode([previous_round])}')
                print(f'Last token in text: {self.tokenizer.decode([self.last_id])}')
                self.to_consume_ids.insert(0, self.last_id)
                self.last_id = None
                self.previous_interpunct = True
            else:
                print('weird')
                print(f'Last token in text: {self.tokenizer.decode([self.last_id])}')
                print(f'Last generated ID: {previous_round} Token: {self.tokenizer.decode([previous_round])}')
                exit()

    def decide_token(self, generated, scores, just_generated):
        # if self.previous_interpunct == False:
        #     # At this point we know for sure the token is not an interpunct and the not the correct token
        #     allowed_ids = [torch.tensor([x for x in self.interpunct.values()], dtype=torch.int)]
        # else:
        #     print('Next round cant be an interpunct!')
        allowed_ids = []
        
        # We store the actual token to decide in the next round if the token was a text token or interpunctuation
        self.last_id = self.to_consume_ids.pop(0) # This is checked in prev_round if this is a good idea
        
        actual_id = int(self.last_id)
        actual_word = self.tokenizer.decode([actual_id])
        print(f'{actual_id=}')
        print(f'Correct:\t{actual_word}')
        print(f'Predicted:\t{just_generated}')
        
        try:
            to_copy = torch.tensor([actual_id])
            allowed_ids.append(to_copy)
            # print(f'{allowed_ids=}')
            # allowed.append(torch.tensor(to_copy).unsqueeze(0))
            # allowed_interpunct.append(torch.tensor(to_copy2).unsqueeze(0))
        except IndexError:
            to_copy = None
        
        allowed_ids = torch.cat(allowed_ids)
        
        mask = torch.zeros_like(scores)
        mask[0, allowed_ids] = 1
        mask[0, self.tokenizer.eos_token_id] = 0
        
        scores.masked_fill_(mask == 0, float("-inf"))
        allowed_ids_after_mask = (scores != float("-inf")).nonzero()[:,1]
        print(allowed_ids_after_mask)
        print(f'{self.tokenizer.decode(allowed_ids_after_mask)=}')
        
        
        return scores
        # else:
        #     try:
        #         # to_copy = tokenized_to_consume[0]
        #         to_copy2 = tokenized_to_consume[0]
        #         to_copy3 = torch.tensor(tokenized_to_consume)
        #         allowed_interpunct = []
        #         allowed_interpunct.append(to_copy3)
        #         # allowed.append(torch.tensor(to_copy).unsqueeze(0))
        #         # allowed_interpunct.append(torch.tensor(to_copy2).unsqueeze(0))
        #     except IndexError:
        #         to_copy = None
        #     allowed_tokens = torch.cat(allowed_interpunct)
        #     mask = torch.zeros_like(scores)
        #     mask[0, allowed_tokens] = 1
        #     # if len(self.tokens) == 0:
        #     #     mask[0, self.tokenizer.eos_token_id] = 1
        #     # else:
        #     # mask[0, :] = 1
        #     # print("Allowed tokens:", self.tokenizer.convert_ids_to_tokens(allowed_tokens))
        #     print("Allowed tokens length:", len(allowed_tokens))
        #     self.previous_interpunct = False
    
    def __call__(self, generated, scores):
        print(f'{len(generated[0])=}')
        self.check_prev(generated)
        if generated.shape[-1] > 0:
            just_generated_id = generated[0,-1]
            just_generated_token = self.tokenizer.convert_ids_to_tokens([just_generated_id], skip_special_tokens=True)
            print(f'{just_generated_id=}')
            print(f'{just_generated_token=}')
            if len(self.to_consume_ids) > 0: # Still text to process
                if (just_generated_id == 29871):
                    print('Underscore. Continue')
                if (just_generated_id == self.to_consume_ids[0]): # Correct token. Consume one token go on
                    print(f'Correct token. Next. {just_generated_id=} {just_generated_token=}')
                    self.consumed_text.append(self.to_consume_ids.pop(0))
                
                elif just_generated_id in self.interpunct.values(): # and not self.previous_interpunct: # Check if token is a interpunctuation (yay). Go on
                    print('Thats an interpunct!')
                
                # elif (just_generated_id == self.to_consume_ids[1]): # Correct token but one position off. Consume one token go on
                #     print(f'One off token but correct. Next. {just_generated_id=} {just_generated_token=}')
                    
                #     pop = self.to_consume_ids.pop(0)
                #     print(f'{pop=}')
                #     self.consumed_text.append(pop)
                #     pop = self.to_consume_ids.pop(0)
                #     print(f'{pop=}')
                #     self.consumed_text.append(pop)
                
                else: # Wrong token. Change scores
                    scores = self.decide_token(generated, scores, just_generated_id)
                
                    # self.previous_interpunct = False
                # next_id = self.to_consume_ids.pop(0)
                # print(f'Correct ID: {next_id}')
                # print(f'Predicted ID: {just_generated_id}')
                    # self.previous_interpunct = True
            elif just_generated_id in self.interpunct.values(): # No text left but interpunctuation is coming
                    print('Thats an interpunct! Coming to an End!')
                    mask = torch.zeros_like(scores)
                    mask[0, self.tokenizer.eos_token_id] = 0
                    scores.masked_fill_(mask == 0, float("-inf"))
            else: # No text to consume anymore
                mask = torch.zeros_like(scores)
                mask[0, self.tokenizer.eos_token_id] = 1
                scores.masked_fill_(mask == 0, float("-inf"))
                pass
        print(30*'-')
        return scores

def get_response(text, tokenizer=tokenizer, model=model):
    tokenized = tokenizer(text, return_tensors="pt")
    input_ids, attention_mask = tokenized['input_ids'].to(device), tokenized['attention_mask'].to(device)
    input_len = input_ids.shape[-1]
    print(f'{text=}')
    generate_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        logits_processor=LogitsProcessorList([processor]),
        # top_p=0.9,
        # temperature=0.3,
        max_length=2048,
        min_length=input_len + 4,
        # repetition_penalty=1.2,
        # do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    # print(f'{generate_ids['sequences']=}')
    ids = generate_ids['sequences']
    response = tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response :")[-1].lstrip()
    return response


# text = prompt_ar.format_map({'Question': ques})
# print(get_response(text))
ds = create_dataset('original', tokenizer)
ds_train = ds["train"]
ds_test = ds["test"]

y_pred = []
tqdm_obj = tqdm(ds_test)
for sample in tqdm_obj:
    input_text = sample['text']
    reference = sample['reference']
    processor = PunctuationProcessor(tokenizer, input_text)
    actual_text = {"role": "user",
                   "content": f"{prompt}{input_text}"}
    chat_template = list(few_shot)
    chat_template.append(actual_text)
    chat_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    
    # text = prompt_eng.format_map({'text': input_text})
    response = get_response(chat_prompt)
    y_pred.append({"prediction":response,"reference":reference})

def compute_metrics(y_pred):
    wer = evaluate.load("wer")
    predictions = [remove_characters(x["prediction"]) for x in y_pred]
    references = [remove_characters(x["reference"]) for x in y_pred]
    wer_score = wer.compute(predictions=predictions, references=references)
    print(wer_score)

with open("results_untrained_13b_res_1.json", "w") as f:
    json.dump(y_pred, f, ensure_ascii=False)

compute_metrics(y_pred)
# ques = "What is the capital of UAE?"
# text = prompt_eng.format_map({'text': ques})
# print(get_response(text))
