import json
import re

import evaluate
wer = evaluate.load("wer")

def remove_characters(text, regex=r"[~$&+,:;=?@#|'<>.^*()%!-\[\]]"):
    new_text = re.sub(regex, " ", text)
    new_text = new_text.replace("  ", " ")
    return new_text

def compute_metrics(y_pred, clean=False):
    wer = evaluate.load("wer")
    if clean:
        predictions = [remove_characters(x["prediction"]) for x in y_pred]
        references = [remove_characters(x["reference"]) for x in y_pred]
    else:
        predictions = [x["prediction"] for x in y_pred]
        references = [x["reference"] for x in y_pred]
    wer_score = wer.compute(predictions=predictions, references=references)
    print(wer_score)


def evaluate_character(y_pred, character):
    import re
    wer = evaluate.load("wer")
    predictions = [x["prediction"] for x in y_pred]
    references = [x["reference"] for x in y_pred]
    pred_clean = []
    ref_clean = []
    for prediction in predictions:
        clean = re.sub(rf'[^\{character}\s]', '_', prediction)
        clean = re.sub(r'[\_]+', '_', clean)
        clean = re.sub(rf'[\{character}]', 'C', clean)
        pred_clean.append(clean)
    # print(pred_clean)
    for reference in references:
        clean_ref = re.sub(rf'[^\{character}\s]', '_', reference)
        clean_ref = re.sub(r'[\_]+', '_', clean_ref)
        clean_ref = re.sub(rf'[\{character}]', 'C', clean_ref)
        ref_clean.append(clean_ref)
    # print(ref_clean)
    values = {'words':0, 'tp':0, 'fp':0, 'fn':0, 'tn':0}
    for p,r in zip(pred_clean, ref_clean):
        for p_c, r_c in zip(p, r):
            if (p_c == 'C') and (r_c == 'C'):
                values['tp']+=1
            elif (p_c == 'C') and (r_c != 'C'):
                values['fp']+=1
            elif (p_c != 'C') and (r_c == 'C'):
                values['fn']+=1
            elif (p_c != 'C') and (r_c != 'C'):
                values['tn']+=1
            else:
                print('crazy, innit?')
    # print(ref_clean)
    print(values)
    wer_score = wer.compute(predictions=pred_clean, references=ref_clean)
    print(wer_score)
    tp = values['tp']
    tn = values['tn']
    fp = values['fp']
    fn = values['fn']
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision * recall) / (precision + recall)
    print(f'{accuracy=}')
    print(f'{precision=}')
    print(f'{recall=}')
    print(f'{f1=}')

with open("results.json", "r") as f:
    data = json.load(f)

compute_metrics(data)
# evaluate_character(data, '،')
# evaluate_character(data, ',')
evaluate_character(data, '.')

# print(data)
