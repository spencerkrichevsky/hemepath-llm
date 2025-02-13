import os
from bert_score import score
import pandas as pd
import numpy as np
 
def load_reports(cleaned_folder):
    reference_path = os.path.join(cleaned_folder, 'cleaned_msk_sample_report.txt')
    with open(reference_path, 'r', encoding='utf-8') as file:
        reference = file.read().strip()
    candidates = []
    for filename in os.listdir(cleaned_folder):
        if filename.endswith(".txt") and filename != 'cleaned_msk_sample_report.txt':
            with open(os.path.join(cleaned_folder, filename), 'r', encoding='utf-8') as file:
                candidates.append(file.read().strip())
    reference_list = [reference] * len(candidates)
    return reference_list, candidates

output_folder = './data/cleaned_reports'
references, candidates = load_reports(output_folder)

P, R, F1 = score(candidates, references, lang='en', verbose=False)
df = pd.DataFrame({'precision': np.array(P), 'recall': np.array(R), 'f1': np.array(F1)})
df.to_csv('./data/compare_raw_reports.csv')
