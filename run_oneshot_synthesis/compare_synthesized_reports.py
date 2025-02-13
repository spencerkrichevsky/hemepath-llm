import glob
import os
from bert_score import score
import pandas as pd
import numpy as np
import re
import random

## Compare N=150 synthesized reports vs N=40 real reports
# Detect real reports
real_reports_dir = './data/cleaned_reports/*.txt'
real_reports_paths = glob.glob(real_reports_dir)
real_reports_list = []
for real_report_path in real_reports_paths:
  with open(real_report_path, 'r', encoding='utf-8') as f:
    real_reports_list.append(f.read())
# Detect synthesized reports
synthesized_reports_dir = './data/synthesized_reports/*/cleaned_reports/*.txt'
synthesized_reports_paths = glob.glob(synthesized_reports_dir)
synthesized_reports_list = []
for synthesized_report_path in synthesized_reports_paths:
  with open(synthesized_report_path, 'r', encoding='utf-8') as f:
    synthesized_reports_list.append(f.read())
# Transform reference
reference_list = real_reports_list + random.choices(real_reports_list, k=len(synthesized_reports_list) - len(real_reports_list))
# Calculate performance metrics
P, R, F1 = score(synthesized_reports_list, reference_list, lang='en', verbose=False)
report_models = synthesized_reports_paths.copy()
for i in range(len(report_models)):
  report_models[i] = re.search(r'synthesized_reports/([^/]+)/cleaned_reports', report_models[i]).group(1)
df = pd.DataFrame({'model': report_models, 'precision': np.array(P), 'recall': np.array(R), 'f1': np.array(F1)})
df.to_csv('./data/stats/compare_synthensized_reports_and_real_report.csv')
