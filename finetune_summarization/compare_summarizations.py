import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Define file info
file_info = {'./data/annotations/gemini_heldout_annotations.csv': 'gemini',
             './data/annotations/gpt_heldout_annotations.csv': 'gpt'}
dfs = []
for file, identifier in file_info.items():
  df = pd.read_csv(file)
  df['source'] = identifier
  df = df.loc[:, ~df.columns.str.contains('Unnamed', case=False)]
  dfs.append(df)
combined_df = pd.concat(dfs, ignore_index=True)

# Handle NaN and ensure correct data types
categorical_cols = ['dx', 'stage', 'flow_blasts']
binary_cols = ['aspirate_disease', 'pblood_disease']
for col in categorical_cols:
  combined_df[col].fillna('Unknown', inplace=True)
  combined_df[col] = combined_df[col].astype(str).str.strip()
for col in binary_cols:
  combined_df[col].fillna(False, inplace=True)
  combined_df[col] = combined_df[col].astype(bool)

# Compute AUC and F1
sources = ['nguyen', 'gemini', 'gpt', 'claude']
models = ['gemini', 'gpt']
results = []
for source in sources:
  for model in models:
    temp = combined_df[(combined_df['subset']==source)&(combined_df['source']==model)]
    for col in categorical_cols + binary_cols:
      y_true = temp[col]
      y_pred = temp[f"{col}_{model}"]
      if col in categorical_cols:
        y_pred.fillna('Unknown', inplace=True)
        y_pred = y_pred.astype(str).str.strip()
      else:
        y_pred.fillna(False, inplace=True)
        y_pred = y_pred.astype(bool)
      lb = LabelBinarizer()
      y_true_bin = lb.fit_transform(y_true)
      y_pred_bin = lb.transform(y_pred)
      f1 = f1_score(y_true, y_pred, average='weighted')
      if len(lb.classes_) > 2:
        auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
      else:
        auc = roc_auc_score(y_true_bin.ravel(), y_pred_bin.ravel())
      results.append({'ReportSource': source, 'ModelType': model, 'Metric': col, 'F1': f1, 'AUC': auc})
df = pd.DataFrame(results)
df.to_csv('./data/stats/compare_summarizations.csv')
