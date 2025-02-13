from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import json

def format_entry(prompt, response):
  return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}

# Initialize OpenAI client
client = OpenAI(api_key=[API Key])

# Read in the sampled annotations file
annotations = pd.read_csv('./data/annotations/sampled_annotations.csv')

# Set the prompts
dx_prompt = 'Extract the diagnosis reported in the following bone marrow pathology report, where the returned value must exist within the set {"acute myeloid leukemia", "acute promyelocytic leukemia", "aplastic anemia", "b-cell acute lymphoblastic leukemia", "chronic lymphocytic leukemia", "chronic myeloid leukemia", "chronic myelomonocytic leukemia", "diffuse large b-cell lymphoma", "essential thrombocytosis", "follicular lymphoma", "hemophagocytic lymphohistiocytosis", "hodgkin lymphoma", "immune thrombocytopenic purpura", "lymphoplasmacytic lymphoma", "myelodysplastic syndrome", "myelofibrosis", "myeloproliferative neoplasm NOS", "non hodgkin lymphoma", "normal," plasma cell myeloma", "polycythemia vera", "t-cell acute lymphoblastic leukemia"} without any additional information: '                                                   
stage_prompt = 'Extract the staging reported in the following bone marrow pathology report, where the returned value must exist within the set {"first presentation", "minimal residual disease", "progression", "relapse" "remission", "undergoing treatment"} without any additional information: '                                
aspirate_disease_prompt = 'Extract from the following bone marrow pathology report whether there is disease detected specifically in components of the bone marrow aspirate section, including characteristics such as dysplasia or morphological abnormalities. If these characteristics are identified in the bone marrow aspirate section then return "True", if these characteristics are not identified in the bone marrow aspirate section or only identified in sections other than the bone marrow aspirate then return "False", and if the bone marrow aspirate section is not analyzed then return "NA", and do not return any additional information: '
pblood_disease_prompt = 'Extract from the following bone marrow pathology report whether there is disease detected specifically in components of the peripheral blood specimen section, including characteristics such as dysplasia or morphological abnormalities. If these characteristics are identified in the peripheral blood specimen section then return "True", if these characteristics are not identified in the peripheral blood specimen section or only identified in sections other than the peripheral blood specimen section then return "False", and if the peripheral blood specimen section is not analyzed then return "NA", and do not return any additional information: '
flow_summary_prompt = 'Extract from the following bone marrow pathology report any findings reported specifically in components of the flow cytometry section as a single string and do not return any additional information: '
flow_blasts_prompt = 'Extract from the following bone marrow pathology report whether there are increased blasts or myeloblasts reported specifically in the flow cytometry section. The returned value must exist within the set {"Normal", "Increased"}, and if the diagnosis reported is "acute myeloid leukemia" then the returned value is likely "Increased", and do not return any additional information: '

# Organize the finetuning data
input_dat = []
output_dat = []
for i in range(annotations.shape[0]):
  input_dat.append(str(dx_prompt + annotations['report'][i]))
  output_dat.append(str(annotations['dx'][i]))
  input_dat.append(str(stage_prompt + annotations['report'][i]))
  output_dat.append(str(annotations['stage'][i]))
  input_dat.append(str(aspirate_disease_prompt + annotations['report'][i]))
  output_dat.append(str(annotations['aspirate_disease'][i]))
  input_dat.append(str(pblood_disease_prompt + annotations['report'][i]))
  output_dat.append(str(annotations['pblood_disease'][i]))
  input_dat.append(str(flow_blasts_prompt + annotations['report'][i]))
  output_dat.append(str(annotations['flow_blasts'][i]))
finetuning_dat = pd.DataFrame({'prompt': input_dat, 'response': output_dat})
# Write fine-tuning data to JSONL
output_file = './data/annotations/gpt.jsonl'
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in finetuning_dat.iterrows():
        entry = format_entry(row["prompt"], row["response"])
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
# Upload finetuning file
job = client.files.create(file=open('./data/annotations/gpt.jsonl', 'rb'), purpose='fine-tune')
# Create fine-tuned model
# Note: the step below can take ~20 minutes. Check progress on the OpenAI platform before proceeding
ft = client.fine_tuning.jobs.create(training_file=job.id, model='gpt-4o-mini-2024-07-18')

# Read in the sampled annotations file
heldout_df = pd.read_csv('./data/annotations/heldout_annotations.csv')
heldout_dat = []
for i in range(annotations.shape[0]):
  heldout_dat.append(str(dx_prompt + heldout_df['report'][i]))
  heldout_dat.append(str(stage_prompt + heldout_df['report'][i]))
  heldout_dat.append(str(aspirate_disease_prompt + heldout_df['report'][i]))
  heldout_dat.append(str(pblood_disease_prompt + heldout_df['report'][i]))
  heldout_dat.append(str(flow_blasts_prompt + heldout_df['report'][i]))
heldout_dat = pd.DataFrame({'prompt': heldout_dat})
# Write heldout data to JSONL
with open('./data/annotations/heldout_gpt.jsonl', "w", encoding="utf-8") as f:
  for _, row in finetuning_dat.iterrows():
    entry = format_entry(row["prompt"], row["response"])
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Generate responses
model_id = [MODEL ID]
# Initialize lists
output_list = []
for i in tqdm(range(heldout_df.shape[0])):
  output_list.append(client.chat.completions.create(model=model_id, messages=[heldout_dat[i]]).choices[0].message)
dx_list, stage_list, aspirate_disease_list, pblood_disease_list, flow_blasts_list = [],[],[],[],[],[]
sublists = [dx_list, stage_list, aspirate_disease_list, pblood_disease_list, flow_blasts_list]
for i, entry in enumerate(output_list):
  sublists[i%5].append(entry)

# Add summaries to annotations file
heldout_df['dx_gemini'] = dx_list
heldout_df['stage_gemini'] = stage_list
heldout_df['aspirate_disease_gemini'] = aspirate_disease_list
heldout_df['pblood_disease_gemini'] = pblood_disease_list
heldout_df['flow_gemini'] = flow_summary_list
heldout_df['flow_blasts_gemini'] = flow_blasts_list             
heldout_df.to_csv('./data/annotations/gpt_heldout_annotations.csv')
