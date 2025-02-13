from google import genai
import pandas as pd
import os
from tqdm import tqdm
import time

# Initialize Gemini client
client = genai.Client(api_key=[API Key])

# Read in the sampled annotations file
annotations = pd.read_csv('./data/annotations/sampled_annotations.csv')

# Set the prompts
dx_prompt = 'Extract the diagnosis reported in the following bone marrow pathology report, where the returned value must exist within the set {"acute myeloid leukemia", "acute promyelocytic leukemia", "aplastic anemia", "b-cell acute lymphoblastic leukemia", "chronic lymphocytic leukemia", "chronic myeloid leukemia", "chronic myelomonocytic leukemia", "diffuse large b-cell lymphoma", "essential thrombocytosis", "follicular lymphoma", "hemophagocytic lymphohistiocytosis", "hodgkin lymphoma", "immune thrombocytopenic purpura", "lymphoplasmacytic lymphoma", "myelodysplastic syndrome", "myelofibrosis", "myeloproliferative neoplasm NOS", "non hodgkin lymphoma", "normal," plasma cell myeloma", "polycythemia vera", "t-cell acute lymphoblastic leukemia"} without any additional information: '                                                   
stage_prompt = 'Extract the staging reported in the following bone marrow pathology report, where the returned value must exist within the set {"first presentation", "minimal residual disease", "progression", "relapse" "remission", "undergoing treatment"} without any additional information: '                                
aspirate_disease_prompt = 'Extract from the following bone marrow pathology report whether there is disease detected specifically in components of the bone marrow aspirate section, including characteristics such as dysplasia or morphological abnormalities. If these characteristics are identified in the bone marrow aspirate section then return "True", if these characteristics are not identified in the bone marrow aspirate section or only identified in sections other than the bone marrow aspirate then return "False", and if the bone marrow aspirate section is not analyzed then return "NA", and do not return any additional information: '
pblood_disease_prompt = 'Extract from the following bone marrow pathology report whether there is disease detected specifically in components of the peripheral blood specimen section, including characteristics such as dysplasia or morphological abnormalities. If these characteristics are identified in the peripheral blood specimen section then return "True", if these characteristics are not identified in the peripheral blood specimen section or only identified in sections other than the peripheral blood specimen section then return "False", and if the peripheral blood specimen section is not analyzed then return "NA", and do not return any additional information: '
flow_summary_prompt = 'Extract from the following bone marrow pathology report any findings reported specifically in components of the flow cytometry section as a single string and do not return any additional information: '
flow_blasts_prompt = 'Extract from the following bone marrow pathology report whether there are increased blasts or myeloblasts reported specifically in the flow cytometry section. The returned value must exist within the set {"Normal", "Increased"}, and if the diagnosis reported is "acute myeloid leukemia" then the returned value is likely "Increased", and do not return any additional information: '

# Set the finetuning data
finetuning_dat = []
for i in range(annotations.shape[0]):
  finetuning_dat.append([str(dx_prompt + annotations['report'][i]), str(annotations['dx'][i])])
  finetuning_dat.append([str(stage_prompt + annotations['report'][i]), str(annotations['stage'][i])])
  finetuning_dat.append([str(aspirate_disease_prompt + annotations['report'][i]), str(annotations['aspirate_disease'][i])])
  finetuning_dat.append([str(pblood_disease_prompt + annotations['report'][i]), str(annotations['pblood_disease'][i])])
  finetuning_dat.append([str(flow_summary_prompt + annotations['report'][i]), str(annotations['flow'][i])])
  finetuning_dat.append([str(flow_blasts_prompt + annotations['report'][i]), str(annotations['flow_blasts'][i])])
finetuning_dat = genai.types.TuningDataset(examples=[genai.types.TuningExample(text_input=i, output=o,) for i,o in finetuning_dat],)
tuning_job = client.tunings.tune(base_model='models/gemini-1.0-pro-001', training_dataset=finetuning_dat, config=genai.types.CreateTuningJobConfig(epoch_count=5, batch_size=4, learning_rate=0.001, tuned_model_display_name='Gemini 1.0 Pro Summarization v1.0'))

# Read in the sampled annotations file
heldout_df = pd.read_csv('./data/annotations/heldout_annotations.csv')

# Initialize lists
dx_list, stage_list, aspirate_disease_list, pblood_disease_list, flow_summary_list, flow_blasts_list = [],[],[],[],[],[]

# Generate responses
for i in tqdm(range(heldout_df.shape[0])):
  dx_list.append(client.models.generate_content(model=tuning_job.tuned_model.model, contents=dx_prompt+heldout_df['report'][i]).text)
  time.sleep(5)
  stage_list.append(client.models.generate_content(model=tuning_job.tuned_model.model, contents=stage_prompt+heldout_df['report'][i]).text)
  time.sleep(5)
  aspirate_disease_list.append(client.models.generate_content(model=tuning_job.tuned_model.model, contents=aspirate_disease_prompt+heldout_df['report'][i]).text)
  time.sleep(5)
  pblood_disease_list.append(client.models.generate_content(model=tuning_job.tuned_model.model, contents=pblood_disease_prompt+heldout_df['report'][i]).text)
  time.sleep(5)
  flow_summary_list.append(client.models.generate_content(model=tuning_job.tuned_model.model, contents=flow_summary_prompt+heldout_df['report'][i]).text)
  time.sleep(5)
  flow_blasts_list.append(client.models.generate_content(model=tuning_job.tuned_model.model, contents=flow_blasts_prompt+heldout_df['report'][i]).text)
  time.sleep(5)

# Add summaries to annotations file
heldout_df['dx_gemini'] = dx_list
heldout_df['stage_gemini'] = stage_list
heldout_df['aspirate_disease_gemini'] = aspirate_disease_list
heldout_df['pblood_disease_gemini'] = pblood_disease_list
heldout_df['flow_gemini'] = flow_summary_list
heldout_df['flow_blasts_gemini'] = flow_blasts_list             
heldout_df.to_csv('./data/annotations/gemini_heldout_annotations.csv')
