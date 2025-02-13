from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI(api_key=[API Key])

# Read in the annotations file
annotations = pd.read_csv('./data/annotations/annotations.csv')

# Set the prompts
dx_prompt = 'Extract the diagnosis reported in the following bone marrow pathology report, where the returned value must exist within the set {"acute myeloid leukemia", "acute promyelocytic leukemia", "aplastic anemia", "b-cell acute lymphoblastic leukemia", "chronic lymphocytic leukemia", "chronic myeloid leukemia", "chronic myelomonocytic leukemia", "diffuse large b-cell lymphoma", "essential thrombocytosis", "follicular lymphoma", "hemophagocytic lymphohistiocytosis", "hodgkin lymphoma", "immune thrombocytopenic purpura", "lymphoplasmacytic lymphoma", "myelodysplastic syndrome", "myelofibrosis", "myeloproliferative neoplasm NOS", "non hodgkin lymphoma", "normal," plasma cell myeloma", "polycythemia vera", "t-cell acute lymphoblastic leukemia"} without any additional information: '
stage_prompt = 'Extract the staging reported in the following bone marrow pathology report, where the returned value must exist within the set {"first presentation", "minimal residual disease", "progression", "relapse" "remission", "undergoing treatment"} without any additional information: '
aspirate_disease_prompt = 'Extract from the following bone marrow pathology report whether there is disease detected specifically in components of the bone marrow aspirate section, including characteristics such as dysplasia or morphological abnormalities. If these characteristics are identified in the bone marrow aspirate section then return "True", if these characteristics are not identified in the bone marrow aspirate section or only identified in sections other than the bone marrow aspirate then return "False", and if the bone marrow aspirate section is not analyzed then return "NA", and do not return any additional information: '
pblood_disease_prompt = 'Extract from the following bone marrow pathology report whether there is disease detected specifically in components of the peripheral blood specimen section, including characteristics such as dysplasia or morphological abnormalities. If these characteristics are identified in the peripheral blood specimen section then return "True", if these characteristics are not identified in the peripheral blood specimen section or only identified in sections other than the peripheral blood specimen section then return "False", and if the peripheral blood specimen section is not analyzed then return "NA", and do not return any additional information: '
flow_summary_prompt = 'Extract from the following bone marrow pathology report any findings reported specifically in components of the flow cytometry section as a single string and do not return any additional information: '
flow_blasts_prompt = 'Extract from the following bone marrow pathology report whether there are increased blasts or myeloblasts reported specifically in the flow cytometry section. The returned value must exist within the set {"Normal", "Increased"}, and if the diagnosis reported is "acute myeloid leukemia" then the returned value is likely "Increased", and do not return any additional information: '

# Initialize lists
dx_list, stage_list, aspirate_disease_list, pblood_disease_list, flow_summary_list, flow_blasts_list = [],[],[],[],[],[]

# Generate responses
for i in tqdm(range(annotations.shape[0])):
  dx_list.append(client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': dx_prompt+annotations['report'][i]}]).choices[0].message.content)
  stage_list.append(client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': stage_prompt+annotations['report'][i]}]).choices[0].message.content)
  aspirate_disease_list.append(client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': aspirate_disease_prompt+annotations['report'][i]}]).choices[0].message.content)
  pblood_disease_list.append(client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': pblood_disease_prompt+annotations['report'][i]}]).choices[0].message.content)
  flow_summary_list.append(client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': flow_summary_prompt+annotations['report'][i]}]).choices[0].message.content)
  flow_blasts_list.append(client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': flow_blasts_prompt+annotations['report'][i]}]).choices[0].message.content)

# Add summaries to annotations file
annotations['dx_gpt'] = dx_list
annotations['stage_gpt'] = stage_list
annotations['aspirate_disease_gpt'] = aspirate_disease_list
annotations['pblood_disease_gpt'] = pblood_disease_list
annotations['flow_gpt'] = flow_summary_list
annotations['flow_blasts_gpt'] = flow_blasts_list
annotations.to_csv('./data/annotations/gpt_annotations.csv')

