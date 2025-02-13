import spacy
import pandas as pd
from tqdm import tqdm

# Load the pre-trained spacy model
nlp = spacy.load('en_core_web_sm')
# Define sets for diagnosis, stage, disease presence, etc.
diagnosis_set = {'acute myeloid leukemia', 'aml', 'acute promyelocytic leukemia', 'apl', 'aplastic anemia', 'b-cell acute lymphoblastic leukemia', 'b cell acute lymphoblastic leukemia', 'b cell lymphoblastic leukemia', 'chronic lymphocytic leukemia', 'cll', 'chronic myeloid leukemia', 'cml', 'chronic myelomonocytic leukemia', 'cmml', 'diffuse large b-cell lymphoma', 'diffuse large b cell lymphoma', 'dlbcl', 'essential thrombocytosis', 'follicular lymphoma', 'follicular', 'hemophagocytic lymphohistiocytosis', 'hodgkin lymphoma', 'immune thrombocytopenic purpura', 'itp', 'lymphoplasmacytic lymphoma', 'myelodysplastic syndrome', 'mds', 'myelofibrosis', 'mf', 'myeloproliferative neoplasm nos', 'mpn nos', 'mpn-nos', 'non hodgkin lymphoma', 'normal', 'plasma cell myeloma', 'pcm', 'polycythemia vera', 'pv', 't-cell acute lymphoblastic leukemia', 't cell acute lymphoblastic leukemia'}
stage_set = {'first presentation', 'minimal residual disease', 'progression', 'relapse', 'remission', 'undergoing treatment'}

# Negation words
negation_words = {'no', 'not', 'without', 'none', 'never', 'nothing', 'absent'}

# Read in the annotations file
annotations = pd.read_csv('./data/annotations/annotations.csv')

def extract_diagnosis(report):
  # Search for diagnosis
  for diagnosis in diagnosis_set:
    if diagnosis.lower() in report.lower():
      return diagnosis.lower()

def extract_stage(report):
  for stage in stage_set:
    if stage.lower() in report.lower():
      return stage.lower()

def detect_negation(doc, keywords):
  for token in doc:
    # Check if the token is a negation word and nearby keyword
    if token.text.lower() in negation_words:
      for next_token in doc[token.i+1:]:
        if next_token.text.lower() in keywords:
          return True

def extract_aspirate_disease(report):
  if any(section in report.lower() for section in ['bone marrow aspirate', 'aspirate smear']):
    doc = nlp(report)
    keywords = ['dysplasia', 'dysplastic', 'abnormalities', 'abnormality', 'disease', 'abnormal cells', 'morphology']
    if detect_negation(doc, keywords):
      return False
    for token in doc:
      if token.text.lower() in keywords:
        return True
    return False

def extract_pblood_disease(report):
  if any(section in report.lower() for section in ['peripheral blood', 'peripheral blood smear']):
    doc = nlp(report)
    keywords = ['dysplasia', 'dysplastic', 'abnormalities', 'abnormality', 'disease', 'abnormal cells', 'morphology']
    if detect_negation(doc, keywords):
      return False
    for token in doc:
      if token.text.lower() in keywords:
        return True
    return False

def extract_flow_blasts(report):
  if 'flow cytometry' in report.lower():
    doc = nlp(report)
    keywords = ["blasts", "myeloblasts", "increased blasts", "myeloblasts increased"]
    if detect_negation(doc, keywords):
      return 'Normal'
    for token in doc:
      if token.text.lower() in keywords:
        return 'Increased'
    return 'Normal'


# Initialize lists
dx_list, stage_list, aspirate_disease_list, pblood_disease_list, flow_summary_list, flow_blasts_list = [],[],[],[],[],[]

# Generate responses
for i in tqdm(range(annotations.shape[0])):
  dx_list.append(extract_diagnosis(annotations['report'][i]))
  stage_list.append(extract_stage(annotations['report'][i]))
  aspirate_disease_list.append(extract_aspirate_disease(annotations['report'][i]))
  pblood_disease_list.append(extract_pblood_disease(annotations['report'][i]))
  flow_blasts_list.append(extract_flow_blasts(annotations['report'][i]))

# Add summaries to annotations file
annotations['dx_nlp'] = dx_list
annotations['stage_blp'] = stage_list
annotations['aspirate_disease_nlp'] = aspirate_disease_list
annotations['pblood_disease_nlp'] = pblood_disease_list
annotations['flow_blasts_nlp'] = flow_blasts_list
annotations.to_csv('./data/annotations/nlp_annotations.csv')
