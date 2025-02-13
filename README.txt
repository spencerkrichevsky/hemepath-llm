# This is the codespace for 'Application of LLMs for Bone Marrow Report Generation and Structured Data Extraction' developed by Spencer Krichevsky (2025) using Ubuntu 22.04.3

# Dependencies
# Steps:

# 1. Setup the environment install
conda create -n your_env python=3.10
conda activate your_env
git clone x.git
cd x/
pip install -e .
pip install -r requirements.txt

# 2. Compare raw reports: sample report (N=1) and educational templates (N=40)
# Source of educational templates: https://hemepathreview.com/ReportTemplates/HemeInterps-BoneMarrow.htm
cd compare_raw_reports/
# Code below cleans raw reports - e.g., removes escape sequences, unnecessary punctuation
python3.10 clean_reports.py
# Code below computes BERTScore metrics
python3.10 compare_raw_reports.py
# Sample reports are already contained in ./compare_raw_reports/data/raw_reports and ./compare_raw_reports/data/cleaned_reports
# Inspect ./compare_raw_reports/data/compare_raw_reports.csv for results

# 3. Run one-shot report synthesis 
cd ..
cp -r compare_raw_reports/data/cleaned_reports/*.txt run_oneshot_synthesis/data/cleaned_reports/
cd run_oneshot_synthesis/
# Obtain API key to run Google Gemini model: https://ai.google.dev/gemini-api/docs/api-key
# Insert Google API key into run_oneshot_gemini.py
python3.10 run_oneshot_gemini.py
# Obtain API key to run OpenAI GPT model: https://platform.openai.com/api-keys
# Insert OpenAI API key into run_oneshot_gpt.py
python3.10 run_oneshot_gpt.py
# Obtain API key to run Anthropic Claude model: https://console.anthropic.com/settings/keys
# Insert Anthropic API key into run_oneshot_claude.py
python3.10 run_oneshot_claude.py
# Code below cleans raw reports synthesized by all three models
python3.10 clean_reports.py
# Code below computes BERTScore metrics
python3.10 compare_synthesized_reports.py
# Inspect ./run_oneshot_synthesis/data/stats/compare_synthensized_reports_and_real_report.csv for results

# 4. Run zero-shot report summarization
cd ..
cd run_zeroshot_summarization/
# Note: manually derived annotations are made available in: ./data/annotations/annotations.csv . These data will not match synthesized reports produced by running Step 3.
# Insert Google API key into run_zeroshot_gemini.py
python3.10 run_zeroshot_gemini.py
# Insert OpenAI API key into run_zeroshot_gpt.py
python3.10 run_zeroshot_gpt.py
# Insert Anthropic API key into run_zeroshot.claude.py
python3.10 run_zeroshot_claude.py
# Code below will run regular-expression NLP
python3.10 run_nlp.py
# Code below will compare summarizations across all models and synthesization sources
pythone.10 compare_summarizations.py
# Inspect ./run_zeroshot_summarization/data/stats/compare_summarizations.csv for results



