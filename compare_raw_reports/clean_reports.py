import re
import os

def clean_text(text):
    # Remove escape characters like \n, \t, etc.
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    # Remove extra spaces and other delimiters
    text = re.sub(r'[^\w\s.,;]', ' ', text)  # Keeps words, spaces, and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
    return text

def process_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"cleaned_{filename}")
            with open(input_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
            cleaned_text = clean_text(raw_text)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_text)
            print(f"Processed: {filename} -> {output_path}")

# MSK report + Nguyen reports = Raw reports (reference set)
input_folder = './data/raw_reports'
output_folder = './data/cleaned_reports'
process_files(input_folder, output_folder)
