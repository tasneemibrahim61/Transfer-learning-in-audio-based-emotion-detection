import pandas as pd
import re
from transformers import AutoTokenizer

english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
chinese_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def detect_language(text):
    if re.search(r'[\u4e00-\u9fff]', str(text)):
        return 'chinese'
    return 'english'

def preprocess_text(text):
    text = text.lower()
    return str(text).strip()

def preprocess_csv_sentence_column(csv_path):
    df = pd.read_csv(csv_path)
    if 'Sentence' not in df.columns:
        raise ValueError("CSV file must contain a 'Sentence' column.")

    df['Preprocessed_Sentence'] = df['Sentence'].astype(str).apply(preprocess_text)

    sample_text = df['Preprocessed_Sentence'].iloc[0]
    lang = detect_language(sample_text)

    tokenizer = english_tokenizer if lang == 'english' else chinese_tokenizer

    encodings = tokenizer(
        df['Preprocessed_Sentence'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
    )

    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']

    return df[['Sentence', 'Preprocessed_Sentence']]

path = 'patho to csv file'
result_df = preprocess_csv_sentence_column(path)
print(result_df.head())
