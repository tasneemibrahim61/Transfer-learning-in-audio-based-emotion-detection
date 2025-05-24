import pandas as pd
import re
import nltk
import jieba
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import torch

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
english_stopwords = set(stopwords.words('english'))
chinese_stopwords = set([])  # Fill in your Chinese stopwords here

# Language detection based on presence of Chinese characters
def detect_language(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'chinese'
    return 'english'

# Text preprocessing for English & Chinese
def preprocess_text(text):
    text = str(text).strip()
    lang = detect_language(text)

    if lang == 'english':
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in english_stopwords]
        tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
    else:
        tokens = jieba.lcut(text)
        tokens = [word for word in tokens if word not in chinese_stopwords and re.match(r'[\u4e00-\u9fff]', word)]

    return ' '.join(tokens)

# Build vocabulary from tokenized sentences
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        tokens = sentence.split()
        counter.update(tokens)

    vocab = {"[PAD]": 0, "[UNK]": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# Convert sentence to token indices
def tokenize_and_index(sentence, vocab):
    return [vocab.get(token, vocab["[UNK]"]) for token in sentence.split()]

# Pad or truncate a sequence to max_len
def pad_sequence(seq, max_len, pad_value=0):
    return seq[:max_len] if len(seq) >= max_len else seq + [pad_value] * (max_len - len(seq))

# Convert all sentences to padded tensor
def process_and_index_sentences(preprocessed_sentences, vocab, max_len):
    sequences = []
    for sent in preprocessed_sentences:
        indices = tokenize_and_index(sent, vocab)
        padded = pad_sequence(indices, max_len)
        sequences.append(padded)
    return torch.tensor(sequences, dtype=torch.long)

# Create attention mask (1 for real token, 0 for PAD)
def create_attention_mask(batch_input, pad_idx=0):
    return (batch_input != pad_idx).long()

# Master function: load CSV → preprocess → vocab → tensor
def preprocess_transcript_csv(csv_path, max_len=50, min_freq=1):
    df = pd.read_csv(csv_path)
    if 'Sentence' not in df.columns:
        raise ValueError("CSV must have a 'Sentence' column.")
    
    # Preprocess each sentence
    df['Preprocessed_Sentence'] = df['Sentence'].astype(str).apply(preprocess_text)

    # Build vocab
    vocab = build_vocab(df['Preprocessed_Sentence'].tolist(), min_freq=min_freq)

    # Index and pad
    tensor_input = process_and_index_sentences(df['Preprocessed_Sentence'], vocab, max_len)

    # Create attention mask
    attention_mask = create_attention_mask(tensor_input)

    return df[['Sentence', 'Preprocessed_Sentence']], vocab, tensor_input, attention_mask
