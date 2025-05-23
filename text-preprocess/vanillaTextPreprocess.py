import pandas as pd
import re
import nltk
import jieba
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

english_stopwords = set(stopwords.words('english'))

chinese_stopwords = set([
    '的', '了', '在', '是', '我', '有', '和', '不', '就', '人', '都', '一', '一个', '上', '也', '很',
    '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '但', '给',
    '来', '我们', '为', '着', '那', '被', '与', '或', '及', '如果', '而', '并', '被', '被', '之',
    '于', '和', '与', '以', '被', '此', '所', '其', '及', '各', '还', '更', '又', '将', '把',
    '哪', '些', '呢', '吧', '吗', '啊', '呀', '呵', '唉', '呃', '嗯',
    '和', '而且', '但是', '所以', '因为', '虽然', '如果', '或者', '以及',
    '那么', '时候', '这样', '这样子', '这么', '那么', '这么样', '这么点',
    '还要', '还有', '尤其', '到底', '根本', '简直', '几乎', '差不多', '大概',
    '总是', '经常', '往往', '已经', '曾经', '正好', '突然', '忽然', '到底',
    '非常', '特别', '尤其', '最', '都', '每', '各', '任何', '每个', '自己',
])


def detect_language(text):
    """Detect if text is Chinese or English."""
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'chinese'
    return 'english'

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
        # Tokenize Chinese text
        tokens = jieba.lcut(text)
        tokens = [word for word in tokens if word not in chinese_stopwords and re.match(r'[\u4e00-\u9fff]', word)]

    return ' '.join(tokens)

def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'Sentence' not in df.columns:
        raise ValueError("CSV file must contain a 'Sentence' column.")
    
    df['Preprocessed_Sentence'] = df['Sentence'].astype(str).apply(preprocess_text)
    return df[['Sentence', 'Preprocessed_Sentence']]


path = 'path to csv file'
result_df = preprocess_csv(path)
print(result_df.head())
