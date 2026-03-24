import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_email(text):
    text = text.lower()
    text = re.sub(r'^subject[:\s]+', '', text)
    text = re.sub(r'\b(?:\w\s){2,}\w\b', lambda x: x.group().replace(" ", ""), text)
    text = re.sub(r'\d+', ' numtoken ', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)