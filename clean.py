import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_text(text):
    text = text.replace("\n", " ")
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    text = " ".join(words)

    return text

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def save_text(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

input_file = "training_data.txt"
output_file = "preprocessed_training_data.txt"

text = read_file(input_file)
preprocessed_text = preprocess_text(text)
save_text(preprocessed_text, output_file)