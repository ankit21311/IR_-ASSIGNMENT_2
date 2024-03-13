import common
import math
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    # Remove URLs, hashtags, and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Stopwords removal, stemming, and lemmatization
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if not word in stop_words]
    return tokens

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


text_data = common.df['Review Text'].fillna('').tolist()

# Preprocess text data
tokenized_texts = [preprocess_text(text) for text in text_data]

# Manual TF-IDF Calculation
def compute_tf_idf(docs):
    # Calculate TF (term frequency)
    tf = [{word: doc.count(word) / len(doc) for word in doc} for doc in docs]

    # Calculate document frequency (DF)
    df = {}
    for doc in docs:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    # Calculate IDF (inverse document frequency)
    idf = {word: math.log(len(docs) / freq) for word, freq in df.items()}

    # Calculate TF-IDF
    tf_idf = [{word: freq * idf[word] for word, freq in doc.items()} for doc in tf]
    return tf_idf

tf_idf_scores = compute_tf_idf(tokenized_texts)

# Save the tokenized texts and TF-IDF scores using pickle
# output_dir = 'CSE508_Winter2024_A1_2021311-main'
parent_dir = common.os.path.dirname(common.output_dir)  # Get the parent directory

outputs_dir = common.os.path.join(parent_dir, 'CSE508_Winter2024_A2_2021311-main')  # 
common.os.makedirs(outputs_dir, exist_ok=True)

tokenized_texts_path = common.os.path.join(outputs_dir, 'tokenized_texts.pkl')  # Specify the file path
tf_idf_scores_path = common.os.path.join(outputs_dir, 'tf_idf_scores_manual_text.pkl')  # Specify the file path

with open(tokenized_texts_path, 'wb') as f:
    pickle.dump(tokenized_texts, f)

with open(tf_idf_scores_path, 'wb') as f:
    pickle.dump(tf_idf_scores, f)
