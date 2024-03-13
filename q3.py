

import pickle
import common
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
from scipy.spatial.distance import cosine
import torch
from torchvision import models, transforms
import requests
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re


image_features_path = common.os.path.join(common.output_dir, 'image_features.pkl')
tf_idf_scores_path = common.os.path.join(common.output_dir, 'tf_idf_scores_manual_text.pkl')

# Load the dataset
df = common.pd.read_csv(common.dataset_path)

# Load precomputed image features and TF-IDF scores
with open(image_features_path, 'rb') as f:
    image_features = pickle.load(f)
with open(tf_idf_scores_path, 'rb') as f:
    tf_idf_scores = pickle.load(f)

resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()  # Set the model to evaluation mode

# Define image transformations
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def cosine_similarity(v1, v2):

  if isinstance(v1, common.np.ndarray) and all(isinstance(v, common.np.ndarray) for v in v2):
          # Convert list of numpy arrays (v2) to a single 2D numpy array
          v2 = common.np.array(v2)
          # Normalize v1 and v2
          v1_norm = v1 / common.np.linalg.norm(v1)
          v2_norm = v2 / common.np.linalg.norm(v2, axis=1)[:, common.np.newaxis]
          # Calculate cosine similarity
          similarities = common.np.dot(v1_norm, v2_norm.T)

      # Case for sparse vectors (TF-IDF scores)
  elif isinstance(v1, dict) and all(isinstance(v, dict) for v in v2):
          similarities = []
          for tfidf_dict in v2:
              # Intersection of keys (terms present in both vectors)
              common_terms = set(v1.keys()) & set(tfidf_dict.keys())
              # Manual dot product for common terms
              dot_product = sum(v1[term] * tfidf_dict[term] for term in common_terms)
              # Norms of the vectors
              norm_v1 = common.np.sqrt(sum(value ** 2 for value in v1.values()))
              norm_v2 = common.np.sqrt(sum(value ** 2 for value in tfidf_dict.values()))
              # Cosine similarity
              if norm_v1 == 0 or norm_v2 == 0:
                  similarity = 0
              else:
                  similarity = dot_product / (norm_v1 * norm_v2)
              similarities.append(similarity)
          similarities = common.np.array(similarities)

  return similarities



def most_similar_images(processed_image, precomputed_features, top_n=3):
    # Assuming direct comparison of processed_image array to precomputed feature vectors
    similarities = cosine_similarity(processed_image, precomputed_features).flatten()
    top_indices = common.np.argsort(similarities)[-top_n:][::-1]
    return top_indices, [similarities[a] for a in top_indices]

def most_similar_reviews(input_tfidf, precomputed_tfidf_scores, top_n=3):
    # Calculate cosine similarity between the input TF-IDF vector and each precomputed TF-IDF vector
    similarities = cosine_similarity(input_tfidf, precomputed_tfidf_scores).flatten()
    top_indices = common.np.argsort(similarities)[-top_n:][::-1]
    top_similarities = [similarities[a] for a in top_indices]
    return top_indices, top_similarities

def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize by splitting on whitespace
    # Optionally remove stopwords here
    return tokens
def compute_tf(tokenized_review):
    tf = {}
    for word in tokenized_review:
        tf[word] = tf.get(word, 0) + 1

    # Normalize term frequencies by the total number of words in the document
    total_words = len(tokenized_review)
    tf = {word: count / total_words for word, count in tf.items()}

    return tf

def preprocess_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert('RGB')
        processed_image = transform_pipeline(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = resnet_model(processed_image)
        features_np = features.numpy().flatten()
        return features_np
    else:
        print(f"Error: Unable to fetch image from URL {image_url}. Status code: {response.status_code}")
        return None

## INPUT ##
input_image_url = input("Please enter the image URL: ")
input_review_text = input("Please enter the review text: ")


# Preprocess review text
processed_tokens = preprocess_text(input_review_text)
input_review_tfidf = compute_tf(processed_tokens)

# Preprocess the image from URL
processed_image = preprocess_image(input_image_url)

if processed_image is not None:

    # Find the most similar images and reviews
    similar_image_indices, image_similarities = most_similar_images(processed_image, image_features)
    similar_review_indices, review_similarities = most_similar_reviews(input_review_tfidf, tf_idf_scores)

    print("Similar Image Indices:", similar_image_indices)
    print("Image Similarities:", image_similarities)
    print("Similar Review Indices:", similar_review_indices)
    print("Review Similarities:", review_similarities)
else:
    print("The specified image URL and review were not found in the dataset.")

# Save the retrieval results
retrieval_results = {
    'similar_image_indices': similar_image_indices,
    'image_similarities': image_similarities,
    'similar_review_indices': similar_review_indices,
    'review_similarities': review_similarities,
}

results_path = common.os.path.join(common.output_dir, 'retrieval_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(retrieval_results, f)
