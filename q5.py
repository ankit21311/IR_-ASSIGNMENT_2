import common
import pickle
from scipy.spatial.distance import cdist


df = common.pd.read_csv(common.dataset_path)

results_path = common.os.path.join(common.output_dir, 'retrieval_results.pkl')

with open(results_path, 'rb') as f:
    retrieval_results = pickle.load(f)

similar_image_indices = retrieval_results['similar_image_indices']
image_similarities = retrieval_results['image_similarities']
similar_review_indices = retrieval_results['similar_review_indices']
review_similarities = retrieval_results['review_similarities']

def get_data_by_indices(df, image_indices, review_indices):
    # Extract image URLs and reviews by indices
    image_urls = df.loc[image_indices, 'Image'].tolist()
    reviews = df.loc[review_indices, 'Review Text'].tolist()
    return image_urls, reviews


image_urls, reviews = get_data_by_indices(df, similar_image_indices, similar_review_indices)

def calculate_composite_scores(image_similarities, review_similarities):
    composite_scores = []
    # Assuming image_similarities and review_similarities are aligned and of equal length
    index = 0
    composite_scores = []
    while index < len(image_similarities):
        comp_score = (image_similarities[index] + review_similarities[index]) / 2
        composite_scores.append((index, index, comp_score))  # Use index for both indices, or adjust as needed
        index += 1
    composite_scores.sort(key=lambda x: x[2], reverse=True)
    return composite_scores



## Calculate composite scores using only similarity scores
composite_scores = calculate_composite_scores(image_similarities, review_similarities)

## Display the combined retrieval results
print("USING IMAGE RETRIEVAL")
for a, (img_idx, rev_idx, comp_score) in enumerate(composite_scores, start=1):
     # Assuming each index points to relevant data in placeholder lists
     print(f"{a}) Image URL: {image_urls[img_idx]}")  # Example: Single URL or a list if applicable
     print(f"Review: {reviews[rev_idx]}")
     print(f"Cosine similarity of images - {image_similarities[img_idx]:.4f}")
     print(f"Cosine similarity of text - {review_similarities[rev_idx]:.4f}\n")

 # Assuming image_similarities and review_similarities are lists of scores from which composite scores were derived
composite_image_score = sum(image_similarities) / len(image_similarities)
composite_text_score = sum(review_similarities) / len(review_similarities)
final_composite_score = (composite_image_score + composite_text_score) / 2

print("Composite similarity scores of images:", f"{composite_image_score:.4f}")
print("Composite similarity scores of text:", f"{composite_text_score:.4f}")
print("Final composite similarity score:", f"{final_composite_score:.4f}\n")


print("USING TEXT RETRIEVAL")
for a, (img_idx, rev_idx, comp_score) in enumerate(composite_scores, start=1):
     # Assuming 'image_urls[img_idx]' fetches URLs of similar images based on text query
     # and 'reviews[rev_idx]' fetches the corresponding review
     print(f"{a}) Image URL: {image_urls[img_idx]}")  # List of Image URLs could be just this one for simplicity
     print(f"Review: {reviews[rev_idx]}")  # Extracted Review
     print(f"Cosine similarity of images - {image_similarities[img_idx]:.4f}")
     print(f"Cosine similarity of text - {review_similarities[rev_idx]:.4f}\n")

# # To compute final composite scores across all images and reviews:
final_composite_image_score = sum(image_similarities) / len(image_similarities)
final_composite_text_score = sum(review_similarities) / len(review_similarities)
final_composite_score = (final_composite_image_score + final_composite_text_score) / 2

print("Composite similarity scores of images:", final_composite_image_score)
print("Composite similarity scores of text:", final_composite_text_score)
print("Final composite similarity score:", final_composite_score)