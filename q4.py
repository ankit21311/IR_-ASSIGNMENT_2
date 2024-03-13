import common
import pickle
from scipy.spatial.distance import cdist


df = common.pd.read_csv(common.dataset_path)

# Assuming output_dir is already defined
results_path = common.os.path.join(common.output_dir, 'retrieval_results.pkl')

with open(results_path, 'rb') as f:
    retrieval_results = pickle.load(f)

# Extract individual components from the loaded retrieval results
similar_image_indices = retrieval_results['similar_image_indices']
image_similarities = retrieval_results['image_similarities']
similar_review_indices = retrieval_results['similar_review_indices']
review_similarities = retrieval_results['review_similarities']


def composite_scores(image_similarities, review_similarities):
    composite_scores = []
    for image_similarity, review_similarity in zip(image_similarities, review_similarities):
        # Calculate the average similarity score for each pair
        composite_score = (image_similarity + review_similarity) / 2
        composite_scores.append(composite_score)
    return composite_scores

composite_scores = composite_scores(image_similarities, review_similarities)

# Create a list of tuples (composite_score, image_index, review_index) and sort it
ranked_pairs = sorted(zip(composite_scores, similar_image_indices, similar_review_indices), reverse=True, key=lambda x: x[0])

# Display the ranked results
print("Ranked Combined Retrieval Results:")
for rank, (comp_score, img_idx, rev_idx) in enumerate(ranked_pairs, start=1):
    print(f"Rank: {rank}, Image Index: {img_idx}, Review Index: {rev_idx}, Composite Score: {comp_score:.4f}")


ranked_results_path = common.os.path.join(common.output_dir, 'ranked_combined_retrieval_results.pkl')
with open(ranked_results_path, 'wb') as f:
    pickle.dump(ranked_pairs, f)

print(f"Ranked combined retrieval results saved to: {ranked_results_path} \n")


def get_data_by_indices(df, image_indices, review_indices):
    # Extract image URLs and reviews by indices
    image_urls = df.loc[image_indices, 'Image'].tolist()
    reviews = df.loc[review_indices, 'Review Text'].tolist()
    return image_urls, reviews


image_urls = get_data_by_indices(df, similar_image_indices, similar_review_indices)
reviews = get_data_by_indices(df, similar_image_indices, similar_review_indices)

