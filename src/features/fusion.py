import numpy as np

def fuse_features(tfidf_vector, extra_metrics, image_vector):
    # tfidf_vector est une matrice sparse (1, n)
    dense_tfidf = tfidf_vector.toarray().flatten()
    return np.concatenate([dense_tfidf, extra_metrics.flatten(), image_vector])
