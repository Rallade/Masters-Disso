try:
    import cupy as np
except ImportError:
    import numpy as np


def pool(sentence_matrix, pooling_mode):
    if pooling_mode == "mean_pooling":
        return reduce_mean(sentence_matrix)
    elif pooling_mode == "max_pooling_single":
        return reduce_max_single(sentence_matrix)
    elif pooling_mode == "max_pooling_total":
        return reduce_max_total(sentence_matrix)

def reduce_mean(sentence_matrix):
    if sentence_matrix is None:
        return None
    sentence_matrix = np.array(sentence_matrix)
    mean = np.mean(sentence_matrix, axis=0, dtype=np.float64) #change float type if slow
    return mean

def reduce_max_single(sentence_matrix):
    if sentence_matrix is None:
        return None
    sentence_matrix = np.array(sentence_matrix)
    norms = np.linalg.norm(sentence_matrix, ord=2, axis= 1) #remove start and end tokens
    index = np.argmax(norms)
    return sentence_matrix[index] #account for token removal

def reduce_max_total(sentence_matrix):
    if sentence_matrix is None:
        return None
    sentence_matrix = np.array(sentence_matrix)
    return np.amax(sentence_matrix, axis=0)