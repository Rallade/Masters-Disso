try:
    import cupy as np
except ImportError:
    import numpy as np


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
    return np.amax(sentence_matrix, axis=0)

def reduce_max_total(sentence_matrix):
    if sentence_matrix is None:
        return None
    sentence_matrix = np.array(sentence_matrix)
    return np.amax(sentence_matrix, axis=0)