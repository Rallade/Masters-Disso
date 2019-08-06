import numpy as np



def reduce_max(sentence_matrix):
    if sentence_matrix is None:
        return None
    sentence_matrix = np.array(sentence_matrix)
    mean = np.mean(sentence_matrix, axis=0, dtype=np.float64) #change float type if slow
    return mean
