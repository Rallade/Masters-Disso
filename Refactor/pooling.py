try:
    import cupy as np
except ImportError:
    import numpy as np
import nltk


def pool(sentence_matrix, pooling_mode, tokens=None):
    if pooling_mode == "mean_pooling":
        return reduce_mean(sentence_matrix)
    elif pooling_mode == "max_pooling_single":
        return reduce_max_single(sentence_matrix)
    elif pooling_mode == "max_pooling_total":
        return reduce_max_total(sentence_matrix)
    elif pooling_mode == "mean_pooling_pos_filtered":
        return reduce_mean_pos_filtered(sentence_matrix, tokens)
    elif pooling_mode == "max_pooling_pos_filtered_single":
        return reduce_max_single_pos_filtered(sentence_matrix, tokens)
    elif pooling_mode == "max_pooling_pos_filtered_total":
        return reduce_max_total_pos_filtered(sentence_matrix, tokens)
    else:
        print("Pooling mode not supported")
        raise ValueError

def reduce_max_total_pos_filtered(sentence_matrix, tokens):
    sent, x = pos_filter(sentence_matrix, tokens)
    return reduce_max_total(sent)

def reduce_max_single_pos_filtered(sentence_matrix, tokens):
    sent, x= pos_filter(sentence_matrix, tokens)
    return reduce_max_single(sent)

def reduce_mean_pos_filtered(sentence_matrix, tokens):
    sent, x = pos_filter(sentence_matrix, tokens)
    return reduce_mean(sent)

def pos_filter(sentence_matrix, tokens):
    pros_tokens, pros_embeddings = remake_tokens(tokens, sentence_matrix)
    pros_embeddings = simplify_nested_embeddings(pros_embeddings)
    pros_tokens = ["." if token == "[SEP]" else token for token in pros_tokens]
    pos_tags = nltk.pos_tag(pros_tokens[1:-1])
    new_embeddings = []
    new_tokens = []
    for i, pos in enumerate(pos_tags):
        if (pos[1] == 'JJ' or 'VB' in pos[1] or pos[1] == 'NN') and  ("|" not in pos[0]):
            new_embeddings.append(pros_embeddings[i+1]) #account for [CLS] token
            new_tokens.append(pos[0])
    return new_embeddings, new_tokens

def simplify_nested_embeddings(embeddings):
    new = []
    for embedding in embeddings:
        if len(embedding) > 1:
            new.append(reduce_mean(embedding))
        else:
            new.append(embedding[0])
    return new

def remake_tokens(tokens, embeddings):
    new_tokens = []
    new_embeddings = []
    for i, token in enumerate(tokens):
        if "##" not in token:
            new_tokens.append(token)
            new_embeddings.append([embeddings[i]])
        else:
            new_tokens[-1] += token[2:]
            new_embeddings[-1].append(embeddings[i])
    
    return new_tokens, new_embeddings

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