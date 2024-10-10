import numpy as np


def max_inner_product_linear_search(vectors: np.ndarray, query: np.ndarray):
    max_value = float("-inf")
    for v in vectors:
        dot = np.dot(v, query)
        if dot > max_value:
            ret_vector: np.ndarray = v
            max_value = dot
    return ret_vector.copy(), max_value
