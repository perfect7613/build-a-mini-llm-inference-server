"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np

def stable_softmax(logits):
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits

    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    if temperature <= 0:
        return logits

    return logits / temperature

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    V = logits.shape[-1]

    if k >= V:
        return logits.copy()

    threshold = np.partition(logits, -k, axis=-1)[..., -k, None]

    result = logits.copy()
    result[logits < threshold] = -np.inf

    return result

