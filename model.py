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

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    if p >= 1.0:
        return logits.copy()

    probs = stable_softmax(logits)

    # Stable descending sort preserves original order for ties
    sorted_indices = np.argsort(
        -probs, axis=-1, kind="stable"
    )

    sorted_probs = np.take_along_axis(
        probs, sorted_indices, axis=-1
    )

    cumulative_probs = np.cumsum(sorted_probs, axis=-1)

    previous_cumulative = cumulative_probs - sorted_probs
    keep_sorted = previous_cumulative < p

    keep_mask = np.zeros_like(logits, dtype=bool)

    np.put_along_axis(
        keep_mask, sorted_indices, keep_sorted, axis=-1
    )

    result = logits.copy()
    result[~keep_mask] = -np.inf

    return result

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    return int(rng.choice(len(probs), p=probs))

