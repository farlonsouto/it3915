import numpy as np


def apply_mask(wandb_config, aggregated, masking_portion, masking_token):
    """
    Replace a given percentage of the aggregated array with a specific masking value.
    The masked positions are the ones to be considered for loss computation in MLM.

    Parameters:
    - aggregated (np.ndarray): Input array, aggregated readings
    - masking_portion (float): Percentage of elements to replace (0 <= p <= 1).
    - masking_token (scalar): Value to replace with.

    Returns:
    - np.ndarray: Modified aggregated array with replaced values.
    - np.ndarray: Mask array indicating masked positions.
    """
    if not wandb_config.mlm_mask or wandb_config.model not in ['bert', 'transformer']:
        # For either other models or no MLM, return a mask of ones, meaning to compute loss based on all data
        mask = np.ones_like(aggregated)
        assert aggregated.shape == mask.shape, "Shape mismatch between aggregated and mask"
        return aggregated, mask

    # Flatten the aggregated array for easier manipulation
    flat_aggregated = aggregated.ravel()
    num_elements = flat_aggregated.size
    num_to_replace = int(num_elements * masking_portion)

    # Handle edge cases
    if num_to_replace <= 0:
        mask = np.ones_like(aggregated)
        assert aggregated.shape == mask.shape, "Shape mismatch between aggregated and mask"
        return aggregated, mask

    # Generate random indices for replacement
    replace_indices = np.random.choice(num_elements, size=num_to_replace, replace=False)

    # Initialize a flat mask array and mark replacement indices
    flat_mask = np.zeros_like(flat_aggregated)
    flat_mask[replace_indices] = 1.0

    # Apply the masking token at the chosen indices
    flat_aggregated[replace_indices] = masking_token

    # Reshape back to the original aggregated shape
    modified_aggregated = flat_aggregated.reshape(aggregated.shape)
    mask = flat_mask.reshape(aggregated.shape)

    # Final assertion to ensure shapes match
    assert modified_aggregated.shape == mask.shape, "Shape mismatch between modified_aggregated and mask"

    return modified_aggregated, mask
