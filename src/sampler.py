import torch
from torch.utils.data import WeightedRandomSampler

def get_oversampler(labels):
    """
    Balances classes by giving higher probability to the minority class.
    """
    # Convert labels to a tensor
    labels_tensor = torch.as_tensor(labels)
    
    # 1. Count how many samples we have per class
    # torch.bincount counts occurrences of each integer in the tensor
    class_sample_count = torch.bincount(labels_tensor)
    
    # 2. Calculate the weight for a single sample of each class
    # Formula: $weight = \frac{1}{\text{count}}$
    weights = 1. / class_sample_count.float()
    
    # 3. Map those weights to every sample in our dataset
    # If label is 0, it gets weights[0]. If label is 1, it gets weights[1].
    samples_weight = weights[labels_tensor]
    
    # 4. Create the sampler
    return WeightedRandomSampler(samples_weight, len(samples_weight))