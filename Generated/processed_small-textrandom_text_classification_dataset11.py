import torch
import random
import string
from typing import List, Optional

class PytorchTextClassificationDataset:
    def __init__(self, data, vocab, multi_label, target_labels):
        self.data = data
        self.vocab = vocab
        self.multi_label = multi_label
        self.target_labels = target_labels

def random_text_classification_dataset(num_samples: int, 
                                       max_length: int, 
                                       num_classes: int, 
                                       multi_label: bool, 
                                       vocab_size: int, 
                                       device: str, 
                                       target_labels: Optional[List[str]] = None, 
                                       dtype: torch.dtype = torch.float32) -> PytorchTextClassificationDataset:
    # Generate a random vocabulary
    vocab = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(vocab_size)]
    
    # Ensure all labels are included in the dataset
    if target_labels is None:
        target_labels = [f'class_{i}' for i in range(num_classes)]
    else:
        assert len(target_labels) == num_classes, "Length of target_labels must match num_classes"
    
    # Generate random text data
    data = []
    for _ in range(num_samples):
        text_length = random.randint(1, max_length)
        text = random.choices(vocab, k=text_length)
        
        if multi_label:
            labels = torch.zeros(num_classes, dtype=dtype)
            num_labels = random.randint(1, num_classes)
            label_indices = random.sample(range(num_classes), num_labels)
            labels[label_indices] = 1
        else:
            label_index = random.randint(0, num_classes - 1)
            labels = torch.tensor(label_index, dtype=dtype)
        
        data.append((text, labels))
    
    # Convert data to tensors and move to the specified device
    text_data = [torch.tensor([vocab.index(word) for word in text], dtype=torch.long) for text, _ in data]
    label_data = [labels for _, labels in data]
    
    text_data = torch.nn.utils.rnn.pad_sequence(text_data, batch_first=True).to(device)
    label_data = torch.stack(label_data).to(device)
    
    dataset = PytorchTextClassificationDataset(data=(text_data, label_data), 
                                               vocab=vocab, 
                                               multi_label=multi_label, 
                                               target_labels=target_labels)
    return dataset

