import torch
from torch.utils.data import Dataset
import tensorflow as tf
import numpy as np

class ClickbaitDataset(Dataset):
    def __init__(self, examples, max_seq_length):
        """PyTorch Dataset class for clickbait text classification.

        Args:
            examples (list): A list of tuples where each tuple contains a text example and a label mask.
            max_seq_length (int): The maximum sequence length allowed for the encoded text.

        Methods:
            __len__(): Returns the length of the dataset.
            __getitem__(idx): Returns the encoded text, label, sequence length, and label mask for the given index.

        """
        self.examples = examples
        self.max_seq_length = max_seq_length
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens = 50000,
                                               output_mode = 'int',)

        self.vectorizer.adapt([e[0] for e in examples])

    def __len__(self):
        """Returns the length of the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns the encoded text, label, sequence length, and label mask for the given index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            tuple: A tuple containing the encoded text, label, sequence length, and label mask for the given index.
        """
        text,label = self.examples[idx]
        encoded_sent = self.vectorizer([text])
        encoded_sent = list(encoded_sent.numpy()[0][:self.max_seq_length])
        seq_len = len(encoded_sent)
        return encoded_sent, label, seq_len
    
    
def pad(batch):
    """Pads the input sequences in a batch to a fixed length.

    Args:
        batch (list): A list of tuples where each tuple contains an encoded text, a label, a sequence length.
    Returns:
        tuple: A tuple containing the padded encoded text, labels and sequence lengths.
    """
    f = lambda x: [sample[x] for sample in batch]
    token_ids = f(0)
    labels = f(1)
    seqlens = f(2)
    
    maxlen = np.array(seqlens).max()
    attention_mask = [[1.0] * len(ids) + [0.0] * (maxlen - len(ids)) for ids in token_ids]
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] 
    x = f(0, maxlen)
    f = torch.LongTensor
    
    return f(x).T, f(labels), f(seqlens)