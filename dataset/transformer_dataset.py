import torch
from transformers import AutoTokenizer, BertTokenizer, XLMRobertaTokenizer
import pandas as pd
from torch.utils.data import Dataset, \
                            DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning import LightningDataModule
import transformers
from utils.functions import preproc_pipeline
import os
import numpy as np
import math
import random


class ClickbaitDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_length):
        """PyTorch Dataset class for clickbait text classification.

        Args:
            examples (list): A list of tuples where each tuple contains a text example and a label mask.
            tokenizer (transformers.PreTrainedTokenizer): A tokenizer object used to encode the text.
            max_seq_length (int): The maximum sequence length allowed for the encoded text.

        Methods:
            __len__(): Returns the length of the dataset.
            __getitem__(idx): Returns the encoded text, label, sequence length, and label mask for the given index.

        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns the encoded text, label, sequence length, and label mask for the given index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            tuple: A tuple containing the encoded text, label, sequence length, and label mask for the given index.
        """
        text, label_mask = self.examples[idx]
        encoded_sent = self.tokenizer.encode(
            text[0], max_length=self.max_seq_length, truncation=True
        )
        label = text[1]
        seq_len = len(encoded_sent)
        return encoded_sent, label, seq_len, label_mask

        
def pad(batch):
    """Pads the input sequences in a batch to a fixed length.

    Args:
        batch (list): A list of tuples where each tuple contains an encoded text, a label, a sequence length, and a label mask.

    Returns:
        tuple: A tuple containing the padded encoded text, attention mask, labels, and label mask.
    """
    
    f = lambda x: [sample[x] for sample in batch]
    token_ids = f(0)
    labels = f(1)
    seqlens = f(2)
    label_mask = f(3)
    
    maxlen = np.array(seqlens).max()
    attention_mask = [[1.0] * len(ids) + [0.0] * (maxlen - len(ids)) for ids in token_ids]
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] 
    x = f(0, maxlen)
    f = torch.LongTensor
    
    return f(x), torch.tensor(attention_mask,dtype=torch.float) \
            , f(labels), torch.BoolTensor(label_mask)

    

class ClickbaitDataModule(LightningDataModule):
  def __init__(
      self,
      batch_size: int = 64,
      num_workers: int = 4,
      label_list=[0,1],
      model_name_or_path: str = 'csebuetnlp/banglabert',
      model_type: str = 'gan',
      max_seq_length:int = 256,
      data_folder:str = 'dataset/data',
      data_column:str = 'title',
      label_column:str = 'label',
      tokenizer_class:str = 'AutoTokenizer',
      ratio = 100
  ):  
      """
        Initializes the ClickbaitDataModule for data loading and processing.

        Args:
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of CPU workers for data loading.
            label_list (list): List of class labels.
            model_name_or_path (str): Pre-trained model name or path.
            model_type (str): Type of the model, 'gan' or 'transformer'.
            max_seq_length (int): Maximum sequence length.
            data_folder (str): Path to the dataset folder.
            data_column (str): Name of the column containing text data.
            label_column (str): Name of the column containing labels.
            tokenizer_class (str): Tokenizer class name.
            ratio (int): Ratio of unlabeled examples for GAN training.
        """
      super().__init__()
      self.batch_size = batch_size
      self.num_workers = num_workers  
      self.label_list = label_list
      self.num_classes = len(self.label_list)
      self.model_name = model_name_or_path
      self.max_seq_length = max_seq_length
      self.tokenizer = eval(f"{tokenizer_class}.from_pretrained('{self.model_name}')")
      self.model_type = model_type
      
      #load data based on model type
      for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):

            if 'train' in file_name:
                traindf = pd.read_csv(os.path.join(data_folder,file_name))
                traindf = traindf.sample(frac=1,random_state=1234).reset_index(drop=True)
                cleaned_titles = [preproc_pipeline(title) for title in traindf[data_column].values]
                self.train_examples = list(zip(cleaned_titles,traindf[label_column].values))
                
                #  The labeled train dataset is assigned with a mask set to True (1)
                self.train_label_masks = np.ones(len(self.train_examples), dtype=bool)

            elif 'test' in file_name:
                testdf = pd.read_csv(os.path.join(data_folder,file_name))
                cleaned_titles = [preproc_pipeline(title) for title in testdf[data_column].values]
                self.test_examples = list(zip(cleaned_titles,testdf[label_column].values))
                
                #  The labeled test dataset is assigned with a mask set to True (1)
                self.test_label_masks = np.ones(len(self.test_examples), dtype=bool)

            elif self.model_type == 'gan' and 'unlabel' in file_name:
                unlabelled_df = pd.read_csv(os.path.join(data_folder,file_name))
                unlabelled_df = unlabelled_df.sample(frac=1,random_state=1234).reset_index(drop=True)
                cleaned_titles = [preproc_pipeline(title) for title in unlabelled_df[data_column].values]
                self.unlabelled_examples = list(zip(cleaned_titles,[2]*unlabelled_df.shape[0]))
                ratio = ratio // 100
                self.unlabelled_examples = self.unlabelled_examples[:len(self.unlabelled_examples)*ratio]
                

            elif 'dev' in file_name or 'val' in file_name:
                devdf = pd.read_csv(os.path.join(data_folder,file_name))
                cleaned_titles = [preproc_pipeline(title) for title in devdf[data_column].values]
                self.dev_examples = list(zip(cleaned_titles,devdf[label_column].values))
                
                #  The labeled dev dataset is assigned with a mask set to True (1)
                self.dev_label_masks = np.ones(len(self.dev_examples), dtype=bool)   
      
      self.label_map = {label: i for i, label in enumerate(self.label_list)}
    
      # If unlabelled examples are available and semi supervised gan method is used for training
      if self.model_type == 'gan' and self.unlabelled_examples:
        self.train_examples = self.train_examples + self.unlabelled_examples
        
        #The unlabeled (train) dataset is assigned with a mask set to False (0)
        self.tmp_masks = np.zeros(len(self.unlabelled_examples), dtype=bool)
        self.train_label_masks = np.concatenate([self.train_label_masks,self.tmp_masks])
        train_ex_label_masks = [(ex,mask) for ex,mask in zip(self.train_examples,self.train_label_masks)]
        random.shuffle(train_ex_label_masks)
        self.train_examples, self.train_label_masks = zip(*train_ex_label_masks)
        self.train_examples, self.train_label_masks = list(self.train_examples), list(self.train_label_masks)
      
  def generate_data_loader(self,input_examples, label_masks, do_shuffle = False, balance_label_examples = True):
    """
    Generate a DataLoader given the input examples, eventually masked if they are 
    to be considered NOT labeled.

    Args:
        input_examples (list): List of input examples.
        label_masks (list): List of label masks.
        do_shuffle (bool): Whether to shuffle the data.
        balance_label_examples (bool): Whether to balance the ration between labeled and unlabelled examples.

    Returns:
        DataLoader: DataLoader instance for the input examples.
    """
    examples = []

    # Count the percentage of labeled examples  
    num_labeled_examples = 0
    for label_mask in label_masks:
      if label_mask==1: 
        num_labeled_examples += 1
    label_mask_rate = num_labeled_examples/len(input_examples)
    print(f"label_mask_rate:{label_mask_rate}")
    
    # if required it applies the balance
    for index, ex in enumerate(input_examples): 
      if label_mask_rate == 1 or not balance_label_examples:
        examples.append((ex, label_masks[index]))
      else:
        # IT SIMULATE A LABELED EXAMPLE
        if label_masks[index]==1:
          balance = int(1/label_mask_rate)
          balance = int(math.log(balance,2))
          if balance < 1:
            balance = 1
          for b in range(0, int(balance)):
            examples.append((ex, label_masks[index]))
        else:
          examples.append((ex, label_masks[index]))
    
    labelled, unlabelled = 0, 0
    for ex in examples:
        if ex[1]==1:
            labelled+=1
        else:
            unlabelled+=1
    print(f"labelled examples : {labelled}, unlabelled examples : {unlabelled}")
        

    dataset = ClickbaitDataset(examples,self.tokenizer,self.max_seq_length)
    # return dataset
    if do_shuffle:
      sampler = RandomSampler
    else:
      sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
                dataset,  # The training samples.
                sampler = sampler(dataset), 
                batch_size = self.batch_size,
                pin_memory = True,
                num_workers=self.num_workers,
                collate_fn = pad) # Trains with this batch size.

  def train_dataloader(self):
      if self.model_type == 'gan':
          return self.generate_data_loader(self.train_examples, self.train_label_masks,
                                  do_shuffle = True, balance_label_examples = True)
      else:
          return self.generate_data_loader(self.train_examples, self.train_label_masks,
                                  do_shuffle = True, balance_label_examples = False)

  def val_dataloader(self):
      return self.generate_data_loader(self.dev_examples, self.dev_label_masks,
                                  do_shuffle = False, balance_label_examples = False)

  def test_dataloader(self):
      return self.generate_data_loader(self.test_examples, self.test_label_masks, 
                                  do_shuffle = False, balance_label_examples = False)
   