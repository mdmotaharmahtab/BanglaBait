"""
  Attention and CNN class is a modification from following sources -  
- https://github.com/Rowan1697/FakeNews/blob/master/Models/NN/LSTM_Attn.py (for Attention Class)
- https://github.com/Rowan1697/FakeNews/blob/master/Models/NN/CNN.py (for CNN class)
- 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from torch.autograd import Variable

def nn_save_model(save_path:str,model,optimizer):
    """
    Save the PyTorch model and optimizer to a specified file.

    Args:
        save_path (str): The path to save the model and optimizer.
        model: The PyTorch model to be saved.
        optimizer: The PyTorch optimizer associated with the model.

    This function saves the model's state dictionary and the optimizer's state dictionary to
    the specified file path, allowing you to later load and resume training or make predictions
    with the model.

    """
    torch.save({
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

class CNN(nn.Module):
    def __init__(self, embedding_tensors: Tensor, 
              config: object, 
              class_weights: List[float] = None):

        """
        Convolutional Neural Network (CNN) for text classification.

        Args:
            embedding_tensors (Tensor): Pre-trained word embeddings.
            config (object): Configuration object for model parameters.
            class_weights (List[float], optional): Weights for class balancing.

        Attributes:
            ---------
            batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
            output_size : 2 = (pos, neg)
            in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
            out_channels : Number of output channels after convolution operation performed on the input matrix
            kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
            keep_probab : Probability of retaining an activation node during dropout operation
            vocab_size : Size of the vocabulary containing unique words
            embedding_length : Embedding dimension of GloVe word embeddings
            weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
            --------
            
        This class defines a CNN architecture for text classification. It takes pre-trained word embeddings,
        configuration parameters, and optional class weights. The forward method processes input text data
        and produces class logits.
        """

        super(CNN, self).__init__()
        self.batch_size = config.batch_size
        self.output_size = config.output_dim
        self.in_channels = config.model['in_channels']
        self.out_channels = config.model['out_channels']
        self.kernel_heights = config.model['kernel_heights']
        self.stride = config.model['stride']
        self.padding = config.model['padding']
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_tensors, 
                                                        padding_idx = self.padding,
                                                        freeze = config.update_embedding)
        self.embedding_length = self.word_embeddings.weight.shape[-1]
        self.vocab_size = self.word_embeddings.weight.shape[0]

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[0], self.embedding_length), self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[1], self.embedding_length), self.stride, self.padding)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[2], self.embedding_length), self.stride, self.padding)
        self.conv4 = nn.Conv2d(self.in_channels, self.out_channels, (self.kernel_heights[3], self.embedding_length), self.stride, self.padding)
        self.dropout = nn.Dropout(config.model['keep_probab'])
        self.label = nn.Linear(len(self.kernel_heights)*self.out_channels, self.output_size)

        if config.use_class_weight:
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_func = nn.CrossEntropyLoss()


    def conv_block(self, input, conv_layer):
        """
        Apply a convolutional block operation.

        Args:
            input (Tensor): Input tensor.
            conv_layer: Convolutional layer to apply.

        Returns:
            Tensor: Output tensor after applying convolution and pooling.

        This method applies a convolutional block operation to the input tensor using the provided
        convolutional layer. It performs convolution, ReLU activation, and max-pooling operations.

        """

        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)

        return max_out

    def forward(self, input_sentences, text_lengths):
        """
        Forward pass of the Convolutional Neural Network (CNN).
        This method performs a forward pass of the CNN model. It takes input sentences,
        applies convolutional layers with max-pooling, and produces logits as model predictions.

        Args:
            - input_sentences: A tensor containing input sentences with shape (sequence_length, batch_size, embedding_dimension).
            - text_lengths: A tensor containing the lengths of input sentences for padding.

        Returns:
            - logits: The model's predictions as a tensor.
    
        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        input = input.unsqueeze(1)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)
        max_out4 = self.conv_block(input, self.conv4)
        all_out = torch.cat((max_out1, max_out2, max_out3, max_out4), 1)
        fc_in = self.dropout(all_out)
        logits = self.label(fc_in)

        return logits


class ClickbaitLSTM(nn.Module):
    def __init__(self, embedding_tensors:Tensor, 
                 config: object, 
                 class_weights: List[float] = None):
        
        """
        LSTM-based model with attention mechanism for text classification.

        Args:
            emb_tensors (Tensor): Pre-trained word embeddings.
            config (object): Configuration object for model parameters.
            class_weights (List[float], optional): Weights for class balancing.

        This class defines an LSTM-based model with an attention mechanism for text classification.
        It takes pre-trained word embeddings, configuration parameters, and optional class weights.
        The forward method processes input text data and produces class logits.
        """

        super(ClickbaitLSTM,self).__init__()
        self.dropout_rate = config.model['out_dropout_rate']
        self.embedding = nn.Embedding.from_pretrained(embedding_tensors, 
                                                      padding_idx = config.model['pad_idx'],
                                                      freeze = config.update_embedding)
        
        embedding_dim = self.embedding.weight.shape[-1]
        self.lstm_hidden_dim = config.model['lstm_hidden_dim']
        self.rnn = nn.LSTM(embedding_dim, 
                           self.lstm_hidden_dim, 
                           num_layers=config.model['lstm_layers'], 
                           bidirectional=config.model['bidirectional'], 
                           dropout = self.dropout_rate
                          )
        
        self.fc1 = nn.Linear(self.lstm_hidden_dim * 2, self.lstm_hidden_dim)
        self.fc2 = nn.Linear(self.lstm_hidden_dim,self.lstm_hidden_dim//4)
        self.fc3 = nn.Linear(self.lstm_hidden_dim//4,config.output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        if config.use_class_weight:
          self.loss_func = nn.CrossEntropyLoss(weight=class_weights)
        else:
          self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, text, text_lengths):
        """
        Forward pass of the ClickbaitLSTM model.

        Args:
            text (Tensor): Input tensor containing text sequences.
            text_lengths (Tensor): Lengths of input text sequences.

        Returns:
            Tensor: Logits representing the model's predictions.

        This method performs a forward pass of the ClickbaitLSTM model. It takes input text sequences,
        applies an embedding layer, an LSTM layer, and fully connected layers to produce logits as model predictions.

        """
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, 
                                                            text_lengths.to('cpu'),
                                                            enforce_sorted=False)
        packed_output,(hidden,cell) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
        return self.fc3(self.dropout(F.relu(self.fc2(self.dropout(F.relu(self.fc1(hidden)))))))

class Attention(nn.Module):
    def __init__(self, dimension):
        """
        Attention Module for Sequence Data.

        This module computes attention weights and applies them to the input hidden states
        to obtain a weighted representation.

        Args:
            dimension (int): The dimension of the input hidden states.
        """
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, h):
        """
        Forward pass of the Attention module.

        Args:
            h (Tensor): Input tensor representing the hidden states.

        Returns:
            output (Tensor): Output tensor after applying attention mechanism.

        This method performs a forward pass of the Attention module. It takes input hidden states,
        applies the attention mechanism, and produces an output tensor.

        """
        x = self.u(h)
        x = self.tanh(x)
        x = self.softmax(x)
        output = x * h
        output = torch.sum(output, dim=1)
        return output


class ClickbaitLSTMAttention(nn.Module):
    def __init__(self,  
                emb_tensors: Tensor,
                config: object,
                class_weights: List[float] = None,
                 ):
      
      """
        Clickbait LSTM Model with Attention Mechanism.

        This class defines a Clickbait LSTM model with an attention mechanism. It takes input text data and applies
        an LSTM network with attention to make predictions.

        Args:
            emb_tensors (Tensor): Pre-trained word embeddings.
            config (object): Configuration object containing model parameters.
            class_weights (List[float], optional): Class weights for loss calculation.

        Attributes:
            dropout_rate (float): Dropout rate for regularization.
            embedding (nn.Embedding): Embedding layer for word embeddings.
            lstm_hidden_dim (int): Dimension of the LSTM hidden states.
            rnn (nn.LSTM): LSTM layer for sequential modeling.
            attn_module (Attention): Attention module for attending to important parts of the input.
            fc3 (nn.Linear): Fully connected layer for output predictions.
            dropout (nn.Dropout): Dropout layer for regularization.
            loss_func (nn.CrossEntropyLoss): Loss function for training.

        """
      super(ClickbaitLSTMAttention, self).__init__()

      self.dropout_rate = config.model['out_dropout_rate']
      self.embedding = nn.Embedding.from_pretrained(emb_tensors,
                                                    padding_idx = config.model['pad_idx'],
                                                    freeze = config.update_embedding)
      
      embedding_dim = self.embedding.weight.shape[-1]
      self.lstm_hidden_dim = config.model['lstm_hidden_dim']
      self.rnn = nn.LSTM(embedding_dim, 
                          self.lstm_hidden_dim, 
                          num_layers=config.model['lstm_layers'], 
                          bidirectional=config.model['bidirectional'], 
                          dropout = self.dropout_rate
                        )
      
      self.attn_module = Attention(self.lstm_hidden_dim * 2)
      self.fc3 = nn.Linear(self.lstm_hidden_dim*2,config.output_dim)
      self.dropout = nn.Dropout(self.dropout_rate)
      if config.use_class_weight:
          self.loss_func = nn.CrossEntropyLoss(weight=class_weights)
      else:
          self.loss_func = nn.CrossEntropyLoss()

    def forward(self, text, text_lengths):
        """
        Forward pass of the ClickbaitLSTMAttention model.

        Args:
            text (Tensor): Input text data.
            text_lengths (Tensor): Lengths of input sequences.

        Returns:
            Tensor: Model predictions.

        This method performs a forward pass of the ClickbaitLSTMAttention model. It takes input text data,
        applies an LSTM network with attention, and produces model predictions.
        """
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, 
                                                            text_lengths.to('cpu'),
                                                            enforce_sorted=False)
        packed_output,(hidden,cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = output.transpose(0,1)
        attn_output = self.attn_module(output)
        logits = self.fc3(attn_output)
        return logits
