"""
  The Generator and Discriminator code derived from - 
  https://www.aclweb.org/anthology/2020.acl-main.191/
  https://github.com/crux82/ganbert

"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, get_constant_schedule_with_warmup


def build_gan_optimizer(model, config):
    """
    Build and configure separate optimizers and schedulers for the Generator and Discriminator.

    Args:
        model (GAN): The GAN model.
        config (object): Configuration parameters.

    Returns:
        list: List of optimizers for the Discriminator and Generator.
        list: List of learning rate schedulers for the Discriminator and Generator.
    """
    transformer_vars = [i for i in model.transformer.parameters()]
    d_vars = transformer_vars + [v for v in model.discriminator.parameters()]
    g_vars = [v for v in model.generator.parameters()]
    dis_optimizer = torch.optim.AdamW(d_vars, lr=config.learning_rate_discriminator)
    gen_optimizer = torch.optim.AdamW(g_vars, lr=config.learning_rate_generator) 
    num_train_steps = int(config.num_train_examples / config.batch_size * config.epochs)
    num_warmup_steps = int(num_train_steps * config.warmup_proportion)

    scheduler_d = get_constant_schedule_with_warmup(dis_optimizer, 
                                            num_warmup_steps = num_warmup_steps)
    scheduler_g = get_constant_schedule_with_warmup(gen_optimizer, 
                                            num_warmup_steps = num_warmup_steps)
      
    return [dis_optimizer,gen_optimizer],[scheduler_d,scheduler_g]

def gan_save_model(save_path,model,optimizers):
    """
    Save the GAN model and optimizer states to a file.

    Args:
        save_path (str): Path to save the model checkpoint.
        model (GAN): The GAN model.
        optimizers (list): List of optimizers.

    Returns:
        None
    """
    torch.save({
        'model': model.state_dict(),
        'dis_optimizer_state_dict': optimizers[0][0].state_dict(),
        'gen_optimizer_state_dict': optimizers[0][1].state_dict(),
        }, save_path)


class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        """
        Generator module for GAN.

        Args:
            noise_size (int): Size of the input noise vector.
            output_size (int): Size of the output representation.
            hidden_sizes (list): List of hidden layer sizes.
            dropout_rate (float): Dropout rate.

        Attributes:
            layers (nn.Sequential): Sequential layers of the Generator.

        Methods:
            forward(noise): Forward pass through the Generator.
        """
        
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.3):
        """
        Discriminator module for GAN.

        Args:
            input_size (int): Size of the input representation.
            hidden_sizes (list): List of hidden layer sizes.
            num_labels (int): Number of output labels.
            dropout_rate (float): Dropout rate.

        Attributes:
            input_dropout (nn.Dropout): Dropout layer for input.
            layers (nn.Sequential): Sequential layers of the Discriminator.
            logit (nn.Linear): Linear layer for logits.
            softmax (nn.Softmax): Softmax activation function.

        Methods:
            forward(input_rep): Forward pass through the Discriminator.
        """
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) 
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        """
        Forward pass through the Discriminator.

        Args:
            input_rep (torch.Tensor): The input representation to be evaluated by the Discriminator.

        Returns:
            torch.Tensor: The final hidden representation after passing through layers.
            torch.Tensor: Logits produced by the Discriminator.
            torch.Tensor: Probabilities computed from the logits.

        This method takes an input representation, applies dropout, passes it through the layers of
        the Discriminator, computes logits, and finally applies softmax to obtain probabilities.

        """
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs
    
class GAN(nn.Module):
  def __init__(
      self,
      model_name_or_path='csebuetnlp/banglabert',
      noise_size = 100,
      num_hidden_layers_g=2,
      num_hidden_layers_d=2,
      out_dropout_rate =  0.2,
      device='cuda',
      label_list = [0,1,2],
      **kwargs
  ):  
      """
        Generative Adversarial Network (GAN) model for text generation.

        Args:
            model_name_or_path (str): Pre-trained model name or path.
            noise_size (int): Size of the input noise vector.
            num_hidden_layers_g (int): Number of hidden layers in the Generator.
            num_hidden_layers_d (int): Number of hidden layers in the Discriminator.
            out_dropout_rate (float): Dropout rate for output.
            device (str): Device for model training (e.g., 'cuda' or 'cpu').
            label_list (list): List of output labels.

        Attributes:
            model_name (str): Name or path of the pre-trained model.
            config (AutoConfig): Configuration object for the pre-trained model.
            hidden_size (int): Size of the hidden layers.
            noise_size (int): Size of the input noise vector.
            device (str): Device for model training.
            out_dropout_rate (float): Dropout rate for output.
            hidden_levels_g (list): List of hidden layer sizes for Generator.
            hidden_levels_d (list): List of hidden layer sizes for Discriminator.
            label_list (list): List of output labels.
            generator (Generator): Generator module.
            discriminator (Discriminator): Discriminator module.
            transformer (AutoModel): Pre-trained transformer model.

        Methods:
            forward(b_input_ids, b_input_mask): Forward pass through the GAN model.
        """

      super().__init__()

      self.model_name = model_name_or_path
      self.config = AutoConfig.from_pretrained(self.model_name)
      self.hidden_size = int(self.config.hidden_size)
      self.noise_size=noise_size
      self.device = device
      self.out_dropout_rate=out_dropout_rate
      
      # Define the number and width of hidden layers
      self.hidden_levels_g = [self.hidden_size for i in range(0, num_hidden_layers_g)]
      self.hidden_levels_d = [self.hidden_size for i in range(0, num_hidden_layers_d)]
      self.label_list =  label_list
      
      #   Instantiate the Generator and Discriminator
      self.generator = Generator(noise_size=noise_size, output_size=self.hidden_size, hidden_sizes=self.hidden_levels_g)
      self.discriminator = Discriminator(input_size=self.hidden_size, hidden_sizes=self.hidden_levels_d,num_labels=len(self.label_list), dropout_rate=out_dropout_rate)
      
      # Put everything in the GPU if available
      self.transformer = AutoModel.from_pretrained(self.model_name)
      if torch.cuda.is_available():    
        self.generator.cuda()
        self.discriminator.cuda()
        self.transformer.cuda()
  
  def forward(self,b_input_ids,b_input_mask):
    """
        Forward pass through the GAN.

        Args:
            b_input_ids (torch.Tensor): The input IDs for the Transformer model.
            b_input_mask (torch.Tensor): The input mask for the Transformer model.

        Returns:
            torch.Tensor: The final hidden states produced by the Transformer.
            torch.Tensor: Features computed by the Discriminator.
            torch.Tensor: Logits produced by the Discriminator.
            torch.Tensor: Probabilities computed from the logits.

        This method performs a forward pass through the GAN, including encoding real data using
        the Transformer, generating fake data using the Generator, and evaluating both real
        and fake data using the Discriminator. It returns the hidden states from the Transformer,
        Discriminator features, logits, and probabilities.

    """
    # Encode real data in the Transformer
    real_batch_size = b_input_ids.shape[0]
    model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
    if len(model_outputs[-1].shape) == 2:
      hidden_states = model_outputs[-1]
    else:
      hidden_states = torch.mean(model_outputs[0],dim=1)
    
    noise = torch.zeros(real_batch_size, self.noise_size, device=self.device).uniform_(0, 1).to(self.device)
    gen_rep = self.generator(noise)
    disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
    features, logits, probs = self.discriminator(disciminator_input)
    return hidden_states,features,logits,probs