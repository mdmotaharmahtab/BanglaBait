import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, get_constant_schedule_with_warmup


def build_transformer_optimizer(model, config):
    """
    Build an optimizer and learning rate scheduler for a Transformer-based model.

    This function constructs an optimizer and a learning rate scheduler for a Transformer-based model.
    
    Args:
        model (nn.Module): The Transformer-based model to optimize.
        config (object): Configuration object containing optimization parameters.

    Returns:
        Tuple[torch.optim.Optimizer, transformers.optimization.get_scheduler]: 
        A tuple containing the optimizer and the learning rate scheduler.

    """
    transformer_vars = [i for i in model.parameters()]
    optimizer = torch.optim.AdamW(transformer_vars, lr=config.learning_rate)
    num_train_steps = int(config.num_train_examples / config.batch_size * config.epochs)
    num_warmup_steps = int(num_train_steps * config.warmup_proportion)

    scheduler = get_constant_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = num_warmup_steps)
      
    return optimizer,scheduler

def transformer_save_model(save_path:str,model,optimizers):
    """
    Save a Transformer-based model and its optimizer to a file.

    This function saves a Transformer-based model and its associated optimizer to a specified file path.

    Args:
        save_path (str): The file path where the model and optimizer will be saved.
        model (nn.Module): The Transformer-based model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
    """

    torch.save({
        'model': model.state_dict(),
        'optimizer_state_dict': optimizers[0].state_dict(),
        }, save_path)

def transformer_load_model(model:nn.Module, load_path: str, device: str):
    """
    Load a pre-trained Transformer-based model from a file.

    This function loads a pre-trained Transformer-based model from a specified file path and places it on the
    specified device (CPU or GPU).

    Args:
        model (nn.Module): An instance of the Transformer-based model class to load the parameters into.
        load_path (str): The file path from which to load the pre-trained model.
        device (str): The device where the loaded model should be placed ('cpu' or 'cuda:0' for GPU).

    Returns:
        nn.Module: The loaded Transformer-based model.
    """
    model.load_state_dict(torch.load(load_path,map_location=device)['model'])
    return model

class TransformerForSequenceClassification(nn.Module):
    """
    Simplified version of the same class by HuggingFace.
    See transformers/modeling_distilbert.py in the transformers repository.
    """

    def __init__(
        self, model_name_or_path: str, num_classes: int = 2, dropout: float = 0.5,
        mean_pool: bool=True
    ):
        """
        Initialize a custom Transformer-based classification model.

        Args:
            model_name_or_path (str): HuggingFace model name or path to
                a pretrained model file.
            num_classes (int): The number of class labels in the classification task.
            dropout (float): Dropout probability for model layers.
            mean_pool (bool): If True, use mean pooling over sequence dimensions; otherwise, use [CLS] token.

        Note:
            The `dropout` argument sets dropout probabilities for various layers in the transformer model.

        Attributes:
            model (AutoModel): The pretrained Transformer model.
            classifier (nn.Linear): The classifier layer for classification tasks.
            dropout (nn.Dropout): Dropout layer.
            mean_pool (bool): Indicates whether mean pooling is used.
            num_labels (int): The number of class labels.
        """
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_classes
        )
        for k in config.to_dict().keys():
          if 'dropout' in k and 'classifier' not in k:
            config.update({k:0.3})

        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(dropout)
        self.mean_pool=mean_pool
        self.num_labels = num_classes
        if torch.cuda.is_available():
          self.model.cuda()
          self.classifier.cuda()
          self.dropout.cuda()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        """
        Forward pass of the custom Transformer-based classification model.

        Args:
            input_ids (Tensor): Tokenized input IDs.
            attention_mask (Tensor): Attention mask for input IDs.
            token_type_ids (Tensor): Token type IDs.
            position_ids (Tensor): Position IDs.
            head_mask (Tensor): Head mask.
            inputs_embeds (Tensor): Embedded inputs.
            labels (Tensor): Ground-truth labels for the classification task.
            encoder_hidden_states: Hidden states from the encoder.
            encoder_attention_mask: Attention mask for encoder hidden states.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[Tensor]: A tuple containing model outputs.

        Note:
            This method performs a forward pass of the custom Transformer-based classification model.

        """

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_state = output[0]  # (bs, seq_len, dim)
        outputs = (hidden_state,)
        if not self.mean_pool:
          pooled_output = hidden_state[:, 0]  # (bs, dim)
        else:
          pooled_output = hidden_state.mean(axis=1)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        outputs = (logits,) + outputs
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states)
    


