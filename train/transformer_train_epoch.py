import logging
import wandb
import torch
import time
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import os
import math

from utils.functions import format_time
from utils.metrics import compute_metrics
from sklearn.metrics import classification_report

logger = logging.getLogger()


def transformer_train_epoch(model, dataset, optimizers,config,epoch):
  """
    Train a transformer-based model for one epoch.

    Args:
        model (nn.Module): The transformer-based model to train.
        dataset: The dataset containing training and validation data.
        optimizers (tuple): A tuple of two optimizers (optimizer, scheduler).
        config: Configuration settings for training.
        epoch (int): The current training epoch number.

    Returns:
        dict: Metrics and statistics for the training epoch.

    This function trains a transformer-based model for one epoch using the specified dataset, optimizers, and configuration settings.
    It computes loss values, updates model parameters, and logs training progress.


  """

  training_step_outputs=[]
  # Measure how long the training epoch takes.
  t0 = time.time()
  print_each_n_step = 100
  optimizer = optimizers[0]
  scheduler= optimizers[1]
  train_dl = dataset.train_dataloader()
  test_dl = dataset.val_dataloader()
  n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
  # Put the model into training mode.
  model.train()
  
  # For each batch of training data...
  for step, batch in enumerate(train_dl):
    # Progress update every print_each_n_step batches.
    if step % print_each_n_step == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)    
        # Report progress.
        logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dl), elapsed))

    # Unpack this training batch from our dataloader. 
    b_input_ids = batch[0].to(config.device)
    b_input_mask = batch[1].to(config.device)
    b_labels = batch[2].to(config.device)
    real_batch_size=b_input_ids.shape[0]
    loss,logits,hidden_states = model(input_ids=b_input_ids,attention_mask=b_input_mask,labels=b_labels)
    preds = torch.argmax(logits,dim=1)
    #---------------------------------
    #  OPTIMIZATION
    #---------------------------------
    # Avoid gradient accumulation
    optimizer.zero_grad()

    # Calculate weigth updates
    # retain_graph=True is required since the underlying graph will be deleted after backward
    loss.backward() 
    
    #clip grad norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    # Apply modifications
    optimizer.step()
    scheduler.step()
    metrics = {'train/loss_step':loss}
    if step + 1 < n_steps_per_epoch:
      wandb.log(metrics)
    
    training_step_outputs.append(OrderedDict({'loss':loss.detach().cpu(),\
                                              'preds':preds.detach().cpu(),\
                                              'labels':b_labels.detach().cpu(),
                                              }))
    
  
  all_losses = np.mean([x['loss'].item() for x in training_step_outputs])
  all_preds = [y.item() for x in training_step_outputs for y in x['preds']]
  all_labels =  [y.item() for x in training_step_outputs for y in x['labels']]
  result = compute_metrics((all_preds,all_labels))

  epoch_metrics = {"train/epoch_accuracy": result['accuracy'],
                   "train/epoch_precision_nc": result['precision'][0],
                   "train/epoch_recall_nc":result['recall'][0],
                   "train/epoch_f1_nc": result['f1'][0],
                   "train/epoch_precision": result['precision'][1],
                   "train/epoch_recall": result['recall'][1],
                   "train/epoch_f1": result['f1'][1],
                   'train/epoch_loss':all_losses,
                   'epoch':epoch
                   }
  wandb.log(epoch_metrics) 
  logger.info('Epoch %s', epoch)
  logger.info('Training acc %s',result['accuracy'])
  logger.info('Training prec non-clickbait %s',result['precision'][0])
  logger.info('Training recall non-clickbait %s',result['recall'][0])
  logger.info('Training f1 non-clickbait %s',result['f1'][0])
  logger.info('Training prec clickbait %s',result['precision'][1])
  logger.info('Training recall clickbait %s',result['recall'][1])
  logger.info('Training f1 clickbait %s',result['f1'][1])
  logger.info('Training epoch loss %s',all_losses)

  training_time = format_time(time.time() - t0)                    
  logger.info("  Training epcoh took: {:}".format(training_time))
  validation_metrics = transformer_validate_model(model,test_dl,config,epoch)
  return validation_metrics


def transformer_validate_model(model, test_dl,config,epoch,eval_type = 'dev'):
  """
    Validate a transformer-based model.

    Args:
        model (nn.Module): The transformer-based model to validate.
        test_dl: The data loader for validation data.
        config: Configuration settings for validation.
        epoch (int): The current training epoch number.
        eval_type (str): The type of evaluation ('dev' or 'test').

    Returns:
        dict: Metrics and statistics for the validation.

    This function evaluates a transformer-based model using the provided validation data and configuration settings.
    It computes loss values, logs validation progress, and returns validation metrics.
  """

  print("Running Test...")
  t0 = time.time()
  val_step_outputs = []
  nll_loss = torch.nn.CrossEntropyLoss()
  # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
  model.eval()

  for batch in test_dl:
    # Unpack this training batch from our dataloader. 
    b_input_ids = batch[0].to(config.device)
    b_input_mask = batch[1].to(config.device)
    b_labels = batch[2].to(config.device)
    # Tell pytorch not to bother with constructing the compute graph during the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        
      loss,logits,hidden_states = model(input_ids=b_input_ids,attention_mask=b_input_mask,labels=b_labels)
    
    # Accumulate the test loss.
    preds = torch.argmax(logits, dim=1)
    loss = nll_loss(logits, b_labels)
    wandb.log({f'{eval_type}/loss':loss})
  
    val_step_outputs.append(OrderedDict({'loss':loss.detach().cpu(),\
                                          'preds':preds.detach().cpu(),'labels':b_labels.detach().cpu()}))
  
  all_losses = np.mean([x['loss'].item() for x in val_step_outputs])
  
  all_preds = [y.item() for x in val_step_outputs for y in x['preds']]
  all_labels =  [y.item() for x in val_step_outputs for y in x['labels']]
  result = compute_metrics((all_preds,all_labels))
  epoch_metrics = {f"{eval_type}/epoch_accuracy": result['accuracy'],
                   f"{eval_type}/epoch_precision_nc": result['precision'][0],
                   f"{eval_type}/epoch_recall_nc":result['recall'][0],
                   f"{eval_type}/epoch_f1_nc": result['f1'][0],
                   f"{eval_type}/epoch_precision": result['precision'][1],
                   f"{eval_type}/epoch_recall": result['recall'][1],
                   f"{eval_type}/epoch_f1": result['f1'][1],
                   f"{eval_type}/epoch_loss":all_losses,
                   "epoch":epoch}
  if eval_type == 'dev':
    wandb.log(epoch_metrics)
    
  logger.info(f'{eval_type} acc %s',result['accuracy'])
  logger.info(f'{eval_type} prec clickbait  %s',result['precision'][1])
  logger.info(f'{eval_type} recall clickbait  %s',result['recall'][1])
  logger.info(f'{eval_type} f1 clickbait  %s',result['f1'][1])
  logger.info(f'{eval_type} prec non clickbait  %s',result['precision'][0])
  logger.info(f'{eval_type} recall non clickbait  %s',result['recall'][0])
  logger.info(f'{eval_type} non clickbait  %s',result['f1'][0])
  logger.info(f'{eval_type} loss  %s',all_losses)
  test_time = format_time(time.time() - t0)
  print("  Test took: {:}".format(test_time))
  if eval_type == 'test':
    report = classification_report(all_labels,all_preds,target_names = ['Non Clickbait','Clickbait'],digits=4)
    fout = open(os.path.join(config.output_path,'test_report.txt'),'w')
    fout.write(report)
    fout.close()
    
  return epoch_metrics