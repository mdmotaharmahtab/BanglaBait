import collections
import logging
import wandb
import torch
import math
import time
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import os

from utils.functions import format_time
from utils.metrics import compute_metrics
from sklearn.metrics import classification_report


logger = logging.getLogger()

def gan_train_epoch(model, dataset, optimizers,config,epoch):
  """
    Train a Generative Adversarial Network (GAN) model for one epoch.

    Args:
        model (nn.Module): The GAN model to train.
        dataset: The dataset containing training data.
        optimizers (tuple): A tuple of optimizers for the discriminator and generator models.
        config: Configuration settings for training.
        epoch (int): The current training epoch number.

    Returns:
        dict: Metrics and statistics for the training epoch.

    This function trains a GAN model for one epoch using the specified dataset and optimization settings.
    It computes loss values, updates model parameters, and logs training progress.
  """

  training_step_outputs=[]
  # Measure how long the training epoch takes.
  t0 = time.time()
  print_each_n_step = 100
  dis_optimizer,gen_optimizer = optimizers[0]
  scheduler_d,scheduler_g= optimizers[1]
  train_dl = dataset.train_dataloader()
  dev_dl = dataset.val_dataloader()
  
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
    b_label_mask = batch[3].to(config.device)
    real_batch_size=b_input_ids.shape[0]
    hidden_states,features,logits,probs = model(b_input_ids,b_input_mask)
    
    # Finally, we separate the discriminator's output for the real and fake data
    features_list = torch.split(features, real_batch_size)
    D_real_features = features_list[0]
    D_fake_features = features_list[1]
    
    logits_list = torch.split(logits, real_batch_size)
    D_real_logits = logits_list[0]
    D_fake_logits = logits_list[1]
    
    probs_list = torch.split(probs, real_batch_size)
    D_real_probs = probs_list[0]
    D_fake_probs = probs_list[1]

    b_real_labels = torch.split(b_labels,real_batch_size)[0]
    b_labels_labelled = torch.masked_select(b_real_labels,b_label_mask.to(config.device))
    filtered_logits = D_real_logits[:,0:-1]
    D_real_preds = torch.argmax(filtered_logits, dim=1)
    D_preds_labelled = torch.masked_select(D_real_preds, b_label_mask.to(config.device))

    #---------------------------------
    #  LOSS evaluation
    #---------------------------------
    # Generator's LOSS estimation
    D_fake_preds = torch.argmax(D_fake_probs,dim=1)
    g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + config.epsilon))
    g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
    g_loss = g_loss_d + g_feat_reg

    # Disciminator's LOSS estimation
    logits = D_real_logits[:,0:-1]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # The discriminator provides an output for labeled and unlabeled real data so the loss evaluated for unlabeled data is ignored (masked)
    label2one_hot = torch.nn.functional.one_hot(b_labels, len(model.label_list))
    per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
    per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(config.device))
    labeled_example_count = per_example_loss.type(torch.float32).numel()

    # It may be the case that a batch does not contain labeled examples, so the "supervised loss" in this case is not evaluated
    if labeled_example_count == 0:
      D_L_Supervised = 0
    else:
      D_L_Supervised = torch.div(torch.sum(per_example_loss.to(config.device)), labeled_example_count)
              
    D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + config.epsilon))
    D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + config.epsilon))
    d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

    #---------------------------------
    #  OPTIMIZATION
    #---------------------------------
    # Avoid gradient accumulation
    gen_optimizer.zero_grad()
    dis_optimizer.zero_grad()

    # Calculate weigth updates
    # retain_graph=True is required since the underlying graph will be deleted after backward
    g_loss.backward(retain_graph=True)
    d_loss.backward() 
    
    #clip grad norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    # Apply modifications
    gen_optimizer.step()
    dis_optimizer.step()
    scheduler_d.step()
    scheduler_g.step()
    metrics = {'train/d_loss_step':d_loss,'train/g_loss_step':g_loss}
    if step + 1 < n_steps_per_epoch:
      wandb.log(metrics)
    
    training_step_outputs.append(OrderedDict({'d_loss':d_loss.detach().cpu(),\
                                              'g_loss':g_loss.detach().cpu(),\
                                              'g_loss_d':g_loss_d.detach().cpu(),\
                                              'g_feat_reg':g_feat_reg.detach().cpu(),\
                                              'preds':D_preds_labelled.detach().cpu(),\
                                              'labels':b_labels_labelled.detach().cpu(),
                                              'fake_corrects':collections.Counter(D_fake_preds.cpu().numpy())[3]}))
    
  
  all_d_losses = np.mean([x['d_loss'].item() for x in training_step_outputs])
  all_g_losses = np.mean([x['g_loss'].item() for x in training_step_outputs])
  all_g_losses_d = np.mean([x['g_loss_d'].item() for x in training_step_outputs])
  all_g_losses_feat_reg = np.mean([x['g_feat_reg'].item() for x in training_step_outputs])
  all_preds = [y.item() for x in training_step_outputs for y in x['preds']]
  all_labels =  [y.item() for x in training_step_outputs for y in x['labels']]
  result = compute_metrics((all_preds,all_labels))

  # show how many fake instances discriminator could correctly identify. 
  # This should increase over time and increase g_loss_d
  all_fake_corrects = sum([x['fake_corrects'] for x in training_step_outputs])

  epoch_metrics = {"train/epoch_accuracy": result['accuracy'],
                   "train/epoch_precision_nc": result['precision'][0],
                   "train/epoch_recall_nc":result['recall'][0],
                   "train/epoch_f1_nc": result['f1'][0],
                   "train/epoch_precision": result['precision'][1],
                   "train/epoch_recall": result['recall'][1],
                   "train/epoch_f1": result['f1'][1],
                   'train/epoch_loss':all_d_losses,
                   'train/epoch_g_loss':all_g_losses,
                   'epoch':epoch
                   }
  wandb.log(epoch_metrics) 
  
  logger.info('Epoch %s', epoch)
  logger.info('Training acc %s',result['accuracy'])
  logger.info('Training acc %s',result['accuracy'])
  logger.info('Training prec non-clickbait %s',result['precision'][0])
  logger.info('Training recall non-clickbait %s',result['recall'][0])
  logger.info('Training f1 non-clickbait %s',result['f1'][0])
  logger.info('Training prec clickbait %s',result['precision'][1])
  logger.info('Training recall clickbait %s',result['recall'][1])
  logger.info('Training f1 clickbait %s',result['f1'][1])
  logger.info('Training discriminator loss %s',all_d_losses)
  logger.info('Training generator loss %s',all_g_losses)
  logger.info('Training generator loss discriminator %s',all_g_losses_d)
  logger.info('Training generator loss features match %s',all_g_losses_feat_reg)
  logger.info(f'Discriminator correctly labelled {all_fake_corrects} instances among {n_steps_per_epoch*config.batch_size} fake datas')  

  training_time = format_time(time.time() - t0)                    
  logger.info("  Training epcoh took: {:}".format(training_time))
  validation_metrics = gan_validate_model(model,dev_dl,config,epoch)
  return validation_metrics

def gan_validate_model(model, test_dl,config,epoch,eval_type = 'dev'):
  """
    Validate a Generative Adversarial Network (GAN) model.

    Args:
        model (nn.Module): The GAN model to validate.
        test_dl: The data loader for validation data.
        config: Configuration settings for validation.
        epoch (int): The current training epoch number.
        eval_type (str): The type of evaluation ('dev' or 'test').

    Returns:
        dict: Metrics and statistics for the validation.

    This function evaluates a GAN model using the provided validation data and configuration settings.
    It computes loss values, logs validation progress, and returns validation metrics.
  """

  print("Running Test...")
  t0 = time.time()
  val_step_outputs = []
  nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
  # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
  model.eval()

  for batch in test_dl:
    # Unpack this training batch from our dataloader. 
    b_input_ids = batch[0].to(config.device)
    b_input_mask = batch[1].to(config.device)
    b_labels = batch[2].to(config.device)

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        
      model_outputs = model.transformer(b_input_ids, attention_mask=b_input_mask)
      if len(model_outputs[-1].shape) == 2:
        hidden_states = model_outputs[-1]
      else:
        hidden_states = torch.mean(model_outputs[0],dim=1)
      _, logits, probs = model.discriminator(hidden_states)
      filtered_logits = logits[:,0:-1]
    
    # Accumulate the test loss.
    preds = torch.argmax(filtered_logits, dim=1)
    loss = nll_loss(filtered_logits, b_labels)
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
    
