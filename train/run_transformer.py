import sys
import logging
import wandb
import torch
import os
sys.path.insert(0,os.getcwd())

import yaml
from utils.functions import setup_logger

import argparse
from tqdm import tqdm
from pytorch_lightning import seed_everything

from dataset.transformer_dataset import ClickbaitDataModule
from model.gan import GAN, build_gan_optimizer, gan_save_model
from model.transformer import TransformerForSequenceClassification, \
                              build_transformer_optimizer,\
                              transformer_save_model

from train.transformer_train_epoch import transformer_train_epoch, transformer_validate_model
from train.gan_train_epoch import gan_train_epoch, gan_validate_model
import pdb

logger = logging.getLogger()
   
def main(config_file: str):
    with open(config_file,'r') as f:
            config = yaml.safe_load(f)
            
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
        
    setup_logger(logger,config['output_path'])

    best_dev_f1, best_epoch = 0,0
    seed_everything(42)
    
    # run = wandb.init(project="banbait",entity='colab-team',config=config)
    run = wandb.init(project="banbait",config=config)
    config = wandb.config
    model_type = config.model_type
    ratio = getattr(config, 'ratio',100)
    main_dataset = ClickbaitDataModule(model_name_or_path = config.model_name_or_path, 
                                       label_list = config.label_list, batch_size = int(config.batch_size),
                                       data_folder = config.data_dir,
                                       data_column= config.data_column,
                                       label_column= config.label_column,
                                       model_type = model_type,
                                       tokenizer_class = config.tokenizer_class,
                                       ratio = ratio)
    config.num_train_examples = len(main_dataset.train_examples)
    if model_type == 'gan':
      clickbaitmodel = GAN(
          model_name_or_path = config.model_name_or_path,
          num_hidden_layers_g=config.num_hidden_layers_g,
          num_hidden_layers_d=config.num_hidden_layers_d,
          out_dropout_rate = config.out_dropout_rate,
          label_list = config.label_list)
      optimizers = build_gan_optimizer(clickbaitmodel, config)
      train_epoch = gan_train_epoch
      validate_model = gan_validate_model
      save_model = gan_save_model
    elif model_type == 'transformer':
        clickbaitmodel = TransformerForSequenceClassification(model_name_or_path=config.model_name_or_path,
                                                              num_classes=len(config.label_list))
        optimizers = build_transformer_optimizer(clickbaitmodel,config)
        train_epoch = transformer_train_epoch
        validate_model = transformer_validate_model
        save_model = transformer_save_model
    
    for epoch in tqdm(range(config.epochs)):
        validation_metrics = train_epoch(clickbaitmodel,main_dataset, optimizers,config,epoch+1)
        dev_f1 = validation_metrics['dev/epoch_f1']
        save_best_model = False
        if epoch==0:
          best_dev_f1 = dev_f1
          save_best_model = True

        elif dev_f1>best_dev_f1:
          best_dev_f1 = dev_f1
          save_best_model = True
          best_epoch= epoch

        if save_best_model:
          save_model(os.path.join(config.output_path,f'{model_type}.pt'),clickbaitmodel,optimizers)
          save_best_model = False 
        
    test_metrics = validate_model(clickbaitmodel,main_dataset.test_dataloader(),
                                  config,epoch,'test')
    run.summary['best_dev_f1'] = best_dev_f1
    run.summary['test_f1'] = test_metrics['test/epoch_f1']
    test_f1 = run.summary['test_f1']
    run.summary['best epoch based on dev F1 score'] = best_epoch
    
    logger.info('Best Epoch based on dev F1 score %s:',best_epoch)
    model_artifact = wandb.Artifact(
          f"{model_type}_{test_f1:.3f}", type=f'{model_type}',
          description=f"{model_type}-test f1 clickbait:{test_f1}",
          metadata=dict(config))
    PATH=os.path.join(config.output_path,f'{model_type}.pt')
    model_artifact.add_file(PATH)
    run.log_artifact(model_artifact)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--config",type=str,required=True,metavar='PATH',help='The model inference config file location')
    args = parser.parse_args()
    main(args.config)