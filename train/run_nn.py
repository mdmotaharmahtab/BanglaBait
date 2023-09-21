import logging
import wandb
import torch
import os
import yaml

import sys
sys.path.insert(0,os.getcwd())
from utils.functions import setup_logger
import pandas as pd
import pickle
import gensim
from gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors
from utils.functions import generate_ce_weights, preproc_pipeline

import argparse
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import OneCycleLR

from dataset.nn_dataset import ClickbaitDataset, pad
from model.nn import ClickbaitLSTM, ClickbaitLSTMAttention, CNN, nn_save_model

from train.nn_train_epoch import nn_train_epoch, nn_validate_model
import pdb
from tqdm import tqdm

logger = logging.getLogger()
   
def main(config_file):
    with open(config_file,'r') as f:
            config = yaml.safe_load(f)
    
    config['model'] = config[f"{config['model_type']}"]
    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
        
    setup_logger(logger,config['output_path'])

    best_dev_f1, best_epoch = 0,0
    seed_everything(42)
    
    # run = wandb.init(project="banbait",entity='colab-team',config=config)
    run = wandb.init(project="banbait",config=config)
    config = wandb.config
    model_type = config.model_type
    data_dir = config.data_dir
    #load data based on model type
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            if 'train' in file_name:
                traindf = pd.read_csv(os.path.join(data_dir,file_name))
                traindf = traindf.sample(frac=1,random_state=1234).reset_index(drop=True)
                cleaned_titles = [preproc_pipeline(title) for title in traindf[config.data_column].values]
                train_examples = [(text,label) for text,label in zip(cleaned_titles,traindf[config.label_column].values)]
                train_dataset = ClickbaitDataset(train_examples,max_seq_length = 64)
                train_dl = torch.utils.data.DataLoader(train_dataset,batch_size = config.batch_size,collate_fn=pad)

            elif 'test' in file_name:
                testdf = pd.read_csv(os.path.join(data_dir,file_name))
                cleaned_titles = [preproc_pipeline(title) for title in testdf[config.data_column].values]
                test_examples = [(text,label) for text,label in zip(cleaned_titles,testdf[config.label_column].values)]
                test_dataset = ClickbaitDataset(test_examples,max_seq_length = 64)
                test_dl = torch.utils.data.DataLoader(test_dataset,batch_size = config.batch_size,collate_fn=pad)
                
            elif 'dev' in file_name or 'val' in file_name:
                devdf = pd.read_csv(os.path.join(data_dir,file_name))
                cleaned_titles = [preproc_pipeline(title) for title in devdf[config.data_column].values]
                dev_examples = [(text,label) for text,label in zip(cleaned_titles,devdf[config.label_column].values)]
                dev_dataset = ClickbaitDataset(dev_examples,max_seq_length = 64)
                dev_dl = torch.utils.data.DataLoader(dev_dataset,batch_size = config.batch_size,collate_fn=pad)

    config.num_train_examples = len(train_examples)

    # load the embedding tensor directly from saved tensor file if load_embedding = True
    if os.path.exists(config.load_embedding_path):
        embedding_weights = torch.load(config.load_embedding_path)
    
    else:
        # laod embedding model
        if config.embedding == 'fasttext':
            ft_bangla =load_facebook_model(config.embedding_path,encoding='utf-8')
            wordvec = ft_bangla.wv
            del ft_bangla
        elif config.embedding == 'word2vec':
            wordvec = KeyedVectors.load_word2vec_format(config.embedding_path)
        # create embedding tensor from the vocab of the tokenizer
        vocab = train_dataset.vectorizer.get_vocabulary()
        MAX_VOCAB_SIZE = train_dataset.vectorizer.vocabulary_size()
        embedding_weights= torch.randn(MAX_VOCAB_SIZE,wordvec[0].shape[0])
        for i in tqdm(range(2,MAX_VOCAB_SIZE)):
            word = vocab[i]
            if word in wordvec.index_to_key:
                embedding_weights[i,:] = torch.FloatTensor(wordvec.get_vector(word)) 
        # for [UNK] token        
        embedding_weights[1]=torch.zeros(100)
        torch.save(embedding_weights,'embedding/embedding_weights_fasttext.pt',
                   pickle_protocol=pickle.HIGHEST_PROTOCOL)

    if config.use_class_weight:
        labels = [e[1] for e in train_examples]
        class_weights = generate_ce_weights(labels)
    
    config.output_dim = len(config.label_list)

    # load nn model CNN/LSTM
    if model_type == 'cnn':
        clickbaitmodel = CNN(embedding_weights,config,class_weights)
    
    elif model_type == 'lstm':
        if config.model['use_attention']:
            clickbaitmodel = ClickbaitLSTMAttention(embedding_weights,config,class_weights)
        else:
            clickbaitmodel = ClickbaitLSTM(embedding_weights,config,class_weights)
        
    # load model and schedular
    optimizer = torch.optim.Adam(
                clickbaitmodel.parameters(),
                lr=config.learning_rate,
                weight_decay=5e-4
            )
    steps_per_epoch = len(train_dl.dataset)//config.batch_size + 1
    scheduler = OneCycleLR(
            optimizer,
            0.001,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch
        )
    train_epoch = nn_train_epoch
    save_model = nn_save_model
    validate_model = nn_validate_model
    clickbaitmodel.to(config.device)
    for epoch in tqdm(range(config.epochs)):
        validation_metrics = train_epoch(clickbaitmodel,
                                        train_dl,
                                        dev_dl, 
                                        optimizer,
                                        scheduler,
                                        config,
                                        epoch+1)
        
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
          save_model(os.path.join(config.output_path,
                                  f'{model_type}.pt'),
                                  clickbaitmodel,
                                  optimizer)
          save_best_model = False 
        
    test_metrics = validate_model(clickbaitmodel,test_dl,
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
    parser = argparse.ArgumentParser(description="nn model training")
    parser.add_argument("--config",type=str,required=True,metavar='PATH',help='The nn model config file path')
    args = parser.parse_args()
    main(args.config)