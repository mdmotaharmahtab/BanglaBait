import re, os
import logging
import datetime
import collections
import torch
import torch.nn as nn

from bnunicodenormalizer.langs import bangla
from normalizer import normalize
from bnunicodenormalizer import Normalizer 

bnorm=Normalizer(legacy_maps='default')
puncs = bangla.punctuations
puncs = list(set(puncs))


def remove_unnec_punc_within_words(word):
    global puncs
    new_word=""
    for char in word:
        if not(char in puncs and char!='-' and char!=':' and char!='।'):
            new_word+=char
    return new_word

def reverse_broken_nukta_norm(word):
    global bnorm
    new_word=''
    for char in word:
        if char=="য়":
            char = 'য'+bnorm.lang.nukta
        # if char=="র":
        #     char = "ব"+bnorm.lang.nukta
        if char=="ড়":
            char = "ড"+bnorm.lang.nukta
        if char=="ঢ়":
            char = "ঢ"+bnorm.lang.nukta
        new_word+=char
    return new_word

def preproc_pipeline(text):
    global bnorm
    normalized_text = text.replace('\u2013','-').\
                            replace('\u2018',"'").\
                            replace('\u2019',"'").replace('\u201c',"'").\
                            replace('\u201d',"'").replace('\u2014',"-").\
                            replace('- Techzoom.TV','').\
                            replace('বিডি-প্রতিদিন','')
    '''
    single quotation left ‘  u2018
    single quotation right ’ u2019
    double quotation left “  u201c
    double quotation right ” u201d
    different dash — u2014
    '''
    normalized_text = re.sub(r'(তথ্যসূত্র.*)|(সুত্র.*)|(সূত্র.*)','',normalized_text,flags=re.U|re.S)
    normalized_text = re.sub(r'(আরো পড়ুন.*)|(আরও পড়ুন.*)','',normalized_text,flags=re.U|re.S) 
    
    normalized_text = normalize(
    normalized_text,
    unicode_norm="NFKC",         
    punct_replacement=None,      
    url_replacement='',         
    emoji_replacement='',       
    apply_unicode_norm_last=True)

    normalized_text=normalized_text.split()

    for index in range(len(normalized_text)):
        word = normalized_text[index]
        if len(word)==1:
            continue
        start,end = 0,-1
        while start<len(word) and word[start] in puncs:
            start+=1
        while end>=-len(word) and word[end] in puncs:
            end-=1
        end = len(word)+end
        new_word = word[start:end+1]
        new_word = remove_unnec_punc_within_words(new_word)
        # print(word[:start]+new_word+word[end+1:])
        normalized_text[index] = word[:start]+new_word+word[end+1:]


    # print(text)
    normalized_text = [bnorm(t)['normalized'] for t in normalized_text]
    # if None in normalized_text:
    #     print(normalized_text,i)
    normalized_text = ' '.join([reverse_broken_nukta_norm(t) for t in normalized_text if t is not None])
    return normalized_text


def format_time(elapsed):
    # Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_id_from_url(url:str):
    """
    Extracts the file ID from a public download url. Supppose, the donwload url
    is - https://drive.google.com/file/d/1bjHvSQrKLtIYdextXBBKrk2l5P_xWdE1/view?usp=share_link
    the function will return the file id  - 1bjHvSQrKLtIYdextXBBKrk2l5P_xWdE1
    Args:
        url (str): share link of the file (google drive link for now)
    """

    url = url.split("https://drive.google.com/file/d/")[1]
    url = re.split("/view\?usp=(sharing|share_link)",url)[0]
    return url


def setup_logger(logger,out_path):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    fh = logging.FileHandler(os.path.join(out_path,'output.log'))
    fh.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(fh)
    
def class_from_args(args):
    class ArgsClass:
        def __init__(self):
            self.__dict__.update(args)
    args_main = ArgsClass()
    return args_main

def generate_ce_weights(train_labels):
    """
    Generate class weights for Cross-Entropy Loss (CE) based on the distribution of class labels in the training data.

    This function calculates class weights to address class imbalance during the training of a machine learning model.
    The class weights are used in conjunction with CE loss to give higher importance to underrepresented classes.

    Args:
        train_labels (list or array-like): A list or array containing class labels for the training dataset.

    Returns:
        torch.Tensor: A tensor containing class weights, where each weight corresponds to a class label.
                     These weights can be used in a loss function during training.

    """
    d = collections.Counter(train_labels)
    data_dist = list(dict(collections.Counter(train_labels)).values())
    max_example = max(data_dist)
    min_example = min(data_dist)
    class_weight = []
    max_example_index = data_dist.index(max_example)
    for i in range(len(data_dist)):
        class_weight.append((data_dist[i]/max_example)*10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    crit_weights = torch.tensor(class_weight).to(device)
    return crit_weights


def load_model(model:nn.Module, load_path: str, device: str):
   model.load_state_dict(torch.load(load_path,map_location=device)['model'])
   return model