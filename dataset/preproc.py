from bnunicodenormalizer.langs import bangla
import argparse
import pandas as pd
import re
import numpy as np
import unicodedata
from normalizer import normalize
from bnunicodenormalizer import Normalizer 

bnorm=Normalizer(legacy_maps='default')
puncs = bangla.punctuations
puncs = list(set(puncs))


def remove_unnec_punc_within_words(word):
    """
    Remove unnecessary punctuation characters within a Bengali word.

    Args:
        word (str): The Bengali word to process.

    Returns:
        str: The word with unnecessary punctuation removed.
    """
    global puncs
    new_word=""
    for char in word:
        if not(char in puncs and char!='-' and char!=':' and char!='।'):
            new_word+=char
    return new_word

def reverse_broken_nukta_norm(word):
    """
    Reverse the normalization of specific Bengali characters containing nukta.

    Args:
        word (str): The Bengali word to process.

    Returns:
        str: The word with reversed normalization.
    """
    global bnorm
    new_word=''
    for char in word:
        if char=="য়":
            char = 'য'+bnorm.lang.nukta
        # this normalization increases OOV for transformer tokenizers
        # if char=="র":
        #     char = "ব"+bnorm.lang.nukta
        if char=="ড়":
            char = "ড"+bnorm.lang.nukta
        if char=="ঢ়":
            char = "ঢ"+bnorm.lang.nukta
        new_word+=char
    return new_word

def preproc_pipeline(text):
    """
    Apply a preprocessing pipeline to normalize and clean Bengali text.

    Args:
        text (str): The Bengali text to preprocess.

    Returns:
        List[str]: A list of preprocessed Bengali words.
    """
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
        normalized_text[index] = word[:start]+new_word+word[end+1:]

    normalized_text = [bnorm(t)['normalized'] for t in normalized_text]
    normalized_text = [reverse_broken_nukta_norm(t) for t in normalized_text if t is not None]
    return normalized_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',type=str,default='dataset/data/train.csv',help='file to preprocess')
    parser.add_argument('--col_name',type=str,default='cleaned_title',help='file to preprocess')
    parser.add_argument('--out_col_name',type=str,default='cleaned_title',help='file to preprocess')
    parser.add_argument('--save_path',type=str,default='dataset/data/train.csv',help='file to preprocess')
    args = parser.parse_args()
    
    new_titles = []
    data_df = pd.read_csv(args.file_path)
    data_df = data_df.dropna(subset=[args.col_name])
    dropped_rows = []
    for i,text in enumerate(data_df[args.col_name]):
        print(f'processing {i}')
        new_title = ' '.join(preproc_pipeline(text)).strip()
        if new_title:
            new_titles.append(new_title)
        else:
            new_titles.append(None)
            dropped_rows.append(i)
    
    data_df[args.out_col_name] = new_titles
    data_df = data_df.drop(dropped_rows)
    data_df.to_csv(args.save_path,index=None)