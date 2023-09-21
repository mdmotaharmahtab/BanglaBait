"""
Following functions are slightly modified from - 
https://github.com/Rowan1697/FakeNews/blob/master/Models/Basic/ngram.py
https://github.com/Rowan1697/FakeNews
"""

import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import re
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from scipy import sparse, hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import string
import pickle, yaml
from model.statistical import load_feature, config
import argparse
from utils.functions import class_from_args
from sklearn.metrics import classification_report

def mp(X):

    return load_feature.mp(X),"MP"


def pos(X):

    return load_feature.pos(X),"POS"



#Unigram
def unigram(X):

    return load_feature.tfidf_wordF(X, 1, 1),"Unigram"


#Bigram
def bigram(X):

    return load_feature.tfidf_wordF(X, 2, 2),"Bigram"

#Trigram
def trigram(X):

    return load_feature.tfidf_wordF(X, 3, 3),"Trigram"


#U+B+T
def u_b_t(X):

    return load_feature.tfidf_wordF(X, 1, 3),"U+B+T"


#C3
def char_3(X):

    return load_feature.tfidf_charF(X, 3, 3,True),"C3-gram"

def char_4(X):

    return load_feature.tfidf_charF(X, 4, 4),"C4-gram"

def char_5(X):

    return load_feature.tfidf_charF(X, 5, 5),"C5-gram"

def char_3_4_5(X):

    return load_feature.tfidf_charF(X, 3, 5),"C3+C4+C5"


#Linguistic
def lexical(X):

    X_char = load_feature.tfidf_charF(X, 3, 5)
    X_word = load_feature.tfidf_wordF(X, 1, 3)
    return sparse.hstack((X_word, X_char)),"Lexical"


#Word Embedding Fasttext
def word_300(X):
    return load_feature.word_emb(300,X),"Emb_F"


#Word Embedding News
def word_100(X):

    return load_feature.word_emb(100,X),"Emb_N"

#L+POS
def L_POS(X):

    return sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(X))),"L+POS"


#L+POS+Emb(F)
def L_POS_Emb_F(X):
    

    return sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(X), load_feature.word_emb(300,X))),"L+POS+Emb(F)"

#L+POS+Emb(N)
def L_POS_Emb_N(X):
    
    return sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(X), load_feature.word_emb(100,X))),"L+POS+Emb(N)"


#L+POS+E(F)+MP
def L_POS_Emb_F_MP(X):
    
    return sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(X), load_feature.word_emb(300,X), load_feature.mp(X))),"L+POS+E(F)+MP"


#L+POS+E(N)+MP
def L_POS_Emb_N_MP(X):

    return sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(X), load_feature.word_emb(100,X), load_feature.mp(X))),"L+POS+E(N)+MP"


#Allfeatures
def allfeatures(X):

    return sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(X), load_feature.word_emb(300,X), load_feature.word_emb(100,X), load_feature.mp(X))),"Allfeatures"

def main(config_file):
    with open(config_file,'r') as f:
        config = yaml.safe_load(f)

    config['model'] = config[f"{config['model_type']}"]
    config = class_from_args(config)
    config.output_path = os.path.join('output',config.feature+'_'+config.model_type)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    model_type = config.model_type
    data_dir = config.data_dir
    
    #load data based on model type
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            if 'train' in file_name:
                traindf = pd.read_csv(os.path.join(data_dir,file_name))
                traindf = traindf.sample(frac=1,random_state=1234).reset_index(drop=True)
                
            elif 'test' in file_name:
                testdf = pd.read_csv(os.path.join(data_dir,file_name))
                
            elif 'dev' in file_name or 'val' in file_name:
                devdf = pd.read_csv(os.path.join(data_dir,file_name))

    # check if model already exists
    model_path = os.path.join(config.output_path,'model.pkl')
    if os.path.exists(model_path):
        clf = pickle.load(open(model_path,'rb'))

    else:
        # train model
        X,Y =  np.concatenate((traindf[config.data_column].values,
                            devdf[config.data_column].values)), \
                np.concatenate((traindf[config.label_column].values,
                            devdf[config.label_column].values))
        
        X_features,exp = eval(f"{config.feature}(X)")

        if config.model_type == "LR":
            clf = LogisticRegression(solver=config.model['solver'],
                                    penalty=config.model['penalty'],
                                    C = config.model['C']
                                    )
        
        elif config.model_type == "RF":
            class_weight = dict({1:5,0:1})
            clf = RandomForestClassifier(bootstrap=True,
                                        class_weight=class_weight,
                                        criterion=config.model['criterion'],
                                        max_depth=config.model['max_depth'], 
                                        max_features=config.model['max_features'], 
                                        n_estimators=config.model['n_estimators'],
                                        )

        
        
        clf.fit(X_features, Y)
        if config.save_model:
            pickle.dump(clf,open(model_path,'wb'),protocol=pickle.HIGHEST_PROTOCOL)

    test_labels = testdf[config.label_column].values
    test_features,exp = eval(f"{config.feature}(testdf[config.data_column].values)")
    test_preds = clf.predict(test_features)

    report = classification_report(test_labels,test_preds,target_names = ['Non Clickbait','Clickbait'],digits=4)
    fout = open(os.path.join(config.output_path,'test_report.txt'),'w')
    fout.write(report)
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="statistical model run")
    parser.add_argument("--config",type=str,required=True,metavar='PATH',help='The statistical model config file location')
    args = parser.parse_args()
    main(args.config)




