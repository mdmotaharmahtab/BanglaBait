"""
Following functions are slightly modified from - 
https://github.com/Rowan1697/FakeNews/blob/master/Models/Basic/load_feature.py
https://github.com/Rowan1697/FakeNews
"""
import sys
import os
sys.path.insert(0,os.getcwd())
import pandas as pd
import re
import json
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import string
import collections
import pickle
from gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors
from bnlp import POS
from utils.functions import preproc_pipeline
from bnunicodenormalizer import langs
from model.statistical import config


def getStopWords():
    """
    Get a list of stop words from a JSON file.

    Returns:
        list: A list of stop words.
    """
    stopWords = json.load(open(config.STOP_WORD,'r',encoding='utf-8'))
    stopWords = stopWords['stopwords']
    return stopWords

def tokenizer(doc):
    """
    Tokenize a document by removing punctuation and other characters.

    Args:
        doc (str): The input document.

    Returns:
        list: A list of tokens after tokenization.
    """
    puncList = langs.bangla.punctuations
    # remove punctuation
    tokens = []
    def cleanword(word):
        words = preproc_pipeline(word)
        for p in puncList:
            word = word.replace(p, "")
        word = re.sub(r'[\u09E6-\u09EF]', "", word, re.DEBUG)  # replace digits

        return word

    for word in doc.split(" "):
        word = cleanword(word)
        if word != "":
            tokens.append(word)

    return tokens


def word_emb(size,X):
    """
    Create word embeddings for a given dataset using pre-trained word vectors.

    Args:
        size (int): The size of the word embeddings (e.g., 100 or 300).
        X (pd.Series): The input data.

    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix containing word embeddings.
    """
    if size == 100:
        vector = KeyedVectors.load_word2vec_format(config.EMBEDDING_100)
    elif size == 300:
        ft_bangla =load_facebook_model(config.EMBEDDING_300,encoding='utf-8')
        vector = ft_bangla.wv
        del ft_bangla    
    
    print("Vocab Size =>%s" %(len(vector.index_to_key)))
    vocab = set(vector.index_to_key)

    def doc2MeanValue(doc):
        tokens = tokenizer(doc)
        tokentovaluelist = [vector.get_vector(token) for token in tokens if token in vocab]
        return np.array(tokentovaluelist)

    df =  X

    featureVector = []
    labels = []
    for val in df:
        mean = doc2MeanValue(val)
        if mean.size == 0:
            mean = [0] * size
            featureVector.append(mean)
            continue
        mean = np.mean(mean, axis=0)
        mean = (mean.tolist())
        featureVector.append(mean)

    df = pd.DataFrame(featureVector)
    df = df.fillna(0)
    return sparse.csr.csr_matrix(df.values)


def tfidf_charF(X, a, b, save_model=True):
    """
    Generate TF-IDF features for character n-grams of a given dataset.

    Args:
        X (pd.Series): The input data.
        a (int): Minimum n-gram length.
        b (int): Maximum n-gram length.
        save_model (bool): Whether to save the TF-IDF model.

    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix containing TF-IDF features.
    """
    train_values = [preproc_pipeline(sen) for sen in X]
    name = f"tfidf_char_{a}_{b}.pkl"
    path = config.API+name
    if os.path.exists(path):
        tfidf_char = pickle.load(open(path,'rb'))
    else:
        tfidf_char = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(a, b), stop_words=getStopWords(),
                                    decode_error='replace', encoding='utf-8', analyzer='char')

        tfidf_char.fit(train_values)
        if save_model:        
            if not os.path.exists(config.API):
                os.makedirs(config.API)
            outfile = open(path, 'wb')
            pickle.dump(tfidf_char, outfile, protocol= pickle.HIGHEST_PROTOCOL)
            outfile.close()
    x_char = tfidf_char.transform(train_values)
    return x_char


def tfidf_wordF(X, a, b, save_model = True):
    """
    Generate TF-IDF features for word n-grams of a given dataset.

    Args:
        X (pd.Series): The input data.
        a (int): Minimum n-gram length.
        b (int): Maximum n-gram length.
        save_model (bool): Whether to save the TF-IDF model.

    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix containing TF-IDF features.
    """
    train_values = [preproc_pipeline(sen) for sen in X]
    name = f"tfidf_word_{a}_{b}.pkl"
    path = config.API+name
    if os.path.exists(path):
        tfidf_word = pickle.load(open(path,'rb'))
    else:
        tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(a, b),
                                    stop_words=getStopWords(), decode_error='replace',
                                    encoding='utf-8', analyzer='word', tokenizer=tokenizer)
        
        tfidf_word.fit(train_values)

        if save_model:
            if not os.path.exists(config.API):
                os.makedirs(config.API)
            outfile = open(path, 'wb')
            pickle.dump(tfidf_word, outfile)
            outfile.close()
        
    x_word = tfidf_word.transform(train_values)
    return x_word


def mp(X):
    """
    Calculate the normalized frequency of punctuation marks in each document of a given dataset.

    Args:
        X (pd.Series): The input data.

    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix containing punctuation counts.
    """
    puncList = set(langs.bangla.punctuations)
    def count_punc(content):
        char_list = list(content)
        count = 0
        for c in char_list:
            if c in puncList:
                count += 1
        return count
    df = X
    featureVector = []
    for val in df:
        # row = row[1]
        feature = []
        feature.append(count_punc(val))
        featureVector.append(feature)

    dfMP = pd.DataFrame(featureVector)
    normalized_df = (dfMP - dfMP.mean()) / dfMP.std()
    dfMP = normalized_df.fillna(0)
    return sparse.csr.csr_matrix(dfMP.values)


def pos(X):
    """
    Calculate the normalized frequency of part-of-speech (POS) in each document of a given dataset.

    Args:
        X (pd.Series): The input data.

    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix containing POS features.
    """
    pos_df_dict = collections.defaultdict(list)
    pos_vocab = json.load(open(config.POS_VOCAB,'r'))
    pos_vocab = eval(pos_vocab['pos_vocab'])
    model = POS()
    df = X
    for sen in df:
        sen = preproc_pipeline(sen)
        res = model.tag(config.POS_MODEL_PATH,sen)
        tag_count = dict(collections.Counter([r[1] for r in res]))
        for tag,count in tag_count.items():
            pos_df_dict[tag].append(count)
        for tag in pos_vocab:
            if tag not in tag_count:
                pos_df_dict[tag].append(0)

    pos_df = pd.DataFrame.from_dict(pos_df_dict)
    normalized_df = (pos_df - pos_df.mean()) / pos_df.std()
    normalized_df = normalized_df.fillna(0)
    X_POS = sparse.csr.csr_matrix(normalized_df.values)
    return X_POS




