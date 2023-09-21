#!/bin/bash
# download clickbait data
# gdown --fuzzy https://drive.google.com/file/d/1vxlnsbafgET6s6c8X9l2rsAPcvOC8kWx/view?usp=sharing
# unzip bn_clickbait_data.zip -d dataset/data
# rm bn_clickbait_data.zip
# downlaod embeddings
mkdir embedding
cd embedding
wget https://huggingface.co/sagorsarker/bangla_word2vec/resolve/main/bangla_word2vec_gen4.zip
unzip bangla_word2vec_gen4.zip 
wget https://huggingface.co/sagorsarker/bangla-fasttext/resolve/main/bengali_fasttext_wiki.zip
unzip bengali_fasttext_wiki.zip
cd ../model/statistical
wget https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos.pkl
