## BanglaBait: Semi-Supervised Adversarial Approach for Clickbait Detection on Bangla Clickbait Dataset

**Abstract**

Intentionally luring readers to click on a particular content by exploiting their curiosity defines a title as clickbait. Although several studies focused on detecting clickbait titles in English articles, low-resource language like Bangla has not been given adequate attention. To tackle clickbait titles in Bangla, we have constructed the first Bangla clickbait detection dataset containing 15,406 labeled news articles and 65,405 unlabelled news articles extracted from clickbait-y news sites. Each article has been labeled by three expert linguists and includes an article's title, body, and other metadata. By incorporating labeled and unlabelled data, we finetune a pre-trained Bangla transformer model in an adversarial fashion using Semi-Supervised Generative Adversarial Networks (SS-GANs). The proposed model acts as a good baseline for this dataset, outperforming traditional neural network models (LSTM, GRU, CNN) and linguistic feature-based models. We expect that this dataset and the detailed analysis and comparison of these clickbait detection models will provide a fundamental basis for future research into detecting clickbait titles in Bengali articles. Full paper can be accessed [here](https://aclanthology.org/2023.ranlp-1.81/). The dataset can be accessed [here](https://www.kaggle.com/datasets/motaharmahtab/banglabait-bangla-clickbait-dataset/data).

## Guide

- [Usage](#usage)
- [BanglaBait Dataset](#banglabait-dataset)
- [Training](#training)
  - [Statistical Models](#statistical-models)
  - [NN Models](#nn-models)
  - [Transformer Models](#transformer-models)
  - [Semi Supervised GAN Transformer Models](#semi-supervised-gan-transformer-models)

## Usage
 Requires the following packages:
 * Python 3.8+

It is recommended to use virtual environment packages such as **virtualenv** or **conda** 
Follow the steps below to setup project:
* Clone this repository. 
```bash
git clone https://github.com/mdmotaharmahtab/BanglaBait-Semi-Supervised-Adversarial-Approach-for-Clickbait-Detection-on-Bangla-Clickbait-Dataset.git
```
* create a conda environment and activate it-
```bash
conda create -n env_name python=3.8
conda activate env_name
```
* Install necessary paackages - 
 `pip install -r requirements.txt`
* Run  `bash download_data.sh` file to download Bangla Clickbait data. It also downloads Bangla Word2Vec, Fasttext embedding and Bangla Pars-of-Speech model from [bnlp toolkit](https://github.com/sagorbrur/bnlp).

## BanglaBait Dataset
The datasets used in our paper are available [here](https://drive.google.com/file/d/1vxlnsbafgET6s6c8X9l2rsAPcvOC8kWx/view?usp=sharing)

#### List of files
* train.csv
* dev.csv
* test.csv
* unlabelled_data.csv

**File Format**

| Column Title   | Description |
| ------------- |------------- |
| domain      | article link |
| data      | article publish time |
| category | Category of the news|
| title | Headline of the news|
| content | Article or body of the news|
| translated_title | English translated headline of the news|
| translated_content | English translated article or body of the news|
| label | 1 or 0 . '1' for clickbait '0' for non-clickbait|

## Training
  
### Statistical Models
* Go to root dir of repo
* Run:
```bash
python train/run_statistical.py --config config/statistical_config.yaml
```
* Models:
    * LR (Logistic Regression)
    * RF (Random Forest)
    
To choose between Logistic Regression or Random Forest model and choose which features to use, edit `statistical_config.yaml` file.
The config files are based on yaml format.

* `LR`: Logistic Regression
  * `C`: LR - inverse of regularization strength
  * `penalty`: LR - norm of the penalty
  * `solver`: LR -solver
* `RF`: Random Forest 
  * `criterion`: RF - function for quality split
  * `max_depth`: RF - max tree depth
  * `max_features`: RF - best split max feature number
  * `n_estimators`: RF - number of trees
* `data_column`: column name in dataframe which used to train model.
* `data_dir`: Folder path of clickbait data. default `dataset/data/clickbait`.
* `feature`: Feature name to use to train model
* `label_column`: column name in dataframe holds data labels
* `model_type`: which model to use LR/RF.
* `save_model`: whether to save the trained model or not

Sample config file format to run logisitc regression model by using unigram feature:

```yaml
LR:
  C: 10
  penalty: l2
  solver: liblinear
RF:
  criterion: gini
  max_depth: 100
  max_features: sqrt
  n_estimators: 700
data_column: title # use this column in dataframe for training data
data_dir: dataset/data/clickbait
feature: unigram # which feature to use for training
label_column: label # which column in dataframe contains clickbait label
model_type: LR
save_model: true # if true, save trained model
```
All possible features for statistical models are derived from [BanFake paper](https://github.com/Rowan1697/FakeNews).
* **Possible choices of feature in `statistical_config.yaml` file**:
    * unigram
    * bigram
    * trigram
    * u_b_t: uni,bi and trigram combined
    * char_3: character 3 gram
    * char_4
    * char_5
    * char_3_4_5
    * lexical: u_b_t and char_3_4_5 combined
    * pos: normalized frequency of pars-of-speech
    * L_POS: pars-of-speech and lexical combined
    * word_100: Bangla word2vec feature
    * word_300: Bangla Fasttext feature
    * L_POS_Emb_F: lexical, word_300,part-of-speech combined
    * L_POS_Emb_N: lexical, word_100,part-of-speech combined
    * mp: normalized punctuation frequency feature
    * L_POS_Emb_F_MP: lexical, word_300, part-of-speech, punctuation combined
    * L_POS_Emb_N_MP: lexical, word_100, part-of-speech, punctuation combined
    * allfeatures: all features combined


### NN Models
* Go to root dir of repo
* Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train/run_nn.py --config config/nn_config.yaml
```
* Models:
    * BiLSTM-Attention or only BiLSTM 
    * CNN

To choose between BiLSTM or CNN model and choose which embedding to use, edit `nn_config.yaml` file.
Sample config file for a BiLSTM-attn model with Bangla Fasttext embedding is given below - 

```yaml
model_type: lstm
use_class_weight: true # whether to use weighted cross entropy loss

lstm:
  use_attention: true # whether to use self attention layer on top of biLSTM stack
  out_dropout_rate : 0.5 #dropout rate between LSTM layers
  lstm_hidden_dim: 256 #lstm layer hidden size
  lstm_layers: 2 #number of biLSTM layers
  bidirectional: true #if true, use bidirectional lstm
  pad_idx : 0 #vocab id of pad token
  
cnn:
  in_channels : 1 #number of input channels, always 1
  out_channels : 256 # number of output channels
  kernel_heights : [1, 2, 3, 4] # kernel lengths
  stride : 1 # convolution stride
  padding : 0 # vocab id of pad token
  keep_probab : 0.8 # Probability of retaining an activation node during dropout operation

learning_rate : !!float 2e-05
batch_size : 64

embedding: fasttext
embedding_path: embedding/bengali_fasttext_wiki.bin # relative path of bangla embedding
update_embedding: false # whether to update bangla pretrained word embeddings
load_embedding_path: embedding/embedding_weights_fasttext.pt # if word embedding tensor is saved in this path, load from this path
epochs : 20
device : cuda
output_path: output/lstm_test_1
data_dir: dataset/data/clickbait
data_column: title # use this column in dataframe for training data
label_column: label
label_list: [0,1] # 0=non clickbait, 1=clickbait
```

To use Bangla Word2Vec features, change following:
```yaml
embedding: word2vec
embedding_path: embedding/bangla_word2vec/bnwiki_word2vec.vector
```

To use CNN model, change following: 
```yaml 
model_type: cnn
```
To run BiLSTM model without attention, change following:
```yaml
lstm:
  use_attention: false
```
Other parameters can also be tweaked from the `nn_config.yaml` file. 
Like `lstm_layers` control the number of stacked LSTM layers and `bidirectional` controls 
whether to use bidirectional LSTM or unidirectional LSTM.

### Transformer Models
* Go to root dir of repo
* Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train/run_transformer.py --config config/transformer_config.yaml
```
To choose between differenet transformer models and tweak model parameters, edit `transformer_config.yaml` file.
Sample config file for a BanglaBERT model is given below - 

```yaml
batch_size: 64
data_dir: dataset/data/clickbait
device: cuda
epochs: 20
label_list:
- 0
- 1
learning_rate: 1.0e-05
model_name_or_path: csebuetnlp/banglabert # transformers model name from huggingface hub
model_type: transformer # transformer for supervised training
out_dropout_rate: 0.5 # dropout layer between transformer and final classification layer
output_path: output/transformer_banglaberttest_1
tokenizer_class: AutoTokenizer
warmup_proportion: 0.1 # warmup rate for learning rate scheduler
```
The model_name_or_path parameter can be changed to test other pre-trained transformer models.


### Semi Supervised GAN Transformer Models
* Go to root dir of repo
* Run:
```bash
CUDA_VISIBLE_DEVICES=0 python train/run_transformer.py --config config/gan_config.yaml
```
To choose between differenet transformer models and tweak model parameters, edit `gan_config.yaml` file.
Sample config file for a GAN-BanglaBERT model is given below - 

```yaml
batch_size: 64
data_dir: dataset/data/clickbait
device: cuda
epochs: 20
epsilon: 1.0e-08
label_list:
- 0
- 1
- 2
learning_rate_discriminator: 1.0e-05
learning_rate_generator: 1.0e-05
model_name_or_path: csebuetnlp/banglabert
model_type: gan
num_hidden_layers_d: 2  #number of fully connected layers in discriminator network
num_hidden_layers_g: 2 #number of fully connected layers in generator network
out_dropout_rate: 0.5
output_path: output/gan_banglabert_test_1
tokenizer_class: AutoTokenizer
warmup_proportion: 0.1
```
The model_name_or_path parameter can be changed to test other pre-trained transformer models. 
Other parameters that can be tweaked to change model architecture are -

The `model_name_or_path` parameter can be changed to train other transformer models.
