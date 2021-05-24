# CasRel Model Pytorch reimplement 3
The code is the PyTorch reimplement of the paper "A Novel Cascade Binary Tagging Framework for Relational Triple Extraction" ACL2020. 
The [official code](https://github.com/weizhepei/CasRel) was written in keras. 

I have encountered a lot of troubles with the keras version, so I decided to rewrite the code in PyTorch.
# Introduction
I followed the previous work of [longlongman](https://github.com/longlongman/CasRel-pytorch-reimplement) 
and [JuliaSun623](https://github.com/JuliaSun623/CasRel_fastNLP).

So I have to express sincere thanks to them.

I made some changes in order to better apply to the Chinese Dataset.
The changes I have made are listed:
- I changed the tokenizer from HBTokenizer to BertTokenizer, so Chinese sentences are tokenized by single character.
  (Note that you don't need to worry about keras)
- I substituted the original pretrained model with 'bert-base-chinese'.
- I used fastNLP to build the datasets.
- I changed the encoding and decoding methods in order to fit the Chinese Dataset.
- I reconstruct the structure for readability.
# Requirements
- torch==1.8.0+cu111
- transformers==4.3.3
- fastNLP==0.6.0
- tqdm==4.59.0
- numpy==1.20.1
# Dataset
I preprocessed the open-source dataset from Baidu. I did some cleaning, so the data given have 18 relation types. 
Some noisy data are eliminated.

The data are in form of json. Take one as an example:
```json
{
    "text": "陶喆的一首《好好说再见》推荐给大家，希望你们能够喜欢",
    "spo_list": [
        {
            "predicate": "歌手",
            "object_type": "人物",
            "subject_type": "歌曲",
            "object": "陶喆",
            "subject": "好好说再见"
        }
    ]
}
```
In fact the field object_type and subject_type are not used.

If you have your own data, you can organize your data in the same format.
# Usage
```
python run.py
```
I have already set the default value of the model, but you can still set your own configuration in model/config.py
# Results
The best F1 score on test data is 0.78 with a precision of 0.80 and recall of 0.76.

It is to my expectation although it may not reach its utmost.

I have also trained the [SpERT](https://github.com/lavis-nlp/spert) model, 
and CasRel turns out to perform better. 
More experiments need to be carried out since there are slight differences in both criterion and datasets.

# Experiences
- Learning rate 1e-5 seems a good choice. If you change the learning rate, the model will be dramatically affected.
- It shows little improvement when I substitute BERT with RoBERTa.
- It is crucial to shuffle the datasets in order to avoid overfitting. 



