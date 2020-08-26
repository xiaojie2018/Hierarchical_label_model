# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/26 12:53
# software: PyCharm


from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, \
    XLNetConfig, XLNetTokenizer, XLNetModel, AutoConfig, AutoTokenizer, AutoModel, BertModel, RobertaModel, AlbertModel

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer, BertModel),
    'bert_www': (BertConfig, BertTokenizer, BertModel),
    'roberta': (RobertaConfig, RobertaTokenizer, RobertaModel),
    'albert': (AlbertConfig, AlbertTokenizer, AlbertModel),
    'ernie': (BertConfig, BertTokenizer, BertModel),
    "xlnet_base": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "xlnet_mid": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "electra_base_discriminator": (AutoConfig, AutoTokenizer, AutoModel),
    "electra_base_generator": (AutoConfig, AutoTokenizer, AutoModel),
    "electra_small_discriminator": (AutoConfig, AutoTokenizer, AutoModel),
    "electra_small_generator": (AutoConfig, AutoTokenizer, AutoModel),
}


from model import LanguageHierarchicalClassification


MODEL_TASK = {
    "classification": LanguageHierarchicalClassification,
    "ner": None
}


