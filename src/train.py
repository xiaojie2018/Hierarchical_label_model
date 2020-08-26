# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/24 16:16
# software: PyCharm

import os
from argparse import Namespace

from utils import init_logger, DataPreprocess
import logging

logger = logging.getLogger(__name__)

init_logger()


class HierarchicalLabelModelTrain(DataPreprocess):

    def __init__(self, config_params):

        self.config = Namespace(**config_params)

        super(HierarchicalLabelModelTrain, self).__init__(self.config)

    def preprocess(self):

        pass

    def fit(self):

        pass

    def eval(self):

        pass


if __name__ == '__main__':
    config_params = {
        "ADDITIONAL_SPECIAL_TOKENS": [],
        "model_dir": "./output",
        "model_type": "bert",
        "task_type": ["classification", 'ner'][1],
        "model_name_or_path":
            ["E:\\nlp_tools\\bert_models\\bert-base-chinese", "/home/hemei/xjie/bert_models/bert-base-chinese"][0],
        "seed": 1234,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "max_seq_len": 200,
        "learning_rate": 5e-5,
        "num_train_epochs": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "warmup_proportion": 0.1,
        "dropout_rate": 0.5,
        "logging_steps": 50,
        "save_steps": 50,
        "no_cuda": False,
        "ignore_index": 0,
        "do_train": True,
        "do_eval": True,
        "is_attention": False,
        "is_lstm": False,
        "is_cnn": False,
        "train_file_url": "../ccks_3_nolabel_data//train_base.json",
        "test_file_url": "../ccks_3_nolabel_data//trans_train.json",
        "dev_file_url": "./o_data/train.json",
        "model_save_path": "./output/model",
        "model_decode_fc": ["softmax", "crf", "span"][1],
        "loss_type": ['lsr', 'focal', 'ce', 'bce', 'bce_with_log'][3],
        "do_adv": False,
        "adv_epsilon": 1.0,
        "adv_name": 'word_embeddings',
        "crf_learning_rate": 5e-5,
        "start_learning_rate": 0.0001,
        "end_learning_rate": 0.0001
    }

    model_type = ["bert", "ernie", "albert", "roberta", "bert_www", "xlnet_base", "xlnet_mid",
                  'electra_base_discriminator', 'electra_small_discriminator']

    pre_model_path = {
        "bert": "E:\\nlp_tools\\bert_models\\bert-base-chinese",
        "ernie": "E:\\nlp_tools\\ernie_models\\ERNIE",
        "albert": "E:\\nlp_tools\\bert_models\\albert_base_v1",
        "roberta": "E:\\nlp_tools\\bert_models\\chinese_roberta_wwm_ext_pytorch",
        "bert_www": "E:\\nlp_tools\\bert_models\\chinese_wwm_pytorch",
        "xlnet_base": "E:\\nlp_tools\\xlnet_models\\chinese_xlnet_base_pytorch",
        "xlnet_mid": "E:\\nlp_tools\\xlnet_models\\chinese_xlnet_mid_pytorch",
        "electra_base_discriminator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_discriminator_pytorch",
        # "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
        "electra_small_discriminator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_discriminator_pytorch",
        # "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    }
    lag_path = '/home/hemei/xjie/bert_models'
    pre_model_path = {
        "bert": f"{lag_path}/bert-base-chinese",
        "ernie": f"{lag_path}/ERNIE",
        "albert": f"{lag_path}/albert_base_v1",
        "roberta": f"{lag_path}/chinese_roberta_wwm_ext_pytorch",
        "bert_www": f"{lag_path}/chinese_wwm_pytorch",
        "xlnet_base": f"{lag_path}/chinese_xlnet_base_pytorch",
        "xlnet_mid": f"{lag_path}/chinese_xlnet_mid_pytorch",
        "electra_base_discriminator": f"{lag_path}/chinese_electra_base_discriminator_pytorch",
        "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
        "electra_small_discriminator": f"{lag_path}/chinese_electra_small_discriminator_pytorch",
        "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    }

    config_params['model_type'] = model_type[0]
    config_params['model_name_or_path'] = pre_model_path[config_params['model_type']]
    config_params['pretrained_model_path'] = pre_model_path[config_params['model_type']]
    config_params['model_save_path'] = "./output_2_1/model_{}_{}".format(config_params['model_type'],
                                                                         config_params['model_decode_fc'])

    hlm = HierarchicalLabelModelTrain(config_params)
    hlm.preprocess()
    hlm.fit()


