# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/24 16:16
# software: PyCharm
import codecs
import json
import os
from argparse import Namespace
from trainer import Trainer
from utils import init_logger, DataPreprocess
import logging

logger = logging.getLogger(__name__)

init_logger()


class HierarchicalLabelModelTrain(DataPreprocess):

    def __init__(self, config_params):

        self.config = Namespace(**config_params)
        self.model_save_path = self.config.model_save_path
        super(HierarchicalLabelModelTrain, self).__init__(self.config)

    def preprocess(self):

        train_data, o_label1, m_label1 = self.get_data(self.config.train_file_url)
        test_data, o_label2, m_label2 = self.get_data(self.config.test_file_url)

        o_labels = list(set(o_label1 + o_label2).difference(set(["NEG_TEXT"])))
        m_labels = list(set(m_label1 + m_label2).difference(set(["NEG_TEXT"])))

        self.config.o_labels = o_labels
        self.config.m_labels = m_labels
        self.config.num_o_labels = len(o_labels)
        self.config.num_m_labels = len(m_labels)

        self.train_data, self.train_examples = self._get_data(train_data, o_labels, m_labels, "train")
        logger.info("train data num: {} ".format(str(len(train_data))))
        self.test_data, self.test_examples = self._get_data(test_data, o_labels, m_labels, set_type="test")
        logger.info("test data num: {} ".format(str(len(test_data))))
        self.dev_data, self.dev_examples = self._get_data(test_data, o_labels, m_labels, set_type="dev")
        logger.info("dev data num: {} ".format(str(len(test_data))))

    def fit(self):

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
        self.config.model_save_path = self.model_save_path
        self.config.model_dir = self.model_save_path

        vocab_file = os.path.join(self.config.pretrained_model_path, "vocab.txt")
        out_vocab_file = os.path.join(self.model_save_path, "vocab.txt")

        f_w = open(out_vocab_file, 'w')
        with open(vocab_file, 'r') as f_r:
            for line in f_r:
                f_w.write(line)
        f_w.close()
        f_r.close()

        with codecs.open(os.path.join(self.model_save_path, '{}_config.json'.format(self.config.task_type)), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data,
                               train_examples=self.train_examples,
                               test_examples=self.test_examples,
                               dev_examples=self.dev_examples)
        self.trainer.train()

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
        "max_seq_len": 86,
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
        "logging_steps": 500,
        "save_steps": 500,
        "no_cuda": False,
        "ignore_index": 0,
        "do_train": True,
        "do_eval": True,
        "is_attention": False,
        "is_lstm": False,
        "is_cnn": False,
        "train_file_url": "../o_data/train.json",
        "test_file_url": "../o_data/dev.json",
        "dev_file_url": "../o_data/dev.json",
        "model_save_path": "./output/model",
        "model_decode_fc": ["softmax", "crf", "span"][1],
        "loss_type": ['lsr', 'focal', 'ce', 'bce', 'bce_with_log'][3],
        "do_adv": False,
        "adv_epsilon": 1.0,
        "adv_name": 'word_embeddings',
        "crf_learning_rate": 5e-5,
        "start_learning_rate": 0.0001,
        "end_learning_rate": 0.0001,
        "is_muti_label": False,
        "trans_dim": 300,
        "o_label_dim": 25
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
    # lag_path = '/home/hemei/xjie/bert_models'
    # pre_model_path = {
    #     "bert": f"{lag_path}/bert-base-chinese",
    #     "ernie": f"{lag_path}/ERNIE",
    #     "albert": f"{lag_path}/albert_base_v1",
    #     "roberta": f"{lag_path}/chinese_roberta_wwm_ext_pytorch",
    #     "bert_www": f"{lag_path}/chinese_wwm_pytorch",
    #     "xlnet_base": f"{lag_path}/chinese_xlnet_base_pytorch",
    #     "xlnet_mid": f"{lag_path}/chinese_xlnet_mid_pytorch",
    #     "electra_base_discriminator": f"{lag_path}/chinese_electra_base_discriminator_pytorch",
    #     "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
    #     "electra_small_discriminator": f"{lag_path}/chinese_electra_small_discriminator_pytorch",
    #     "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    # }

    config_params['model_type'] = model_type[0]
    config_params['model_name_or_path'] = pre_model_path[config_params['model_type']]
    config_params['pretrained_model_path'] = pre_model_path[config_params['model_type']]
    config_params['model_save_path'] = "../output/model_{}_{}".format(config_params['model_type'],
                                                                      config_params['model_decode_fc'])

    hlm = HierarchicalLabelModelTrain(config_params)
    hlm.preprocess()
    hlm.fit()


