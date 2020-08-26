# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/24 16:17
# software: PyCharm
from transformers import BertPreTrainedModel

from config import MODEL_CLASSES


class LanguageHierarchicalClassification(BertPreTrainedModel):
    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_labels
        self.num_labels = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageHierarchicalClassification, self).__init__(bert_config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)






