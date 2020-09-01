# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/24 16:17
# software: PyCharm
from transformers import BertPreTrainedModel
import torch
import torch.nn as nn
from config import MODEL_CLASSES
from layers import FCLayer, FCLayerSigmoid, FCLayerSoftmax, KongJianTrans


class LanguageHierarchicalClassification(BertPreTrainedModel):
    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_labels
        self.num_labels = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageHierarchicalClassification, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


class LanguageHierarchicalRelationClassification(BertPreTrainedModel):
    def __init__(self, model_dir, args):
        self.args = args
        self.num_o_labels = args.num_o_labels
        self.num_m_labels = args.num_m_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageHierarchicalRelationClassification, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.trans_layer = []
        self.trans_weight_layer = []
        self.o_label_emb = nn.Embedding(self.num_o_labels, args.o_label_dim)

        for i in range(self.num_o_labels):
            self.trans_layer.append(KongJianTrans(bert_config.hidden_size, args.trans_dim))
            self.trans_weight_layer.append(FCLayer(args.trans_dim, args.o_label_dim, dropout_rate=0, use_activation=False))

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)

        if self.args.is_muti_label:
            self.fc = FCLayerSigmoid(bert_config.hidden_size*3, self.num_m_labels)
        else:
            self.fc = FCLayerSoftmax(bert_config.hidden_size*3, self.num_m_labels)

        # loss
        self.loss_fct_bce = nn.BCELoss()

        self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def trans_voc(self, hidden_output):
        output = []
        for l in self.trans_layer:
            output.append(l(hidden_output))

        return output

    def trans_weight_voc(self, hidden_output):

        output = []
        for ind, (l, h) in enumerate(zip(self.trans_weight_layer, hidden_output)):
            h1 = l(h)
            o1 = self.o_label_emb(torch.LongTensor([ind]))
            h2 = h1.matmul(o1.transpose(0, 1)).squeeze(1)
            output.append(h2)

        # output = torch.cat(output, dim=-1)
        # o = torch.bmm(self.o_label_emb, output)

        return output

    @staticmethod
    def add_weight(o_s, o_w):
        """
        :param o_s: [ (16, 86, 300), , , , ]
        :param o_w: [ (16), , , , ]
        :return:
        """
        res = []
        a, b = o_s[0].shape[1], o_s[0].shape[2]
        for i in range(o_s[0].shape[0]):
            s = torch.zeros((a, b))
            for j in range(len(o_s)):
                s += o_s[j][i]*o_w[j][i]
            res.append(s.unsqueeze(0))
        return torch.cat(res, dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, o_label=None, m_label=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator", "electra_base_generator",
                                    "electra_small_discriminator", "electra_small_generator"]:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[0][:, 0, :]
        else:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        # 空间变换

        o_s = self.trans_voc(sequence_output)
        o_p = self.trans_voc(pooled_output)

        # o_s1 = self.trans_weight_voc(o_s)
        o_w = self.trans_weight_voc(o_p)

        sequence_output = self.add_weight(o_s, o_w)

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.fc(concat_h)

        outputs = (logits,)


class LanguageHierarchicalNER(BertPreTrainedModel):
    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_labels
        self.num_labels = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageHierarchicalNER, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


