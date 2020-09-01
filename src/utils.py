# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/24 16:17
# software: PyCharm
import copy
import json
import logging
import random
import numpy as np
import torch
from transformers import BertTokenizer
import collections
from config import MODEL_CLASSES
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) string. The intent label of the example.
    """

    def __init__(self, guid, text, entities, o_label=None, m_label=None):
        self.guid = guid
        self.text = text
        self.entities = entities
        self.o_label = o_label
        self.m_label = m_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, o_label, m_label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.o_label = o_label
        self.m_label = m_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataPreprocess:

    def __init__(self, config):
        self.config = config

        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>", "[UNK]"]
        self.tokenizer = self.load_tokenizer(self.config)

    # def load_tokenizer(self, args):
    #     if args.model_type in ["albert", "roberta"]:
    #         class CNerTokenizer(BertTokenizer):
    #             def __init__(self, vocab_file, do_lower_case=False):
    #                 super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
    #                 self.vocab_file = str(vocab_file)
    #                 self.do_lower_case = do_lower_case
    #                 self.vocab = load_vocab(vocab_file)
    #
    #             def tokenize(self, text):
    #                 _tokens = []
    #                 for c in text:
    #                     if self.do_lower_case:
    #                         c = c.lower()
    #                     if c in self.vocab:
    #                         _tokens.append(c)
    #                     else:
    #                         _tokens.append('[UNK]')
    #                 return _tokens
    #     else:
    #         class CNerTokenizer(MODEL_CLASSES[args.model_type][1]):
    #             def __init__(self, vocab_file, do_lower_case=False):
    #                 super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
    #                 self.vocab_file = str(vocab_file)
    #                 self.do_lower_case = do_lower_case
    #                 self.vocab = load_vocab(vocab_file)
    #
    #             def tokenize(self, text):
    #                 _tokens = []
    #                 for c in text:
    #                     if self.do_lower_case:
    #                         c = c.lower()
    #                     if c in self.vocab:
    #                         _tokens.append(c)
    #                     else:
    #                         _tokens.append('[UNK]')
    #                 return _tokens
    #
    #     tokenizer = CNerTokenizer.from_pretrained(args.model_name_or_path)
    #
    #     return tokenizer
    
    def load_tokenizer(self, args):
        if args.model_type in ["albert", "roberta"]:
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
            return tokenizer
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

    def convert_examples_to_features(self, examples, max_seq_len, tokenizer, cls_token_at_end=False,
                                     cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]", pad_on_left=False,
                                     pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0,
                                     mask_padding_with_zero=True, ):

        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        # unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            entity1, entity2 = example.entities[0], example.entities[1]
            text = example.text
            if entity2['start_pos'] >= entity1['end_pos']:
                text = text[:entity1['start_pos']] + "<e1>" + text[entity1['start_pos']: entity1['end_pos']] + "</e1>" + text[entity1['end_pos']:]
                text = text[:entity2['start_pos']+9] + "<e2>" + text[entity2['start_pos']+9: entity2['end_pos']+9] + "</e2>" + text[entity2['end_pos']+9:]
            elif entity1['start_pos'] >= entity2['end_pos']:
                text = text[:entity2['start_pos']] + "<e2>" + text[entity2['start_pos']: entity2['end_pos']] + "</e2>" + text[entity2['end_pos']:]
                text = text[:entity1['start_pos'] + 9] + "<e1>" + text[entity1['start_pos'] + 9: entity1['end_pos'] + 9] + "</e1>" + text[entity1['end_pos'] + 9:]
            else:
                continue

            tokens = tokenizer.tokenize(text)

            e11_p = tokens.index("<e1>")
            e12_p = tokens.index("</e1>")

            e21_p = tokens.index("<e2>")
            e22_p = tokens.index("</e2>")

            e1_mask = [0]*len(tokens)
            e2_mask = [0]*len(tokens)

            for i in range(e11_p, e12_p):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p):
                e2_mask[i] = 1

            # 去掉 <e1> </e1> <e2> </e2>
            tokens1, e1_mask1, e2_mask1 = [], [], []
            for i in range(len(tokens)):
                if i in [e11_p, e12_p, e21_p, e22_p]:
                    continue
                tokens1.append(tokens[i])
                e1_mask1.append(e1_mask[i])
                e2_mask1.append(e2_mask[i])

            token_type_ids = [sequence_a_segment_id] * len(tokens1)

            # Add [CLS] token
            tokens1 = [cls_token] + tokens1
            token_type_ids = [cls_token_segment_id] + token_type_ids
            e1_mask1 = [0] + e1_mask1
            e2_mask1 = [0] + e2_mask1

            # Add [SEP] token
            tokens1 += [sep_token]
            token_type_ids += [sequence_a_segment_id]
            e1_mask1 += [0]
            e2_mask1 += [0]

            input_ids = tokenizer.convert_tokens_to_ids(tokens1)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)

            if padding_length > 0:

                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                e1_mask1 = e1_mask1 + ([0 if mask_padding_with_zero else 1] * padding_length)
                e2_mask1 = e2_mask1 + ([0 if mask_padding_with_zero else 1] * padding_length)

            elif padding_length < 0:
                input_ids = input_ids[:max_seq_len - 1] + [input_ids[-1]]
                attention_mask = attention_mask[:max_seq_len - 1] + [attention_mask[-1]]
                token_type_ids = token_type_ids[:max_seq_len - 1] + [token_type_ids[-1]]
                e1_mask1 = e1_mask1[:max_seq_len - 1] + [e1_mask1[-1]]
                e2_mask1 = e2_mask1[:max_seq_len - 1] + [e2_mask1[-1]]

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)

            assert len(e1_mask1) == max_seq_len, "Error with entity1 mask length {} vs {}".format(
                len(e1_mask1), max_seq_len)
            assert len(e2_mask1) == max_seq_len, "Error with entity1 mask length {} vs {}".format(
                len(e2_mask1), max_seq_len)

            o_label = example.o_label

            m_label = example.m_label

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens1]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("e1_mask: %s " % " ".join([str(x) for x in e1_mask1]))
                logger.info("e2_mask: %s " % " ".join([str(x) for x in e2_mask1]))
                logger.info("o_label: %s " % " ".join([str(x) for x in o_label]))
                logger.info("m_label: %s " % " ".join([str(x) for x in m_label]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              e1_mask=e1_mask1,
                              e2_mask=e2_mask1,
                              o_label=o_label,
                              m_label=m_label
                              ))

        return features

    def _get_data(self, data, o_labels=None, m_labels=None, set_type="train"):

        if set_type == 'train':
            random.shuffle(data)

        len_o_label = len(o_labels)
        len_m_label = len(m_labels)

        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text = d['text']
            entities = d['entities']

            entity1, entity2 = entities[0], entities[1]

            if entity2['start_pos'] >= entity1['end_pos']:
                pass
            elif entity1['start_pos'] >= entity2['end_pos']:
                pass
            else:
                continue

            o_label = [0.0] * len_o_label
            if d['o_label'] == "NEG_TEXT":
                o_label = [1.0 / len_o_label] * len_o_label
            else:
                o_label[o_labels.index(d['o_label'])] = 1.0

            m_label = [0.0] * len_m_label
            if d['m_label'] == "NEG_TEXT":
                m_label = [1.0 / len_m_label] * len_m_label
            else:
                m_label[m_labels.index(d['m_label'])] = 1.0

            examples.append(InputExample(guid=guid, text=text, entities=entities, o_label=o_label, m_label=m_label))

        features = self.convert_examples_to_features(examples=examples,
                                                     max_seq_len=self.config.max_seq_len,
                                                     tokenizer=self.tokenizer,
                                                     cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                                                     pad_on_left=bool(self.config.model_type in ['xlnet']),
                                                     cls_token=self.tokenizer.cls_token,
                                                     cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
                                                     sep_token=self.tokenizer.sep_token,
                                                     # pad on the left for xlnet
                                                     pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                     pad_token_segment_id=4 if self.config.model_type in ['xlnet'] else 0,)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e1 mask
        all_o_label_ids = torch.tensor([f.o_label for f in features], dtype=torch.float32)
        all_m_label_ids = torch.tensor([f.m_label for f in features], dtype=torch.float32)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask,
                                all_o_label_ids, all_m_label_ids)

        return dataset, examples

    def get_data(self, file):

        data = []
        o_labe = set()
        m_label = set()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = eval(line)
                o_labe.add(line['o_label'])
                m_label.add(line['m_label'])
                data.append(line)

        return data, list(o_labe), list(m_label)
