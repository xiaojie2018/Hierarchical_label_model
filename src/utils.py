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

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

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
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

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

    def load_tokenizer(self, args):
        if args.model_type in ["albert", "roberta"]:
            class CNerTokenizer(BertTokenizer):
                def __init__(self, vocab_file, do_lower_case=False):
                    super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
                    self.vocab_file = str(vocab_file)
                    self.do_lower_case = do_lower_case
                    self.vocab = load_vocab(vocab_file)

                def tokenize(self, text):
                    _tokens = []
                    for c in text:
                        if self.do_lower_case:
                            c = c.lower()
                        if c in self.vocab:
                            _tokens.append(c)
                        else:
                            _tokens.append('[UNK]')
                    return _tokens
        else:
            class CNerTokenizer(MODEL_CLASSES[args.model_type][1]):
                def __init__(self, vocab_file, do_lower_case=False):
                    super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
                    self.vocab_file = str(vocab_file)
                    self.do_lower_case = do_lower_case
                    self.vocab = load_vocab(vocab_file)

                def tokenize(self, text):
                    _tokens = []
                    for c in text:
                        if self.do_lower_case:
                            c = c.lower()
                        if c in self.vocab:
                            _tokens.append(c)
                        else:
                            _tokens.append('[UNK]')
                    return _tokens

        tokenizer = CNerTokenizer.from_pretrained(args.model_name_or_path)

        return tokenizer

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer,
                                     cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                     sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                     sequence_a_segment_id=0, mask_padding_with_zero=True, ):
        pass

    def _get_data(self):

        pass

