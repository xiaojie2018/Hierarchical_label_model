# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/24 16:17
# software: PyCharm
import torch

from config import MODEL_TASK


class Trainer:

    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None,
                 train_examples=None, test_examples=None, dev_examples=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.dev_examples = dev_examples

        self.model_class = MODEL_TASK[args.task_name]

        self.model = self.model_class(args.model_dir, args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        self.args.n_gpu = torch.cuda.device_count()
        self.args.n_gpu = 1


