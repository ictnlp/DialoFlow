import numpy as np
import torch
from torch.utils.data import Dataset 
from itertools import chain
import json
import torch.nn.functional as F

SPECIAL_TOKENS = ["<bos>", "<eos>", "<info>", "<speaker1>", "<speaker2>", "<empty>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<info>", "<speaker1>", "<speaker2>", "<empty>"], 'pad_token': "<pad>"}


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


class DialoFlowDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path)  as f:
            self.data = [json.loads(i.strip()) for i in f.readlines()]
        self.conv = [[j["text"] for j in i["dialogue"]] for i in self.data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.conv)

    def __getitem__(self, index):
        conv = self.conv[index]
        conv = tokenize(conv, self.tokenizer)
        conv_seq, label_seq, sentence_index, token_type_seq = build_input_from_dialogue(conv, self.tokenizer)
        return conv_seq, label_seq, sentence_index, token_type_seq


def build_input_from_dialogue(conv, tokenizer):
    bos, eos, info, speaker1, speaker2, empty, pad = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)]
    sentence_index = [0]
    conv_seq = []
    label_seq = []
    token_type_seq = []
    for i in range(len(conv)):
        if len(conv_seq) + len(conv[i][:64]) + 10 > 600:
            break
        if i % 2 == 0:
            speaker = 0
            conv_seq.append(speaker1)
        else:
            speaker = 1
            conv_seq.append(speaker2)
        label_seq.append(-100)
        token_type_seq.append(speaker)

        conv_seq.extend(conv[i][:64])
        label_seq.extend(conv[i][:64])
        token_type_seq.extend([speaker]*len(conv[i][:64]))

        conv_seq.append(eos)
        label_seq.append(eos)
        token_type_seq.append(speaker)

        conv_seq.append(empty)
        label_seq.append(-100)
        token_type_seq.append(speaker)
        
        sentence_index.append(len(conv_seq)-1)
        
    conv_seq = torch.LongTensor(conv_seq)
    label_seq = torch.LongTensor(label_seq)
    sentence_index = torch.LongTensor(sentence_index)
    token_type_seq = torch.LongTensor(token_type_seq)
    return conv_seq, label_seq, sentence_index, token_type_seq

def collate_fn(batch, pad_token):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        result = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result
    
    input_ids_list = []
    token_type_ids_list = []
    lm_labels_list = []
    sentence_idx_list = []
    for input_ids, lm_labels, sentence_idx, token_type_ids in batch:
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        lm_labels_list.append(lm_labels)
        sentence_idx_list.append(sentence_idx)
    input_ids = padding(input_ids_list, pad_token)
    token_type_ids = padding(token_type_ids_list, pad_token)
    lm_labels = padding(lm_labels_list, -100)
    input_mask = input_ids != pad_token
    return input_ids, lm_labels, sentence_idx_list, token_type_ids, input_mask.float()
