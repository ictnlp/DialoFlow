# coding: utf-8

'''
  @Date    : 2021-05-05
  @Author  : Zekang Li
  @Mail    : zekangli97@gmail.com
  @Homepage: zekangli.com
'''


import torch
import string
import torch.nn as nn
from transformers import *

SPECIAL_TOKENS = ["<bos>", "<eos>", "<info>", "<speaker1>", "<speaker2>", "<empty>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<info>", "<speaker1>", "<speaker2>", "<empty>"], 'pad_token': "<pad>"}


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


def build_input_from_dialogue(conv, tokenizer):
    bos, eos, info, speaker1, speaker2, empty, pad = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)]
    sentence_index = [0]
    conv_seq = []
    label_seq = []
    token_type_seq = []
    for i in range(len(conv)):
        if len(conv_seq) + len(conv[i][:128]) + 10 > 1000:
            break
        if i % 2 == 0:
            speaker = 0
            conv_seq.append(speaker1)
        else:
            speaker = 1
            conv_seq.append(speaker2)
        label_seq.append(-100)
        token_type_seq.append(speaker)

        conv_seq.extend(conv[i][:128])
        label_seq.extend(conv[i][:128])
        token_type_seq.extend([speaker]*len(conv[i][:128]))

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


def norm(a):
    mean = torch.mean(a, dim=1).unsqueeze(1)
    std = torch.std(a, dim=1).unsqueeze(1)
    return (a - mean) / std


def add_punc(text):
    r = ""
    for i in text:
        if i in string.punctuation:
            r += " "
            r += i
        else:
            r += i
    return r.replace("  ", " ")


class FlowScore:
    def __init__(self, model_path):
        self.cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(model_path, map_location=self.cuda)
        self.model.planing_model.device = self.cuda
        self.tokenizer = GPT2Tokenizer.from_pretrained("./models/")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        self.model.eval()
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-9)

    def score(self, dialogue):
        with torch.no_grad():
            conv_seq, _, sentence_index, token_type_seq = build_input_from_dialogue(tokenize(dialogue, self.tokenizer),
                                                                                            self.tokenizer)
            conv_seq = conv_seq.unsqueeze(0).cuda()
            sentence_index = sentence_index.unsqueeze(0).cuda()
            token_type_seq = token_type_seq.unsqueeze(0).cuda()
            conv_hidden_state = self.model.speak_model(conv_seq, token_type_ids=token_type_seq)[0]
            sentence_hidden = conv_hidden_state.index_select(1, sentence_index[0])
            output, loss_plan = self.model.planing_model(sentence_hidden)
            sentence_hidden_delta = sentence_hidden[:, 1:, :] - sentence_hidden[:, :-1, :]
            output_delta = output[:, :-1, :] - sentence_hidden[:, :-1, :]
            sentence_hidden_len = torch.sqrt(torch.sum(sentence_hidden_delta ** 2, dim=-1))
            output_len = torch.sqrt(torch.sum(output_delta ** 2, dim=-1))
            min_len = torch.min(sentence_hidden_len, output_len)
            x = self.cos(sentence_hidden_delta, output_delta) * (min_len ** 2 / (sentence_hidden_len * output_len))
            DPKS_score = torch.pow(2, -torch.mean(torch.log(((x + 1) / 2)[0, 3::2])))
        return DPKS_score.item()



