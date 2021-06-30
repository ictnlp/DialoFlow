import torch
import copy
import json
import numpy as np
import torch.nn as nn
from transformers import *
import torch.nn.functional as F
import random

SPECIAL_TOKENS = ["<bos>", "<eos>", "<info>", "<speaker1>", "<speaker2>", "<empty>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<info>", "<speaker1>", "<speaker2>", "<empty>"], 'pad_token': "<pad>"}


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def build_input_from_input(conv, current_output, tokenizer):
    bos, eos, info, speaker1, speaker2, empty, pad = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)]
    sentence_index = [0]
    conv_seq = []
    temp_conv = []
    temp_len = 0
    token_type_seq = []
    for i in range(len(conv)):
        if temp_len + len(conv[i]) < 1000:
            temp_conv.append(conv[i])
            temp_len += len(conv[i])
        else:
            while temp_len + len(conv[i]) >= 1000:
                a = len(temp_conv[0])
                temp_conv = temp_conv[1:]
                temp_len -= a
            temp_conv.append(conv[i])
            temp_len += len(conv[i])

    for i in range(len(temp_conv)):
        if i % 2 == 0:
            speaker = 0
            conv_seq.append(speaker1)
        else:
            speaker = 1
            conv_seq.append(speaker2)
        token_type_seq.append(speaker)
        conv_seq.extend(temp_conv[i][:64])
        token_type_seq.extend([speaker] * len(temp_conv[i][:64]))
        conv_seq.append(eos)
        token_type_seq.append(speaker)
        conv_seq.append(empty)
        token_type_seq.append(speaker)
        sentence_index.append(len(conv_seq)-1)

    conv_seq = torch.LongTensor(conv_seq).unsqueeze(0)
    conv_seq = conv_seq.expand(args.batch_size, -1)
    conv_seq = torch.cat([conv_seq, current_output], dim=-1)
    token_type_seq.extend([len(temp_conv) % 2] * current_output.size(1))
    token_type_seq = torch.LongTensor(token_type_seq).unsqueeze(0).expand(args.batch_size, -1)
    sentence_index = torch.LongTensor(sentence_index).unsqueeze(0).expand(args.batch_size, -1)
    return conv_seq, sentence_index, token_type_seq


def work_delta(model, conv_seq, sentence_idx, token_type_seq):
    conv_hidden_state = model.speak_model(conv_seq, token_type_ids=token_type_seq)[0]
    sentence_hidden = conv_hidden_state.index_select(1, sentence_idx[0])
    output, _  = model.planing_model(sentence_hidden)
    delta = output[:, -1, :] - sentence_hidden[:, -1, :]
    return delta

def work(model, conv_seq, past_key_values, delta, token_type_seq, args):
    if past_key_values is not None:
        conv_hidden_state, past_key_values = model.speak_model(conv_seq, token_type_ids=token_type_seq, use_cache=True, past=past_key_values)[:2]
    else:
        conv_hidden_state, past_key_values = model.speak_model(conv_seq, token_type_ids=token_type_seq, use_cache=True)[:2]
    temp_conv_hidden = conv_hidden_state[:, -1:, :]
    lm_logits = model.lm_head(torch.cat([delta.unsqueeze(1), temp_conv_hidden], dim=-1))
    return lm_logits, past_key_values

def standard(x):
    return [i.replace(".", ". ").replace("  ", " ") for i in x]

def sample_sequence(conv, tokenizer, model, args):
    bos, eos, info, speaker1, speaker2, empty, pad = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)]
    special_tokens_ids = [bos, eos, info, speaker1, speaker2, empty, pad]
    conv = standard(conv)
    conv = tokenize(conv, tokenizer)
    if len(conv) % 2 == 1:
        current_output = torch.LongTensor([[speaker2]] * args.batch_size)
        final_output = torch.LongTensor([[speaker2]] * args.batch_size)
    else:
        current_output = torch.LongTensor([[speaker1]] * args.batch_size)
        final_output = torch.LongTensor([[speaker1]] * args.batch_size)
    conv_seq, sentence_idx, token_type_seq = build_input_from_input(conv, current_output, tokenizer)
    conv_seq = conv_seq.to(args.device)
    token_type_seq = token_type_seq.to(args.device)
    sentence_idx = sentence_idx.to(args.device)
    delta = work_delta(model, conv_seq, sentence_idx, token_type_seq)
    past_key_values = None
    for i in range(args.max_length):
        if past_key_values is not None:
            conv_seq = current_output.cuda()
            token_type_seq = token_type_seq[:, -1:]
        lm_logits, past_key_values = work(model, conv_seq, past_key_values, delta, token_type_seq, args)
        lm_logits = lm_logits.squeeze()
        lm_logits = lm_logits / args.temperature
        if i < args.min_length:
            lm_logits[:, eos] = -1e9
        lm_logits = torch.cat([top_filtering(lm_logits[j].squeeze(), top_k=args.top_k, top_p=args.top_p).unsqueeze(0) for j in range(args.batch_size)], dim=0)
        probs = F.softmax(lm_logits, dim=-1)
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        current_output = prev
        final_output = torch.cat([final_output, prev.cpu()], dim=-1)
    decode_result = []
    for i in range(0, args.batch_size):
        temp = final_output[i, 1:].cpu().tolist()
        temp1 = []
        for j in temp:
            if j in special_tokens_ids:
                break
            temp1.append(j)
        decode_result.append((tokenizer.decode(temp1, skip_special_tokens=True) + "\n").replace("1.0 ", "").replace("0.0 ", ""))
    return decode_result

def build_input_from_input_beam(conv, current_output, tokenizer):
    bos, eos, info, speaker1, speaker2, empty, pad = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)]
    sentence_index = [0]
    conv_seq = []
    temp_conv = []
    temp_len = 0
    token_type_seq = []
    for i in range(len(conv)):
        if temp_len + len(conv[i]) < 1000:
            temp_conv.append(conv[i])
            temp_len += len(conv[i])
        else:
            while temp_len + len(conv[i]) >= 1000:
                a = len(temp_conv[0])
                temp_conv = temp_conv[1:]
                temp_len -= a
            temp_conv.append(conv[i])
            temp_len += len(conv[i])

    for i in range(len(temp_conv)):
        if i % 2 == 0:
            speaker = 0
            conv_seq.append(speaker1)
        else:
            speaker = 1
            conv_seq.append(speaker2)
        token_type_seq.append(speaker)
        conv_seq.extend(temp_conv[i][:128])
        token_type_seq.extend([speaker] * len(temp_conv[i][:128]))
        conv_seq.append(eos)
        token_type_seq.append(speaker)
        conv_seq.append(empty)
        token_type_seq.append(speaker)
        sentence_index.append(len(conv_seq)-1)
    conv_seq.extend(current_output)
    token_type_seq.extend([len(temp_conv) % 2] * len(current_output))

    conv_seq = torch.LongTensor(conv_seq).unsqueeze(0)
    token_type_seq = torch.LongTensor(token_type_seq).unsqueeze(0)
    sentence_index = torch.LongTensor(sentence_index).unsqueeze(0)
    return conv_seq, sentence_index, token_type_seq

def beam_search(src, tokenizer, model, args):
    bos, eos, info, speaker1, speaker2, empty, pad = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)]
    special_tokens_ids = [bos, eos, info, speaker1, speaker2, empty, pad]
    current_output = []
    
    conv = tokenize(src, tokenizer)
    if len(conv) % 2 == 1:
        current_output = [speaker2]
    else:
        current_output = [speaker1]
    hyplist = [([], 0., current_output)]
    best_state = None
    comp_hyplist = []

    conv_seq, sentence_idx, token_type_seq = build_input_from_input_beam(conv, current_output, tokenizer)
    conv_seq = conv_seq.to(args.device)
    token_type_seq = token_type_seq.to(args.device)
    sentence_idx = sentence_idx.to(args.device)
    delta = work_delta(model, conv_seq, sentence_idx, token_type_seq)

    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            input_ids, _, token_type_seq  = build_input_from_input_beam(conv, st, tokenizer)
            input_ids = input_ids.to(args.device)
            token_type_seq = token_type_seq.to(args.device)
            conv_hidden_state = model.speak_model(input_ids, token_type_ids=token_type_seq)[0]
            temp_conv_hidden = conv_hidden_state[:, -1:, :]
            lm_logits = model.lm_head(torch.cat([delta.unsqueeze(1), temp_conv_hidden], dim=-1))
            logp = F.log_softmax(lm_logits, dim=-1)[:, -1, :]
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
                            
            if i >= args.min_length:
                new_lp = lp_vec[eos] + args.penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o in special_tokens_ids:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return [tokenizer.decode(maxhyps[0][0], skip_special_tokens=True).replace("\n", "") + "\n"]*2
    else:
        return [([], 0)]


class Config():
    def __init__(self):
        self.max_length = 40
        self.device = "cuda"
        self.top_k = 40
        self.top_p = 0
        self.min_length = 11
        self.no_sample = False
        self.temperature = 0.9
        self.model_checkpoint = "log/"
        self.batch_size = 2
        self.beam_size = 10
        self.penalty = 0.1

args = Config()
model = torch.load("models/DialoFlow_large/model.bin")
model.cuda()
model.eval()
print("finish loading model")
tokenizer = GPT2Tokenizer.from_pretrained("models/DialoFlow_large")
tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
with open("test.refs.txt") as f:
    data = f.readlines()

if_random = False
data_results = []
for i in data:
    temp = i.split("\t")
    history = temp[0].split(" EOS ")
    responses = temp[1:]
    hypstr = beam_search(history, tokenizer, model, args)
    #hypstr = sample_sequence(history, tokenizer, model, args)
    with open("DialoFlow_results_large.txt", "a+", encoding="utf-8") as f:
        f.writelines(hypstr[0])

