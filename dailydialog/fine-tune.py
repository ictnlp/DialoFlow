# coding: utf-8

import os
import math
import numpy as np
import logging
from pprint import pformat
from argparse import ArgumentParser
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from dataset import DialoFlowDataset, collate_fn
import random
from transformers import *
from tqdm import tqdm
from model import *
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel


SPECIAL_TOKENS = ["<bos>", "<eos>", "<info>", "<speaker1>", "<speaker2>", "<empty>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<info>", "<speaker1>", "<speaker2>", "<empty>"], 'pad_token': "<pad>"}

logger = logging.getLogger(__file__)

def load_model(model_path, cuda):
    model = torch.load(os.path.join(model_path, "model.bin"), map_location=cuda)
    model.planing_model.device = cuda
    return model


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    iteration = 0
    loss_cum = 0
    with tqdm(enumerate(train_loader), total=len(train_loader)) as t:
        for idx, batch in t:
            iteration += 1
            conv_seq, label_seq, sentence_index, token_type_seq, input_mask = batch
            conv_seq, label_seq, sentence_index, token_type_seq, input_mask = conv_seq.to(args.device), label_seq.to(args.device), [s.to(args.device) for s in sentence_index], token_type_seq.to(args.device), input_mask.to(args.device)
            loss, loss_gen, loss_plan, loss_bow = model(conv_seq, label_seq, sentence_index, token_type_seq, input_mask)
            loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if iteration % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
                optimizer.zero_grad()
            loss_cum += loss.item()
            t.set_description('Epoch %i' % epoch)
            t.set_postfix(loss=loss_cum / (idx+1))


def eval(args, model, valid_loader, epoch):
    model.eval()
    loss_cum_speak = 0
    loss_cum_plan = 0
    loss_cum_bow = 0
    with torch.no_grad():
        with tqdm(enumerate(valid_loader), total=len(valid_loader)) as t:
            for idx, batch in t:
                conv_seq, label_seq, sentence_index, token_type_seq, input_mask = batch
                conv_seq, label_seq, sentence_index, token_type_seq, input_mask = conv_seq.to(args.device), label_seq.to(args.device), [s.to(args.device) for s in sentence_index], token_type_seq.to(args.device), input_mask.to(args.device)
                _, speak_loss, plan_loss, bow_loss = model(conv_seq, label_seq, sentence_index, token_type_seq, input_mask)
                loss_cum_speak += speak_loss.item()
                loss_cum_plan += plan_loss.item()
                loss_cum_bow += bow_loss.item()

    loss_cum_speak = loss_cum_speak / len(valid_loader)
    loss_cum_plan = loss_cum_plan / len(valid_loader)
    loss_cum_bow = loss_cum_bow / len(valid_loader)
    logger.info("Validation loss at epoch {}: {}, {}, {}".format(epoch, loss_cum_speak, loss_cum_plan, loss_cum_bow))


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/", help="Path of the dataset")
    parser.add_argument("--model_checkpoint", type=str, default="../models/DialoFlow_base/", help="Path of the dataset")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=int, default=1,
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--log_path", type=str, default="log/", help="Log file path")
    parser.add_argument("--logger_path", type=str, default="logger/", help="Logger file path")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)

    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model = load_model(args.model_checkpoint, args.device)
    if args.fp16:
        model = convert_syncbn_model(model)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if args.local_rank != -1:
        if args.fp16:
            model = DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    logger.info("Prepare datasets")
    train_dataset = DialoFlowDataset(args.data_path + "train.json", tokenizer)
    valid_dataset = DialoFlowDataset(args.data_path + "valid.json", tokenizer)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset) if args.local_rank == -1 else torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=8, sampler=train_sampler, collate_fn=lambda x: collate_fn(x, tokenizer.convert_tokens_to_ids("<pad>")))
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=8, sampler=valid_sampler, collate_fn=lambda x: collate_fn(x, tokenizer.convert_tokens_to_ids("<pad>")))
    logger.info("Finish preparing datasets")

    logger.info("Start Training...")
    for epoch in range(args.n_epochs):
        train(args, model, train_loader, optimizer, epoch)
        eval(args, model, valid_loader, epoch)
        if args.local_rank == 0 and epoch % 5 == 4:
            tokenizer.save_pretrained(args.log_path)
            torch.save(args, os.path.join(args.log_path, "training_args.bin"))
            torch.save(getattr(model, 'module', model), os.path.join(args.log_path, "DialoFlow" + str(epoch) + ".bin"))
            getattr(model, 'module', model).speak_model.config.save_pretrained(args.log_path)
            torch.save(optimizer.state_dict(), os.path.join(args.log_path, "optimizer.pkl"))


if __name__ == "__main__":
    main()
