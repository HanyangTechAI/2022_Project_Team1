import torch
import argparse
import os
import pandas as pd

from train import train, predict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification

def generate_data_loader(file_path, tokenizer, args):
    def get_input_ids(data):
        document_bert = ["[CLS] " + str(s) + " [SEP]" for s in data]
        tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=args.maxlen, dtype='long', truncating='post', padding='post')
        return input_ids

    def get_attention_masks(input_ids):
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    def get_data_loader(inputs, masks, labels, batch_size=args.batch):
        data = TensorDataset(torch.tensor(inputs), torch.tensor(masks), torch.tensor(labels))
        sampler = RandomSampler(data) if args.mode == 'train' else SequentialSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return data_loader

    data_df = pd.read_csv(file_path)
    input_ids = get_input_ids(data_df['contents'].values)
    attention_masks = get_attention_masks(input_ids)
    data_loader = get_data_loader(input_ids, attention_masks, data_df['label'].values if args.mode=='train' else [-1]*len(data_df))

    return data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data_preprocessing/train")
    parser.add_argument("--valid_path", type=str, default="../data_preprocessing/valid")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model_ckpt", type=str, default="bert-base-multilingual-cased")
    
    args = parser.parse_args()

    # model load
    model = BertForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=2)
    model.to(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.model_ckpt, do_lower_case=False)
    train_dataloader = generate_data_loader(args.train_path, tokenizer, args)
    validation_dataloader = generate_data_loader(args.valid_path, tokenizer, args)
    train(model, args, train_dataloader, validation_dataloader)
