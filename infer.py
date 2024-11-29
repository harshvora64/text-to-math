# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchtext
import json
from transformers import BertTokenizer, BertModel
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dir = './'
batch_size = 32
num_epochs = 60
embedding_dim = 300
hidden_dim = 256
lr = 0.001

models = {}
models['lstm_lstm'] = 0
models['lstm_lstm_attn'] = 1
models['bert_lstm_attn_frozen'] = 2
models['bert_lstm_attn_tuned'] = 3

# %%
GloVe = torchtext.vocab.GloVe(name='6B', dim=embedding_dim)
bert_vocab = BertTokenizer.from_pretrained('bert-base-uncased')

BERT_frozen = BertModel.from_pretrained('bert-base-uncased')
BERT_fine_tuned = BertModel.from_pretrained('bert-base-uncased')

for param in BERT_frozen.parameters():
    param.requires_grad = False

for param in BERT_fine_tuned.parameters():
    param.requires_grad = False

for param in BERT_fine_tuned.encoder.layer[-1].parameters():
    param.requires_grad = True

vocab = json.load(open(dir + 'var/glove_vocab.json', 'r'))
out_vocab = json.load(open(dir + 'var/glove_out_vocab.json', 'r'))
rev_out_vocab = json.load(open(dir + 'var/glove_rev_out_vocab.json', 'r'))
rev_out_vocab = {int(k):v for k,v in rev_out_vocab.items()}
embedding = torch.load(dir + 'var/glove_embedding.pth')

# %%
# Load data from test.json, train.json, dev.json using DataLoader

class seqDataset(Dataset):
    def __init__(self, path, mode = "", in_vocab = {}, out_vocab = {}):
        self.path = path
        self.mode = mode
        self.Problem = []
        self.ans = 0
        self.load_data()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def load_data(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                x = (data[i]['Problem'] + ' <eos>').split()
                n = 0
                for j in range(len(x)):
                    try:
                        float(x[j])
                    except:
                        continue
                    x[j] = 'n' + str(n)
                    n += 1
                self.Problem.append(x)

    def __len__(self):
        return len(self.Problem)

    def __getitem__(self, idx):
        return (self.Problem[idx]), (self.ans)
    
    def print(self, idx):
        print(' '.join(self.Problem[idx]), '|'.join(self.linear_formula[idx]), self.answer[idx], sep='\n')
    

def collate_fn(batch):
    Problem, ans = zip(*batch)
    Problem = pad_sequence(Problem, batch_first=True, padding_value=0)
    return Problem

def bert_collate(batch):
    Problem, ans = zip(*batch)
    Problem = [' '.join(sentence[:-1]) for sentence in Problem]
    Problem = bert_vocab(Problem, padding=True, return_tensors='pt')
    return Problem

def make_glove_embedding(model, model_vocab, train_sentences):
    vocab = {'<pad>':0, '<start>':1, '<eos>':2, '<unknown>':3}
    i = 4
    for sentence in train_sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = i
                i += 1

    embedding = torch.FloatTensor(len(vocab), embedding_dim).uniform_(-0.25, 0.25)

    for word in vocab:
        if word in model_vocab:
            embedding[vocab[word]] = model.vectors[model_vocab[word]]

    return embedding, vocab

def make_output_vocab(train_out_sentences):
    vocab = {'<pad>':0, '<start>':1, '<eos>':2, '<unknown>':3}
    rev_vocab = {0:'<pad>', 1:'<start>', 2:'<eos>', 3:'<unknown>'}
    i = 4
    for sentence in train_out_sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = i
                rev_vocab[i] = word
                i += 1
    
    return vocab, rev_vocab

def modify_dataset(vocab, data):    
    for i in range(len(data)):
        data[i] = torch.tensor([vocab[word] if word in vocab else vocab['<unknown>'] for word in data[i]])
    return data

class beamdata:
    def __init__(self, score, sequence, hidden, cell):
        self.score = score
        self.sequence = sequence
        self.hidden = hidden
        self.cell = cell

# %%

# %%
class Encoder(nn.Module):
    def __init__(self, embedding = embedding, hidden_dim = hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.linear_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.linear_c = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, (h, c) = self.lstm(x)
        h = (self.linear_h(h.permute(1, 0, 2).reshape(x.size(0), -1)))
        c = (self.linear_c(c.permute(1, 0, 2).reshape(x.size(0), -1)))
        return x, h, c


class Decoder(nn.Module):
    def __init__(self, out_vocab = out_vocab, hidden_dim = hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(len(out_vocab), embedding_dim, padding_idx=0)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, len(out_vocab))

    def forward(self, x, h, c):
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        x = self.dropout(self.embedding(x))
        x = x.unsqueeze(1)
        x, (h, c) = self.lstm(x, (h, c))
        x = self.linear(x.squeeze(1))
        h = h.squeeze(0)
        c = c.squeeze(0)
        return x, h, c

class seq2seq(nn.Module):
    def __init__(self, teacher_forcing_ratio = 0.6, beam_size = 10, out_vocab = out_vocab):
        super(seq2seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        self.out_vocab = out_vocab

    def forward(self, x, y):
        if(self.training):
            return self.train_forward(x, y)
        else:
            return self.test_forward_beam(x, y)
        
    def train_forward(self, x, y):
        _, h, c = self.encoder(x)
        out = out_vocab['<start>']*torch.ones(x.size(0), dtype = torch.long).to(device)
        outputs = torch.zeros(y.size(0), y.size(1), len(out_vocab)).to(device)
        for i in range(y.size(1)):
            if(np.random.random() < self.teacher_forcing_ratio):
                if(i != 0):
                    out = y[:, i-1]
            else:
                if(i != 0):
                    out = torch.argmax(out, dim=1)
            out, h, c = self.decoder(out, h, c)
            outputs[:, i] = out
        return outputs
    
    def test_forward_beam(self, x, y):
        _, h, c = self.encoder(x)
        out = out_vocab['<start>']*torch.ones(x.size(0), dtype = torch.long).to(device)
        beam = [beamdata(0, [out], h, c)]
        for i in range(y.size(1)):
            new_beam = []
            for b in beam:
                out, h, c = self.decoder(torch.tensor(b.sequence[-1]), b.hidden, b.cell)
                out = F.log_softmax(out, dim=1)
                out = torch.topk(out, self.beam_size, dim=1)
                for j in range(self.beam_size):
                    new_beam.append(beamdata(b.score + out[0][:,j], b.sequence + [out[1][:,j]], h, c))

            beam = sorted(new_beam, key=lambda x: x.score.sum(), reverse=True)[:self.beam_size]

        sequence = torch.zeros(y.size(0), y.size(1), dtype=torch.long).to(device)
        for i in range(y.size(1)):
            sequence[:, i] = torch.tensor(beam[0].sequence[i])

        return sequence

# %%
# MODEL 2
# AttnSeq2seq

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden.unsqueeze(1)
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class AttnEncoder(nn.Module):
    def __init__(self, embedding = embedding, hidden_dim = hidden_dim):
        super(AttnEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.linear_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.linear_c = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, (h, c) = self.lstm(x)
        h = torch.tanh(self.linear_h(h.permute(1, 0, 2).reshape(x.size(0), -1)))
        c = torch.tanh(self.linear_c(c.permute(1, 0, 2).reshape(x.size(0), -1)))
        return x, h, c

class AttnDecoder(nn.Module):
    def __init__(self, out_vocab = out_vocab, hidden_dim = hidden_dim):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(len(out_vocab), embedding_dim, padding_idx=0)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_dim + 2*self.hidden_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, len(out_vocab))
        self.attention = Attention(self.hidden_dim)

    def forward(self, x, h, c, encoder_outputs):
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))
        attention_weights = self.attention(encoder_outputs, h)
        attention_weights = attention_weights.unsqueeze(1)
        context = attention_weights.bmm(encoder_outputs)
        x = torch.cat((x, context), dim=2)
        x, (h, c) = self.lstm(x, (h.unsqueeze(0), c.unsqueeze(0)))
        x = self.linear(x.squeeze(1))
        h = h.squeeze(0)
        c = c.squeeze(0)
        return x, h, c
    

class AttnSeq2seq(nn.Module):
    def __init__(self, teacher_forcing_ratio = 0.6, beam_size = 10, out_vocab = out_vocab):
        super(AttnSeq2seq, self).__init__()
        self.encoder = AttnEncoder()
        self.decoder = AttnDecoder()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        self.out_vocab = out_vocab

    def forward(self, x, y):
        if(self.training):
            return self.train_forward(x, y)
        else:
            return self.test_forward_beam(x, y)
        
    def train_forward(self, x, y):
        encoder_outputs, h, c = self.encoder(x)
        out = out_vocab['<start>']*torch.ones(x.size(0), dtype = torch.long).to(device)
        outputs = torch.zeros(y.size(0), y.size(1), len(out_vocab)).to(device)
        for i in range(y.size(1)):
            if(np.random.random() < self.teacher_forcing_ratio):
                if(i != 0):
                    out = y[:, i-1]
            else:
                if(i != 0):
                    out = torch.argmax(out, dim=1)
            out, h, c = self.decoder(out, h, c, encoder_outputs)
            outputs[:, i] = out
        return outputs
    
    def test_forward_beam(self, x, y):
        encoder_outputs, h, c = self.encoder(x)
        out = out_vocab['<start>']*torch.ones(x.size(0), dtype = torch.long).to(device)
        beam = [beamdata(0, [out], h, c)]
        for i in range(y.size(1)):
            new_beam = []
            for b in beam:
                out, h, c = self.decoder(torch.tensor(b.sequence[-1]), b.hidden, b.cell, encoder_outputs)
                out = F.log_softmax(out, dim=1)
                out = torch.topk(out, self.beam_size, dim=1)
                for j in range(self.beam_size):
                    new_beam.append(beamdata(b.score + out[0][:,j], b.sequence + [out[1][:,j]], h, c))

            beam = sorted(new_beam, key=lambda x: x.score.sum(), reverse=True)[:self.beam_size]

        sequence = torch.zeros(y.size(0), y.size(1), dtype=torch.long).to(device)
        for i in range(y.size(1)):
            sequence[:, i] = torch.tensor(beam[0].sequence[i])

        return sequence

# %%
class BertAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BertAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim*4, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden.unsqueeze(1)
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class BertEncoder(nn.Module):
    def __init__(self, BERT_model, hidden_dim = hidden_dim):
        super(BertEncoder, self).__init__()
        self.bert = BERT_model
        self.hidden_dim = hidden_dim
        self.linear_h = nn.Linear(768, self.hidden_dim)
        self.linear_c = nn.Linear(768, self.hidden_dim)

    def forward(self, x, att):
        x = self.bert(x, attention_mask=att).last_hidden_state
        h = torch.tanh(self.linear_h(x[:,0]))
        c = torch.tanh(self.linear_c(x[:,0]))
        return x, h, c
    
class BertDecoder(nn.Module):
    def __init__(self, out_vocab = out_vocab, hidden_dim = hidden_dim):
        super(BertDecoder, self).__init__()
        self.embedding = nn.Embedding(len(out_vocab), embedding_dim, padding_idx=0)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_dim + 3*self.hidden_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, len(out_vocab))
        self.attention = BertAttention(self.hidden_dim)

    def forward(self, x, h, c, encoder_outputs):
        x = x.unsqueeze(1)
        x = self.dropout(self.embedding(x))
        attention_weights = self.attention(encoder_outputs, h)
        attention_weights = attention_weights.unsqueeze(1)
        context = attention_weights.bmm(encoder_outputs)
        x = torch.cat((x, context), dim=2)
        x, (h, c) = self.lstm(x, (h.unsqueeze(0), c.unsqueeze(0)))
        x = self.linear(x.squeeze(1))
        h = h.squeeze(0)
        c = c.squeeze(0)
        return x, h, c
    
class BertSeq2seq(nn.Module):
    def __init__(self, BERT_model, teacher_forcing_ratio = 0.6, beam_size = 10, out_vocab = out_vocab):
        super(BertSeq2seq, self).__init__()
        self.encoder = BertEncoder(BERT_model)
        self.decoder = BertDecoder()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        self.out_vocab = out_vocab

    def forward(self, x, att, y):
        if(self.training):
            return self.train_forward(x, att, y)
        else:
            return self.test_forward_beam(x, att, y)
        
    def train_forward(self, x, att, y):
        encoder_outputs, h, c = self.encoder(x, att)
        out = out_vocab['<start>']*torch.ones(y.size(0), dtype = torch.long).to(device)
        outputs = torch.zeros(y.size(0), y.size(1), len(out_vocab)).to(device)
        for i in range(y.size(1)):
            if(np.random.random() < self.teacher_forcing_ratio):
                if(i != 0):
                    out = y[:, i-1]
            else:
                if(i != 0):
                    out = torch.argmax(out, dim=1)
            out, h, c = self.decoder(out, h, c, encoder_outputs)
            outputs[:, i] = out
        return outputs
    
    def test_forward_beam(self, x, att, y):
        encoder_outputs, h, c = self.encoder(x, att)
        out = out_vocab['<start>']*torch.ones(y.size(0), dtype = torch.long).to(device)
        beam = [beamdata(0, [out], h, c)]
        for i in range(y.size(1)):
            new_beam = []
            for b in beam:
                out, h, c = self.decoder(torch.tensor(b.sequence[-1]), b.hidden, b.cell, encoder_outputs)
                out = F.log_softmax(out, dim=1)
                out = torch.topk(out, self.beam_size, dim=1)
                for j in range(self.beam_size):
                    new_beam.append(beamdata(b.score + out[0][:,j], b.sequence + [out[1][:,j]], h, c))

            beam = sorted(new_beam, key=lambda x: x.score.sum(), reverse=True)[:self.beam_size]

        sequence = torch.zeros(y.size(0), y.size(1), dtype=torch.long).to(device)
        for i in range(y.size(1)):
            sequence[:, i] = torch.tensor(beam[0].sequence[i])

        return sequence

# %%
S = seq2seq()
S.to(device)
# S = nn.DataParallel(S)
optimizer_S = optim.Adam(S.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=out_vocab['<pad>'])

Att_S = AttnSeq2seq()
Att_S.to(device)
# Att_S = nn.DataParallel(Att_S)
optimizer_Att_S = optim.Adam(Att_S.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=out_vocab['<pad>'])

Bert_S_frozen = BertSeq2seq(BERT_frozen)
Bert_S_frozen.to(device)
# Bert_S_frozen = nn.DataParallel(Bert_S_frozen)
optimizer_Bert_S_frozen = optim.Adam(Bert_S_frozen.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=out_vocab['<pad>'])

Bert_S_fine_tuned = BertSeq2seq(BERT_fine_tuned)
Bert_S_fine_tuned.to(device)
# Bert_S_fine_tuned = nn.DataParallel(Bert_S_fine_tuned)
optimizer_Bert_S_fine_tuned = optim.Adam(Bert_S_fine_tuned.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=out_vocab['<pad>'])

# %%
def beam_search_prediction(test_data_file, loader, S, model_to_train):
    f = open(test_data_file, 'rb')
    df = pd.DataFrame(json.load(f))

    #convert the data
    # tr_data = transform(data)

    #write back to file
    # f = open(test_data_file, 'w')
    # json.dump(tr_data, f, indent='\t', separators=(',', ': '))

    # df_copy = pd.DataFrame(df).copy()
    answers = []

    S.eval()
    for i, p in enumerate(loader):
        p = p.to(device)
        l = torch.zeros(1,128)
        l = l.to(device)
        # a = a.to(device)
        output = 0
        if(model_to_train == 0 or model_to_train == 1):
            output = S(p, l)
        else:
            output = S(p['input_ids'], p['attention_mask'], l)

        lst = []
        for j in range(output.size(1)):
            lst.append(rev_out_vocab[output[0][j].item()])

        ans = ","
        # print(lst)
        for j in range(1,len(lst)):
            if(lst[j] == '<eos>'):
                break
            if(lst[j][-1].isdigit()):
                ans += lst[j] + ','
            else:
                ans = ans[:-1] + ')|' + lst[j] + '('

        ans = ans[2:-1] + ')|'
        answers.append(ans)

    df['predicted'] = answers
    df = df.to_json(orient="records")
    df = json.loads(df)
    f = open(test_data_file, 'w')
    json.dump(df, f, indent='\t')

# %%



import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference on models.")
    parser.add_argument("--model_file", type=str, help="Path to the trained model")
    parser.add_argument("--beam_size", type=int, choices=[1, 10, 20], default=10, help="Beam size (choose from 1, 10, 20)")
    parser.add_argument("--model_type", type=str, choices=["lstm_lstm", "lstm_lstm_attn", "bert_lstm_attn_frozen", "bert_lstm_attn_tuned"], required=True, help="Model type")
    parser.add_argument("--test_data_file", type=str, help="Path to the JSON file containing the problems")
    args = parser.parse_args()

    model_file = args.model_file
    beam_size = args.beam_size
    model_type = args.model_type
    test_data_file = args.test_data_file
    model_to_train = models[model_type]

    vocab = json.load(open(dir + 'var/glove_vocab.json', 'r'))
    out_vocab = json.load(open(dir + 'var/glove_out_vocab.json', 'r'))
    rev_out_vocab = json.load(open(dir + 'var/glove_rev_out_vocab.json', 'r'))
    rev_out_vocab = {int(k):v for k,v in rev_out_vocab.items()}
    embedding = torch.load(dir + 'var/glove_embedding.pth')

    collator = collate_fn

    test_dataset = seqDataset(test_data_file)

    if(model_to_train == 0 or model_to_train==1):
        test_dataset.Problem = modify_dataset(vocab, test_dataset.Problem)
    else:
        vocab = bert_vocab.vocab
        collator = bert_collate

    test_acc_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the "module." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # %%
    model_list = [S, Att_S, Bert_S_frozen, Bert_S_fine_tuned]
    model_list[model_to_train].load_state_dict(new_state_dict)
    model_list[model_to_train].beam_size = beam_size
    model_list[model_to_train].to(device)

    print("Model file:", model_file)
    print("Beam size:", beam_size)
    print("Model type:", model_type)
    print("Test data file:", test_data_file)
    print('Model_no: ', model_to_train)

    beam_search_prediction(test_data_file, test_acc_loader, model_list[model_to_train], model_to_train)

if __name__ == "__main__":
    main()