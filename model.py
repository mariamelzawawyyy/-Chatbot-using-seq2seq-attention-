from main import vocab, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, src_max_len, tgt_max_len, train_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

############################### Encoder ########################################

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)
        outputs, (hidden_state, cell_state) = self.bilstm(x)

        # دمج forward + backward (يشتغل صح مع num_layers=1)
        hidden_state = self.fc_hidden(torch.cat([hidden_state[0:1], hidden_state[1:2]], dim=2))
        cell_state   = self.fc_cell(torch.cat([cell_state[0:1],   cell_state[1:2]],   dim=2))

        return outputs, hidden_state, cell_state

############################### Attention ########################################

class Attention(nn.Module):
    def __init__(self, enc_outputs_dim, dec_hidden_dim, attention_dim):
        super().__init__()
        self.W_h  = nn.Linear(enc_outputs_dim, attention_dim)
        self.W_s  = nn.Linear(dec_hidden_dim,  attention_dim)
        self.tanh = nn.Tanh()
        self.v    = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # encoder_outputs : (batch, seq_len, hidden*2)
        # decoder_hidden  : (batch, hidden)

        enc_proj = self.W_h(encoder_outputs)               # (batch, seq_len, attn_dim)
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1)   # (batch, 1, attn_dim)

        energy      = self.tanh(enc_proj + dec_proj)       # (batch, seq_len, attn_dim)
        attn_scores = self.v(energy).squeeze(2)            # (batch, seq_len)

        attn_weights   = F.softmax(attn_scores, dim=1)     # (batch, seq_len)
        context_vector = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)                                       # (batch, hidden*2)

        return context_vector, attn_weights

############################### Decoder ########################################

class Decoder(nn.Module):
    def __init__(self, embedding_size, input_size, output_size, hidden_size, num_layers, attention_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.attention = Attention(
            enc_outputs_dim=hidden_size * 2,
            dec_hidden_dim=hidden_size,
            attention_dim=attention_size
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch,)
        x = x.unsqueeze(1)    # (batch, 1)
        x = self.embedding(x) # (batch, 1, emb)

        dec_hidden = hidden[-1]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs)

        lstm_input = torch.cat((x, context.unsqueeze(1)), dim=2)
        # (batch, 1, emb + hidden*2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc_out(output.squeeze(1))  # (batch, vocab_size)

        return predictions, hidden, cell, attn_weights

############################### Seq2Seq ########################################

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size        = input.shape[0]
        target_len        = target.shape[1]
        target_vocab_size = self.decoder.fc_out.out_features

        # outputs على نفس الـ device زي الـ input
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(input.device)

        encoder_outputs, hidden, cell = self.encoder(input)

        x = target[:, 0]  # <SOS> token

        for t in range(1, target_len):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1

        return outputs 
