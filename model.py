import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
 
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_index, dropout=0.5, weights_matrix=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
         
        if weights_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False, padding_idx=padding_index)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
            
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        encoder_outputs, (hidden, cell) = self.bilstm(embedded)

        hidden = hidden.view(self.num_layers, 2, x.shape[0], self.hidden_size)
        last_hidden = torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)
        final_hidden = self.fc_hidden(last_hidden).unsqueeze(0).repeat(self.num_layers, 1, 1)

        cell = cell.view(self.num_layers, 2, x.shape[0], self.hidden_size)
        last_cell = torch.cat((cell[-1, 0, :, :], cell[-1, 1, :, :]), dim=1)
        final_cell = self.fc_cell(last_cell).unsqueeze(0).repeat(self.num_layers, 1, 1)

        return encoder_outputs, final_hidden, final_cell  

class Attention(nn.Module):
    def __init__(self, enc_outputs_dim, dec_hidden_dim, attention_dim):
        super().__init__()
        self.W_h  = nn.Linear(enc_outputs_dim, attention_dim)
        self.W_s  = nn.Linear(dec_hidden_dim,  attention_dim)
        self.tanh = nn.Tanh()
        self.v    = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        enc_proj = self.W_h(encoder_outputs)               
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1)   
        energy   = self.tanh(enc_proj + dec_proj)       
        attn_scores = self.v(energy).squeeze(2)            
        attn_weights   = F.softmax(attn_scores, dim=1)     
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)            
        return context_vector, attn_weights

class Decoder(nn.Module):
     
    def __init__(self, embedding_size, input_size, output_size, hidden_size, num_layers, attention_size, dropout=0.5, weights_matrix=None):
        super().__init__()
        
        if weights_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(input_size, embedding_size)
            
        self.dropout = nn.Dropout(dropout)
        
        self.attention = Attention(
            enc_outputs_dim=hidden_size * 2,
            dec_hidden_dim=hidden_size,
            attention_dim=attention_size
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)    
        embedded = self.dropout(self.embedding(x))

        dec_hidden = hidden[-1]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs)

        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc_out(output.squeeze(1))  

        return predictions, hidden, cell, attn_weights

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size        = input.shape[0]
        target_len        = target.shape[1]
        target_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(input.device)
        encoder_outputs, hidden, cell = self.encoder(input)
        x = target[:, 0]  
        for t in range(1, target_len):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1
        return outputs 
