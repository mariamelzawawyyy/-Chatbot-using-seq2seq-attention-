from main import vocab, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, src_max_len, tgt_max_len, train_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

############################### Encoder ########################################

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, padding_index):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear layers لتحويل الـ BiLSTM states (hidden*2) إلى حجم الـ Decoder hidden (hidden)
        # نستخدم hidden_size * 2 لأن الـ LSTM bidirectional
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x) 
        # embedded shape: [batch_size, seq_len, embedding_size]
        
        encoder_outputs, (hidden, cell) = self.bilstm(embedded)
        # encoder_outputs shape: [batch_size, seq_len, hidden_size * 2]
        # hidden shape: [num_layers * 2, batch_size, hidden_size]

        # --- التعامل مع الـ Hidden State ---
        # 1. إعادة تشكيل الـ hidden لفك فصل الطبقات عن الاتجاهات
        # الـ hidden حالياً مرتب كـ [L1_fwd, L1_back, L2_fwd, L2_back, ...]
        hidden = hidden.view(self.num_layers, 2, x.shape[0], self.hidden_size)
        
        # 2. نأخذ آخر طبقة فقط (Index -1) للاتجاهين (Forward و Backward)
        # hidden[-1, 0, :, :] هو الـ forward لآخر طبقة
        # hidden[-1, 1, :, :] هو الـ backward لآخر طبقة
        last_hidden_fwd = hidden[-1, 0, :, :]
        last_hidden_back = hidden[-1, 1, :, :]
        
        # 3. ندمجهم مع بعض (Concatenate)
        combined_hidden = torch.cat((last_hidden_fwd, last_hidden_back), dim=1) 
        # shape: [batch_size, hidden_size * 2]
        
        # 4. نمررهم على الـ Linear Layer لتقليل الحجم ليناسب الـ Decoder
        final_hidden = self.fc_hidden(combined_hidden).unsqueeze(0) 
        # shape: [1, batch_size, hidden_size] (جاهز كـ initial hidden للـ decoder)

        # --- نفس الخطوات للـ Cell State ---
        cell = cell.view(self.num_layers, 2, x.shape[0], self.hidden_size)
        last_cell_fwd = cell[-1, 0, :, :]
        last_cell_back = cell[-1, 1, :, :]
        combined_cell = torch.cat((last_cell_fwd, last_cell_back), dim=1)
        final_cell = self.fc_cell(combined_cell).unsqueeze(0)
        # shape: [1, batch_size, hidden_size]

        return encoder_outputs, final_hidden, final_cell  
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
