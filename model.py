from main import vocab,PAD_IDX,SOS_IDX,EOS_IDX,UNK_IDX,src_max_len,tgt_max_len,train_loader 
import torch.nn.functional as F
import torch
import torch.nn as nn  
import random  
 
############################### Encoder Class ############################################ 
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size , hidden_size, num_layers, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size , padding_idx=padding_index)
        self.bilstm = nn.LSTM(
            input_size=embedding_size ,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        outputs, (hidden_state, cell_state) = self.bilstm(x)  # outputs: [batch, seq_len, hidden*2]

        # دمج forward + backward hidden state لكل layer (هنا layer=1)
        hidden_state = self.fc_hidden(torch.cat([hidden_state[0:1], hidden_state[1:2]], dim=2))
        cell_state   = self.fc_cell(torch.cat([cell_state[0:1], cell_state[1:2]], dim=2))

        return outputs, hidden_state, cell_state 
    
##################################### Attention Class ####################################  

class Attention(nn.Module ) :
    def __init__(self,enc_outputs_dim,dec_hidden_dim,attention_dim) :
        super().__init__( ) 
        self.W_h = nn.Linear(enc_outputs_dim,attention_dim) #projection for encoder outputs  
        self.W_s = nn.Linear(dec_hidden_dim,attention_dim) #projection for decoder hidden
        self.tanh = nn.Tanh() #will be used for computing similarity score  
        self.v = nn.Linear(attention_dim ,1, bias=False) 
    def forward(self,decoder_hidden,encoder_outputs ) :
        #encoder_outputs_shape = (batch_size,seq_len,hidden_size*2)
        #decoder_hidden_shape = (batch_size , hidden_size)

        #Projection     
        enc_projection = self.W_h( encoder_outputs ) # (batch_size,seq_len,attn_dim)
        dec_projection = self.W_s(decoder_hidden ).unsqueeze(1) #(batch_size,1,attn_dim)
        
        #Attention scores : Using Additive Attention (Bahdanau Attention): 
        energy = self.tanh(enc_projection + dec_projection) #(batch_size,seq_len,attn_dim)
        attn_scores = self.v(energy).squeeze(2) #(batch_size , seq_len )

        #Computing the context vector   
        attention_weights = nn.functional.softmax(attn_scores,1) #(batch_size,seq_len) 
        attention_weights = attention_weights.unsqueeze(1) # (batch_size,1,seq_len)

        context_vector = torch.bmm(attention_weights,encoder_outputs).squeeze(1)  
        #(batch_size  , 1 ,enc_outputs_dim)  
  
        return context_vector, attention_weights
    
################################  DECODER  ############################################## 
 
class Decoder(nn.Module):
   def __init__(self, embedding_size ,input_size, output_size, hidden_size,num_layers,attention_size ):
       super().__init__( )
       self.embedding = nn.Embedding(input_size, embedding_size)  
       self.attention = Attention(enc_outputs_dim=hidden_size*2, dec_hidden_dim=hidden_size, attention_dim  =attention_size )
       self.lstm = nn.LSTM( input_size=embedding_size + hidden_size*2 ,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
       self.fc_out = nn.Linear(hidden_size, output_size)

   def forward(self, x,  hidden , cell, encoder_outputs): 
       #first: compute the embedding dimension of the input 
       # x shape = (batch_size)
       x = x.unsqueeze(1)   #(batch_size,1)
       x = self.embedding(x) #(batch_size,1,embedding_size) 

  
       #second  : compute context vector from attention 
       dec_hidden =  hidden[-1]  # in case the decoder has more than one
       #layer , we take the hidden state in the last layer , but if we have only one
       # then its only one hidden state but it will still work (General case)
       context, attn_weights =self.attention ( hidden[-1], encoder_outputs)

       # third: add context vector as input to the lstm + embedding layer 
       lstm_input =  torch.cat((x , context.unsqueeze(1) ), dim=2)
       #lstm_input.shape = (batch_size, 1, embedding_dim + enc_outputs_dim)
        

       # pass the inputs to the  lstm 
       outputs, (hidden, cell) = self.lstm ( lstm_input , (hidden, cell))

       #final predictions
       predictions = self.fc_out(outputs.squeeze(1))

       return predictions, hidden, cell, attn_weights

#################################  Seq2seq Model  ######################################
class Seq2seq (nn.Module):
    def __init__(self,encoder,decoder ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 

    def forward(self,input,target,teacher_forcing_ratio = 0.5):
        batch_size = input.shape[0] 
        target_len = target.shape[1] 
        target_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size,target_len,target_vocab_size ) 
        #encoder step 
        encoder_outputs, hidden, cell = self.encoder(input) 

        #decoder step
        x = target[:,0] # the <sos> token 

        for t in range(1, target_len): # at each timestep (one token at a time )
            output, hidden, cell, _ = self.decoder(x, hidden, cell,encoder_outputs)  
            outputs[:, t, :] = output

            # decide if we use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1

        return outputs
    
 
       


         







       

       

