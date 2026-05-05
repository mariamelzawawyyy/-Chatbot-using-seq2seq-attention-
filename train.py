import torch
import torch.nn as nn
import torch.optim as optim
from model import Seq2seq, Encoder, Decoder
from main import vocab, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, train_loader, val_loader
import re

# -------------------------------
# 1️⃣ Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on device: {device}")

# -------------------------------
# 2️⃣ Hyperparameters
# -------------------------------
embedding_size        = 256
hidden_size           = 256  
num_layers            =  1   
attention_dim         = 256
vocab_size            = vocab.n_words
num_epochs            = 15   
learning_rate         = 0.001
teacher_forcing_ratio = 0.5

# -------------------------------
# 3️⃣ Initialize Model
# -------------------------------
encoder = Encoder(
    vocab_size     = vocab_size,
    embedding_size = embedding_size,
    hidden_size    = hidden_size,
    num_layers     = num_layers,
    padding_index  = PAD_IDX
).to(device)

decoder = Decoder(
    embedding_size = embedding_size,
    input_size     = vocab_size,
    output_size    = vocab_size,
    hidden_size    = hidden_size,
    num_layers     = num_layers,
    attention_size = attention_dim
).to(device)

model = Seq2seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 4️⃣ Helper function to test during training
# -------------------------------
def evaluate_sample(sentence):
    model.eval()
    with torch.no_grad():
        sentence = sentence.lower().strip()
        tokens = [vocab.word2index.get(word, UNK_IDX) for word in sentence.split()]
        tokens.append(EOS_IDX)
        input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        
        encoder_outputs, hidden, cell = model.encoder(input_tensor)
        
        outputs = [SOS_IDX]
        for _ in range(20):
            prev_word = torch.LongTensor([outputs[-1]]).to(device)
            prediction, hidden, cell, _ = model.decoder(prev_word, hidden, cell, encoder_outputs)
            best_guess = prediction.argmax(1).item()
            outputs.append(best_guess)
            if best_guess == EOS_IDX: break
            
        words = [vocab.index2word[idx] for idx in outputs if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
        return " ".join(words)

# -------------------------------
# 5️⃣ Training Loop
# -------------------------------
print("Starting Training...")

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs, targets, teacher_forcing_ratio=teacher_forcing_ratio)
 
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        targets = targets[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ── Validation ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets, teacher_forcing_ratio=0)
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            targets = targets[:, 1:].reshape(-1)
            val_loss += criterion(outputs, targets).item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Sample Chat -> Input: 'hello' | Response: {evaluate_sample('hello')}")
    print("-" * 30)

# -------------------------------
# 6️⃣ Save the Model
# -------------------------------
checkpoint = {
    'model_state': model.state_dict(),
    'vocab': vocab,
    'config': {
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'attention_dim': attention_dim
    }
}
torch.save(checkpoint, "chatbot_best_model.pth")
print("Training Complete & Model Saved!")  
