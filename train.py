import torch
import torch.nn as nn
import torch.optim as optim
from model import Seq2seq, Encoder, Decoder
from main import vocab, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
embedding_size = 256
hidden_size    = 256  
num_layers     = 2      

attention_dim  = 256
vocab_size     = vocab.n_words
num_epochs     = 15   
learning_rate  = 0.001

# --- Initialize ---
encoder = Encoder(vocab_size, embedding_size, hidden_size, num_layers, PAD_IDX, dropout=0.3).to(device)
decoder = Decoder(embedding_size, vocab_size, vocab_size, hidden_size, num_layers, attention_dim, dropout=0.3).to(device)
model = Seq2seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # أضفنا weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

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
        return " ".join([vocab.index2word[idx] for idx in outputs if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]])

print("Starting Training...")
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0
     
    tf_ratio = max(0.1, 0.5 - (epoch * 0.05))

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, targets, teacher_forcing_ratio=tf_ratio)
        
        output_dim = outputs.shape[-1]
        loss = criterion(outputs[:, 1:].reshape(-1, output_dim), targets[:, 1:].reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets, teacher_forcing_ratio=0)
            val_loss += criterion(outputs[:, 1:].reshape(-1, output_dim), targets[:, 1:].reshape(-1)).item()
    
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "chatbot_best_model.pth")
        status = "⭐ Model Saved!"
    else: status = ""

    print(f"Epoch [{epoch}/{num_epochs}] | TF: {tf_ratio:.2f} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} {status}")
    print(f"Sample: {evaluate_sample('hello')}")
    print("-" * 20)  
