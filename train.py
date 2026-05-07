import torch
import torch.nn as nn
import torch.optim as optim

from model import Seq2seq, Encoder, Decoder
from main import vocab, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, train_loader, val_loader, load_glove_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 FIX: GloVe path (المهم)
GLOVE_PATH = "/content/-Chatbot-using-seq2seq-attention-/glove_embeddings/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"


embedding_size = 100   
hidden_size    = 256    
num_layers     = 2     
attention_dim  = 128
vocab_size     = vocab.n_words
num_epochs     = 30       
learning_rate  = 0.0005  

# 🔥 تحميل GloVe هنا بدل main
glove_weights = load_glove_weights(vocab, GLOVE_PATH, embed_dim=100)

encoder = Encoder(vocab_size, embedding_size, hidden_size, num_layers, PAD_IDX, 
                  dropout=0.5, weights_matrix=glove_weights).to(device)

decoder = Decoder(embedding_size, vocab_size, vocab_size, hidden_size, num_layers, 
                  attention_dim, dropout=0.5, weights_matrix=glove_weights).to(device)

model = Seq2seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

best_val_loss = float('inf')
patience = 4   
counter = 0

print(f"Starting Training with GloVe 2024 | Vocab Size: {vocab.n_words}")

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0

    tf_ratio = max(0.1, 0.5 - (epoch * 0.03))

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, targets, teacher_forcing_ratio=tf_ratio)

        output_dim = outputs.shape[-1]

        loss = criterion(outputs[:, 1:].reshape(-1, output_dim),
                         targets[:, 1:].reshape(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, targets, teacher_forcing_ratio=0)

            val_loss += criterion(outputs[:, 1:].reshape(-1, output_dim),
                                  targets[:, 1:].reshape(-1)).item()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "chatbot_glove_best.pth")
        counter = 0
        status = "⭐ Best Model Saved!"
    else:
        counter += 1
        status = f"No improvement ({counter}/{patience})"

    print(f"Epoch [{epoch}/{num_epochs}] | TF: {tf_ratio:.2f} | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {status}")

    if counter >= patience:
        print(f"🛑 Early Stopping at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
        break

print("Training Process Finished Successfully.") 
