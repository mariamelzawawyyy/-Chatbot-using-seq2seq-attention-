# train.py
import torch
import torch.nn as nn
from model import Seq2seq, Encoder, Decoder
from main import vocab, PAD_IDX, train_loader, val_loader

# -------------------------------
# 1️⃣ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2️⃣ Hyperparameters
# -------------------------------
embedding_size         = 256
hidden_size            = 128
num_layers             = 1
attention_dim          = 256
vocab_size             = vocab.n_words
num_epochs             = 10
learning_rate          = 1e-3
teacher_forcing_ratio  = 0.5

# -------------------------------
# 3️⃣ Initialize Model
# -------------------------------
encoder = Encoder(
    vocab_size     = vocab_size,
    embedding_size = embedding_size,
    hidden_size    = hidden_size,
    num_layers     = num_layers,
    padding_index  = PAD_IDX
)
decoder = Decoder(
    embedding_size = embedding_size,
    input_size     = vocab_size,
    output_size    = vocab_size,
    hidden_size    = hidden_size,
    num_layers     = num_layers,
    attention_size = attention_dim
)

model     = Seq2seq(encoder, decoder).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 4️⃣ Training Loop
# -------------------------------
for epoch in range(1, num_epochs + 1):

    # ── Train ──
    model.train()
    train_loss = 0

    for inputs, targets in train_loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, targets, teacher_forcing_ratio=teacher_forcing_ratio)

        # skip <SOS> then flatten للـ CrossEntropyLoss
        outputs = outputs[:, 1:, :].reshape(-1, vocab_size)
        targets = targets[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        # ✅ gradient clipping عشان تتجنب exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ── Validation ──
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs, targets, teacher_forcing_ratio=0)  # no teacher forcing

            outputs = outputs[:, 1:, :].reshape(-1, vocab_size)
            targets = targets[:, 1:].reshape(-1)

            val_loss += criterion(outputs, targets).item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}") 
