import re  
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

 
SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3
 
GLOVE_PATH = "/content/-Chatbot-using-seq2seq-attention-/glove_embeddings/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt" 
 
 
##############################
# 2️⃣ Clean Function
##############################
def clean_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"!+", " ! ", s)
    s = re.sub(r"\?+", " ? ", s)
    s = re.sub(r"[^a-zA-Z0-9\s!?]", "", s) 
    s = re.sub(r"\s+", " ", s)
    return s.strip()

##############################
# 3️⃣ Vocab Class (The "Smart" Version)
##############################
class Vocab:
    def __init__(self, min_count=3):
        self.word2index = {"SOS": SOS_IDX, "EOS": EOS_IDX, "PAD": PAD_IDX, "UNK": UNK_IDX}
        self.index2word = {SOS_IDX: "SOS", EOS_IDX: "EOS", PAD_IDX: "PAD", UNK_IDX: "UNK"}
        self.word2count = {}
        self.n_words = 4
        self.min_count = min_count

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.word2count[word] = self.word2count.get(word, 0) + 1

    def build_index(self):
        # قتل الـ Overfitting عن طريق تجاهل الكلمات النادرة جداً
        for word, count in self.word2count.items():
            if count >= self.min_count: 
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.index2word[self.n_words] = word
                    self.n_words += 1

##############################
# 4️⃣ Data Loading & Pairs
##############################
def prepare_pairs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        conversations = f.read().strip().split('\n')
    
    all_pairs = []
    for conversation in conversations:
        utterances = [clean_sentence(u) for u in conversation.split("__eou__") if u.strip()]
        for i in range(len(utterances) - 1):
            all_pairs.append((utterances[i], utterances[i + 1]))
    return all_pairs

def build_vocab(pairs, min_count=3):
    vocab = Vocab(min_count=min_count)
    for inp, tgt in pairs:
        vocab.add_sentence(inp)
        vocab.add_sentence(tgt)
    vocab.build_index() 
    return vocab

##############################
# 5️⃣ GloVe Embeddings Integration
##############################
def load_glove_weights(vocab, glove_path, embed_dim=100):
    print(f"--- Loading GloVe 2024 weights from: {glove_path} ---")
    
   
    weights_matrix = np.random.normal(scale=0.6, size=(vocab.n_words, embed_dim))
    
    count = 0
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab.word2index:
                    vector = np.asarray(values[1:], dtype='float32')
                    weights_matrix[vocab.word2index[word]] = vector
                    count += 1
        print(f"Matched {count}/{vocab.n_words} words with GloVe 2024.")
    except Exception as e:
        print(f"Error loading GloVe: {e}")
        
    return torch.tensor(weights_matrix, dtype=torch.float32)

##############################
# 6️⃣ Numericalize & Dataset
##############################
def sentence_to_tensor(vocab, sentence, max_len, is_target=False):
    indices = [SOS_IDX] if is_target else []
    for word in sentence.split():
        indices.append(vocab.word2index.get(word, UNK_IDX))
    indices.append(EOS_IDX)
    
    if len(indices) < max_len:
        indices += [PAD_IDX] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    return torch.tensor(indices, dtype=torch.long)

class ChatDataset(Dataset):
    def __init__(self, pairs, vocab, src_max, tgt_max):
        self.inputs = [sentence_to_tensor(vocab, p[0], src_max, False) for p in pairs]
        self.targets = [sentence_to_tensor(vocab, p[1], tgt_max, True) for p in pairs]

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

##############################
# ✅ Final Execution
##############################
 
train_pairs = prepare_pairs("train/train/dialogues_train.txt")
val_pairs   = prepare_pairs("validation/validation/dialogues_validation.txt")

 
src_lengths = [len(p[0].split()) for p in train_pairs]
tgt_lengths = [len(p[1].split()) for p in train_pairs]
src_max_len = int(np.percentile(src_lengths, 95)) + 1
tgt_max_len = int(np.percentile(tgt_lengths, 95)) + 2
 
vocab = build_vocab(train_pairs, min_count=3)
 
glove_weights = load_glove_weights(vocab, GLOVE_PATH, embed_dim=100)

 
train_loader = DataLoader(ChatDataset(train_pairs, vocab, src_max_len, tgt_max_len), batch_size=32, shuffle=True)
val_loader   = DataLoader(ChatDataset(val_pairs, vocab, src_max_len, tgt_max_len), batch_size=32, shuffle=False)

print(f"Data is ready! Vocab size: {vocab.n_words}")  


 
 
 
