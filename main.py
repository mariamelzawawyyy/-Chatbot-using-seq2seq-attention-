# main.py
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

##############################
# 0️⃣ Clean Function
##############################
def clean_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"!+", " ! ", s)
    s = re.sub(r"\?+", " ? ", s)
    s = re.sub(r"[^a-zA-Z0-9\s!?]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

##############################
# 1️⃣ Load Raw Conversations
##############################
def load_raw_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        conversations = f.read().strip().split('\n')
    return conversations

##############################
# 2️⃣ Prepare Pairs
##############################
def prepare_pairs(file_path):
    conversations = load_raw_data(file_path)
    all_pairs = []
    for conversation in conversations:
        utterances = [u.strip() for u in conversation.split("__eou__") if u.strip()]
        utterances = [clean_sentence(u) for u in utterances]
        for i in range(len(utterances) - 1):
            all_pairs.append((utterances[i], utterances[i + 1]))
    return all_pairs

##############################
# 3️⃣ Percentile Max Len
##############################
def percentile_max_len(pairs, percentile=95):
    src_lengths = [len(inp.split()) for inp, tgt in pairs]
    tgt_lengths = [len(tgt.split()) for inp, tgt in pairs]
    src_max = int(np.percentile(src_lengths, percentile)) + 1   # +1 for EOS
    tgt_max = int(np.percentile(tgt_lengths, percentile)) + 2   # +2 for SOS + EOS
    return src_max, tgt_max

##############################
# 4️⃣ Vocab
##############################
SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

class Vocab:
    def __init__(self):
        self.word2index = {"SOS": SOS_IDX, "EOS": EOS_IDX, "PAD": PAD_IDX, "UNK": UNK_IDX}
        self.index2word = {SOS_IDX: "SOS", EOS_IDX: "EOS", PAD_IDX: "PAD", UNK_IDX: "UNK"}
        self.word2count = {}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] = self.word2count.get(word, 0) + 1

def build_vocab(pairs):
    vocab = Vocab()
    for inp, tgt in pairs:
        vocab.add_sentence(inp)
        vocab.add_sentence(tgt)
    return vocab

##############################
# 5️⃣ Numericalize
##############################
def sentence_to_tensor(vocab, sentence, is_target=False):
    indices = [SOS_IDX] if is_target else []
    for word in sentence.split():
        indices.append(vocab.word2index.get(word, UNK_IDX))
    indices.append(EOS_IDX)
    return torch.tensor(indices, dtype=torch.long)

def numericalize_pairs(pairs, vocab):
    inputs, targets = [], []
    for inp, tgt in pairs:
        inputs.append(sentence_to_tensor(vocab, inp, is_target=False))
        targets.append(sentence_to_tensor(vocab, tgt, is_target=True))
    return inputs, targets

##############################
# 6️⃣ Padding
##############################
def pad_tensor(tensor, max_len, pad_idx):
    length = tensor.size(0)
    if length < max_len:
        padding = torch.full((max_len - length,), pad_idx, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding])
    else:
        tensor = tensor[:max_len]
    return tensor

##############################
# 7️⃣ Dataset
##############################
class ChatDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs  = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

##############################
# ✅ 8️⃣ Global Setup (accessible for import)
##############################
#  for colab   
train_path = "train/train/dialogues_train.txt"
val_path   = "validation/validation/dialogues_validation.txt"
test_path  = "test/test/dialogues_test.txt" 

 

train_pairs = prepare_pairs(train_path)
val_pairs   = prepare_pairs(val_path)
test_pairs  = prepare_pairs(test_path)

src_max_len, tgt_max_len = percentile_max_len(train_pairs)

vocab = build_vocab(train_pairs)

train_inputs,  train_targets = numericalize_pairs(train_pairs, vocab)
val_inputs,    val_targets   = numericalize_pairs(val_pairs,   vocab)
test_inputs,   test_targets  = numericalize_pairs(test_pairs,  vocab)

train_inputs  = [pad_tensor(t, src_max_len, PAD_IDX) for t in train_inputs]
train_targets = [pad_tensor(t, tgt_max_len, PAD_IDX) for t in train_targets]
val_inputs    = [pad_tensor(t, src_max_len, PAD_IDX) for t in val_inputs]
val_targets   = [pad_tensor(t, tgt_max_len, PAD_IDX) for t in val_targets]
test_inputs   = [pad_tensor(t, src_max_len, PAD_IDX) for t in test_inputs]
test_targets  = [pad_tensor(t, tgt_max_len, PAD_IDX) for t in test_targets]

train_dataset = ChatDataset(train_inputs,  train_targets)
val_dataset   = ChatDataset(val_inputs,    val_targets)
test_dataset  = ChatDataset(test_inputs,   test_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False) 


 
 
 
