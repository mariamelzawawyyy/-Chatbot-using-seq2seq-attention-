# 🤖 Deep Learning from Scratch: A Journey through Seq2Seq & Attention

### 🌟 The Philosophy: Why "From Scratch"?
In an era of pre-built libraries and high-level APIs, I chose to build this architecture **from the ground up**. Every layer of the **Bi-LSTM**, the **Bahdanau Attention** mechanism, and the entire training pipeline was coded manually. Mastering the "shapes" and tensor transformations was essential to transform Deep Learning from a "black box" into an intuitive skill.

### 📉 The Evolution: Overcoming Overfitting
My first attempt was a "Vanilla" Seq2Seq model. It looked good on paper, but the reality of training was harsh: the **Validation Loss was skyrocketing**. The model was simply memorizing the training data instead of learning the art of conversation (Severe Overfitting).

To bridge the gap between memorization and generalization, I re-engineered the pipeline with these core pillars:

*   **Semantic Foundations:** I integrated **GloVe 6B (300d)** embeddings. Instead of starting with random noise, the model began with a pre-trained understanding of human language.
*   **The Struggle for Independence:** I introduced **Teacher Forcing Decay**. By gradually removing the "training wheels," the model transitioned from mimicking the ground truth to generating its own logical sequences (reaching 90% independence).
*   **Precision & Stability:** 
    *   **Label Smoothing (0.1):** To prevent overconfidence and force the model to explore better generalizations.
    *   **ReduceLROnPlateau:** A dynamic learning rate scheduler that allowed the model to "slow down" and refine its weights whenever it hit a plateau.
*   **Vocabulary & Regularization:** I capped the vocabulary at **9,429 words** to reduce noise and added **Dropout** layers to ensure the Bi-LSTM units remained robust.

### 🏗️ Technical Architecture
*   **Encoder:** Multi-layer **Bidirectional LSTM** to capture both forward and backward context from the input.
*   **Attention Bridge:** Custom **Bahdanau (Additive) Attention** to compute relevance scores for each input token at every decoding step.
*   **Decoder:** Unidirectional LSTM optimized with **Log-Softmax** for stable and coherent word generation.

### 🏆 The Result
The transformation was night and day. The unstable, skyrocketing curves were replaced by a **smooth, steadily declining loss**, stabilizing at a **Val Loss of ~5.8**—a solid and reliable benchmark for this architecture on the **DailyDialog** dataset.

---

### 🧠 Lessons from the Trenches
This project was a lesson in persistence. It taught me that data cleaning is 50% of the battle and that **Overfitting** isn't a dead end—it's an invitation to understand your model deeper. Seeing the model finally generate a coherent response after weeks of debugging tensor shapes was the ultimate reward.

---

### 🚀 How to Run
1. Clone the repository.
2. Ensure you have the `nmt_checkpoint.pth` and `vocab.pkl` files.
3. Run the inference script:
   ```python
   # Example usage
   response = chat(model, vocab, "Hello, how are you?")
   print(response)
