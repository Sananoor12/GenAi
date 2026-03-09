# Neural Storyteller: Building an Image Captioning System with CNN-LSTM

*How I built a model that looks at photos and writes captions — from scratch, on Flickr30k*

---

## Introduction

What if a machine could look at a photograph and describe it in plain English — just like a human?

That's exactly what **image captioning** does. It sits at the intersection of Computer Vision and Natural Language Processing, combining the power of convolutional neural networks (CNNs) with recurrent neural networks (RNNs) to generate meaningful text descriptions of images.

In this blog, I'll walk through how I built **Neural Storyteller** — an end-to-end image captioning system trained on the Flickr30k dataset, deployed as a live Gradio web app.

**GitHub Repository:** https://github.com/Sananoor12/GenAi

---

## The Dataset: Flickr30k

The [Flickr30k dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k) contains **31,783 images**, each paired with **5 human-written captions** — over 158,000 captions in total.

| Split | Images |
|-------|--------|
| Train | 28,000 |
| Validation | 1,000 |
| Test | ~2,783 |

Example captions for a single image:
- *"A man in a red shirt is climbing a rock wall."*
- *"A climber scales an outdoor rock face."*

---

## Architecture: Show and Tell

The architecture follows the classic **encoder-decoder** paradigm popularised by Vinyals et al. (2015) in *"Show and Tell"*:

### 🔍 Encoder — ResNet-50

A pre-trained **ResNet-50** acts as the visual backbone. The final classification layer is removed, leaving a **2048-dimensional feature vector** per image. Features are extracted offline and cached — this saves enormous compute time during training.

A linear layer then projects this 2048-d vector down to the LSTM hidden size (512).

```python
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(2048, hidden_size)

    def forward(self, features):
        return torch.tanh(self.linear(features))
```

### 🗣️ Decoder — LSTM

A single-layer **LSTM** generates the caption word-by-word. The encoder output initialises both h₀ and c₀.

```python
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, features):
        embeddings = self.embed(captions)
        h0 = features.unsqueeze(0)
        c0 = features.unsqueeze(0)
        output, _ = self.lstm(embeddings, (h0, c0))
        return self.linear(output)
```

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Embedding size | 256 |
| Hidden size | 512 |
| Learning rate | 3e-4 (Adam) |
| Batch size | 64 |
| Epochs | 20 |
| Loss | CrossEntropyLoss |

Training ran on **dual NVIDIA T4 GPUs** on Kaggle (CUDA 12.1). Each epoch took ~63 seconds.

### Loss Curve

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 3.37 | 3.24 |
| 5 | 2.60 | 2.97 |
| 10 | 2.18 | 3.02 |
| 15 | 1.88 | 3.17 |
| 20 | 1.64 | 3.36 |

Training loss steadily decreases. Validation loss starts increasing after epoch ~6, indicating overfitting in later epochs — a future improvement would be early stopping.

---

## Inference: Greedy vs Beam Search

### Greedy Search
At each step, pick the highest-probability word. Fast but suboptimal.

### Beam Search (k=3)
Maintain the top-3 partial sequences at every step, score by normalised log-probability. Produces more fluent, globally coherent captions.

```python
def generate_caption(image_feature, method='beam', beam_width=3, max_length=50):
    # ... see full code on GitHub
```

---

## Results

Evaluated on ~2,783 test images using beam search:

| Metric | Score |
|--------|-------|
| **BLEU-4** | 0.0267 |
| ROUGE-1 Precision | 0.2237 |
| ROUGE-1 Recall | 0.2274 |
| **ROUGE-1 F1** | 0.2127 |

The BLEU-4 of ~0.027 is consistent with attention-free single-layer LSTM baselines on Flickr30k. The ROUGE-1 F1 of 0.21 shows meaningful word overlap between predictions and references.

---

## Deployment: Gradio App

The model is wrapped in a simple **Gradio** interface — upload any image, get a caption instantly.

```python
demo = gr.Interface(
    fn=predict_image_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Neural Storyteller - Image Captioning"
)
demo.launch(share=True)
```

---

## Key Takeaways

1. **Offline feature extraction** — extracting ResNet features once and caching them reduced epoch time dramatically.
2. **Vocabulary frequency thresholding** (min 5 occurrences) keeps vocab manageable and reduces noise.
3. **Beam search > Greedy** — even with beam width 3, caption quality noticeably improves.
4. **Overfitting is real** — without attention or regularization, the model memorizes training captions. Early stopping at epoch 6 would yield the best validation performance.

---

## What's Next?

- Add **Bahdanau Attention** to let the decoder focus on image regions
- Use a **Transformer decoder** instead of LSTM
- **Fine-tune ResNet** jointly with the decoder (end-to-end training)
- Deploy permanently to **Hugging Face Spaces**

---

## References

1. Vinyals et al. (2015). *Show and Tell: A Neural Image Caption Generator.* CVPR.
2. Young et al. (2014). *From image descriptions to visual denotations: Flickr30k Entities.*
3. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

---

*Full code: https://github.com/Sananoor12/GenAi*
