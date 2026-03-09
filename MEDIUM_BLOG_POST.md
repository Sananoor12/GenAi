==========================================================================
HOW TO PUBLISH THIS ON MEDIUM
==========================================================================
1. Go to medium.com → click your profile → "Write"
2. Copy each section below and paste into Medium's editor
3. For every [UPLOAD IMAGE: filename] line:
   - Click the "+" icon that appears on the left of a new line
   - Choose "Image" → upload the PNG from the report_images/ folder
   - Add the caption text below it
4. After publishing, copy the URL and update the Word report + README
==========================================================================

---TITLE (paste this as the big title at the top)---

Neural Storyteller: Building an Image Captioning System with CNN-LSTM


---SUBTITLE (click "Add a subtitle" below the title)---

How I trained a deep learning model to look at photos and write captions — using ResNet-50 + LSTM on Flickr30k


==========================================================================
START COPYING ARTICLE BODY FROM HERE
==========================================================================

What if a machine could look at a photograph and describe it in plain English, just like a human would?

That is exactly what image captioning does. It sits at the intersection of Computer Vision and Natural Language Processing, and in this article I will walk through how I built Neural Storyteller — an end-to-end image captioning system trained on the Flickr30k dataset and deployed as a live web app.

GitHub: https://github.com/Sananoor12/GenAi

---

The Dataset

The Flickr30k dataset contains 31,783 photographs sourced from Flickr, each with five human-written captions — over 158,000 captions in total. I split the data into:

• Train: 28,000 images
• Validation: 1,000 images
• Test: ~2,783 images

Each image has captions like: "A dog in a swimming pool swims toward somebody we cannot see" — diverse, detailed, and written by different people.

---

The Architecture

The model follows the classic encoder-decoder design from the paper "Show and Tell" (Vinyals et al., 2015).

Encoder — ResNet-50

A pre-trained ResNet-50 extracts visual features from each image. I remove the final classification layer, leaving a 2048-dimensional feature vector per image. These features are extracted once and cached — this saves enormous GPU time during training.

A single linear layer then projects the 2048-d vector down to 512 dimensions and initialises the LSTM's hidden state.

Decoder — LSTM

A one-layer LSTM with 512 hidden units generates the caption word by word. Word embeddings (size 256) are learned from scratch. A vocabulary is built from the training captions using a minimum frequency threshold of 5, keeping only words that appear at least 5 times.

---

Training

I trained for 20 epochs on dual NVIDIA T4 GPUs on Kaggle (CUDA 12.1). Each epoch took about 63 seconds.

Hyperparameters:
• Embedding size: 256
• Hidden size: 512
• Learning rate: 3e-4 (Adam)
• Batch size: 64
• Loss: CrossEntropyLoss (padding tokens ignored)

[UPLOAD IMAGE: loss_curve.png]
Caption: Training vs Validation Loss over 20 epochs. Best model checkpoint is around epoch 6 before overfitting begins.

The training loss drops steadily from 3.37 to 1.64. However, validation loss starts increasing after epoch 6 — a classic sign of overfitting. The best checkpoint should be saved at epoch 6. Adding attention or dropout would likely delay this.

---

Inference: Greedy vs Beam Search

Greedy Search picks the highest-probability word at every step. It is fast but can produce repetitive or suboptimal captions.

Beam Search (k=3) keeps the top 3 partial sequences alive at each step and picks the globally best-scoring sequence at the end. It consistently produces more fluent and meaningful captions and was used for all evaluation.

---

Sample Results

Here are five random test images with their ground truth captions and what the model generated:

[UPLOAD IMAGE: sample1.png]
Caption: Ground Truth: "The three cats, two white with calico accents and a gray tabby, are laying in the brown grass." | Generated: "taking a nap from her dog"

[UPLOAD IMAGE: sample2.png]
Caption: Ground Truth: "A dog in a swimming pool swims toward somebody we cannot see." | Generated: "water into a dog's face"

[UPLOAD IMAGE: sample3.png]
Caption: Ground Truth: "Girl in club DJ both showing camera the cover of Michael Jackson's Thriller." | Generated: "use a kiss on their makeup"

[UPLOAD IMAGE: sample4.png]
Caption: Ground Truth: "A golfer wearing black pants swings at a golf ball while three people look on." | Generated: "practice in a field out of a baseball game"

[UPLOAD IMAGE: sample5.png]
Caption: Ground Truth: "A motorcycle racer riding a yellow motorcycle with number 33 is passing a Kawasaki banner." | Generated: "on a motorcycle on a yellow motorcycle"

The model captures rough themes (animals, sports, vehicles) but struggles with fine-grained detail. This is expected for an attention-free single-layer LSTM.

---

Quantitative Results

Evaluated on the full test set using beam search:

• BLEU-4: 0.0267
• ROUGE-1 Precision: 0.2237
• ROUGE-1 Recall: 0.2274
• ROUGE-1 F1: 0.2127

These scores are consistent with baseline attention-free LSTM models on Flickr30k. The ROUGE-1 F1 of 0.21 shows meaningful word overlap between predictions and references.

---

Deployment with Gradio

The model is wrapped in a Gradio interface. Users upload any image and get a generated caption instantly. A public share link is generated via Gradio's tunnelling service — no server setup required.

The app is titled "Neural Storyteller - Image Captioning" and runs directly inside the Kaggle notebook environment.

---

Key Takeaways

1. Offline feature extraction is worth it — extracting ResNet-50 features once and caching them cut per-epoch time dramatically.

2. Vocabulary thresholding matters — using only words with frequency ≥ 5 keeps the model manageable and reduces noise.

3. Beam search beats greedy — even with beam width 3, captions are noticeably more fluent and complete.

4. Overfitting is the main bottleneck — without attention or dropout, the model memorises training captions after epoch 6. Early stopping here is essential.

---

What is Next

• Add Bahdanau Attention to let the decoder focus on specific image regions
• Use a Transformer decoder instead of LSTM
• Fine-tune the ResNet backbone jointly with the decoder
• Deploy permanently on Hugging Face Spaces

---

References

1. Vinyals et al. (2015). Show and Tell: A Neural Image Caption Generator. CVPR.
2. Young et al. (2014). From image descriptions to visual denotations: Flickr30k Entities.
3. He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.

Full code: https://github.com/Sananoor12/GenAi
