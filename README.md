# Neural Storyteller — Image Captioning with CNN-LSTM

**GenAI Assignment 1 | Student IDs: 22F-3276 | 22F-3377**

## Overview
End-to-end image captioning system trained on the Flickr30k dataset using a ResNet-50 encoder and LSTM decoder with beam search inference.

## Files
| File | Description |
|------|-------------|
| `ai-ass01-22f-3276-22f-3377.ipynb` | Main Jupyter notebook (full pipeline) |
| `GenAI_Assignment1_Report_22F-3276_22F-3377.docx` | Assignment report (Word) |
| `MEDIUM_BLOG_POST.md` | Medium blog article |
| `GenAI_Assignment01.pdf` | Original assignment brief |

## Architecture
- **Encoder**: Pre-trained ResNet-50 → 2048-dim features → Linear(2048, 512)
- **Decoder**: Single-layer LSTM (hidden=512, embed=256) with beam search (k=3)
- **Dataset**: Flickr30k — 31,783 images, 5 captions each

## Results

| Metric | Score |
|--------|-------|
| BLEU-4 | 0.0267 |
| ROUGE-1 F1 | 0.2127 |

## Medium Blog
https://medium.com/@sananoor12/neural-storyteller-building-an-image-captioning-system-with-cnn-lstm

## How to Run
1. Add Flickr30k dataset to Kaggle input
2. Run the notebook cells in order
3. Gradio app launches at the end with a public share link
