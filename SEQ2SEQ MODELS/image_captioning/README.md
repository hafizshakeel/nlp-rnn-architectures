# Image Captioning with CNN-LSTM

An implementation of image captioning using a CNN-RNN architecture that learns to generate natural language descriptions of images.

## Architecture Overview

This implementation consists of two main components:

1. **Encoder (CNN)**
   - Uses pretrained InceptionV3 model
   - Replaces final FC layer to match embedding dimension
   - Features extraction with fine-tuning option
   - Dropout for regularization

2. **Decoder (LSTM)**
   - Word embedding layer for caption tokens
   - LSTM layers for sequence generation
   - Linear layer to map to vocabulary size
   - Teacher forcing during training

## Implementation Details

- `image_caption_model.py`: Core model architecture
- `load_dataset_img_caption.py`: Data loading and preprocessing
- `train_img_caption.py`: Training loop and configuration
- `utils.py`: Helper functions for checkpointing and evaluation

## Dataset

Uses the Flickr8k dataset containing:
- 8,000 images
- 5 captions per image
- Download from Kaggle: [Flickr8k Dataset](https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb)

## Usage

1. Download and prepare dataset:
   ```bash
   # Place images in flickr8k/images/
   # Place captions.txt in flickr8k/
   ```

2. Train the model:
   ```python
   python train_img_caption.py
   ```

## Visual Aids
Refer to architecture diagrams in `architecure_diagram/` folder:
- Image Captioning.png: Overall architecture
- Text_preprocessing.png: Caption preprocessing
- visual_text_preprocess.png: Combined visual-text pipeline