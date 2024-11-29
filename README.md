# Text-to-Math Translation (Part B) - Inference

This repository contains models and scripts for running inference on text-to-math translation problems using Seq2Seq and BERT-based models.
Dataset at: https://www.kaggle.com/datasets/skadoosh64/dataset/data

---

## Features

### Supported Models
1. **LSTM-Based Models**:
   - `lstm_lstm`: Basic Seq2Seq model without attention.
   - `lstm_lstm_attn`: Seq2Seq model with attention mechanism.
2. **BERT-Based Models**:
   - `bert_lstm_attn_frozen`: BERT-based encoder-decoder with frozen BERT parameters.
   - `bert_lstm_attn_tuned`: BERT-based encoder-decoder with fine-tuned BERT parameters.

### Beam Search
- Beam sizes supported: `1`, `10`, `20` (default: `10`).

---

## Running Inference

### Command-Line Arguments
The inference script accepts the following arguments:
- `--model_file`: Path to the trained model file (required).
- `--beam_size`: Beam size for decoding (default: `10`). Choose from `1`, `10`, or `20`.
- `--model_type`: Specify the model architecture to use. Options:
  - `lstm_lstm`
  - `lstm_lstm_attn`
  - `bert_lstm_attn_frozen`
  - `bert_lstm_attn_tuned` (recommended for best performance).
- `--test_data_file`: Path to the JSON file containing text descriptions to translate (required).

### Input File Format
The input file should be a JSON file with a list of problems. Example:
```json
[
  "Calculate the sum of n0 and n1.",
  "Find the square root of n2."
]
```

### Example Usage

To run the inference script with a BERT-based model and a beam size of 10, use the following command:

```bash
python infer.py \
    --model_file ./models/bert_lstm_attn_tuned.pth \
    --beam_size 10 \
    --model_type bert_lstm_attn_tuned \
    --test_data_file ./data/test_problems.json
```
