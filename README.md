
# Fine-Tuning DistilBERT on Poem Sentiment Dataset

![Hugging Face](/Users/sovitnayak/Desktop/Finetuning-DistilBert/img/hf.png)

This repository provides an implementation of fine-tuning the DistilBERT model for sentiment classification using the Poem Sentiment Dataset. The notebook demonstrates data preprocessing, training, evaluation, and feature extraction.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Key Features](#key-features)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [Acknowledgments](#acknowledgments)
9. [Contributing](#contributing)

## Overview

This project fine-tunes the lightweight and efficient DistilBERT model on the **Poem Sentiment Dataset**. The goal is to classify poems into sentiment categories (e.g., positive, negative, neutral). DistilBERT's reduced size and efficiency make it ideal for such fine-tuning tasks.

---

## Dataset

The **Poem Sentiment Dataset** consists of poems labeled with sentiment categories. The dataset includes text data and corresponding labels, making it suitable for supervised learning tasks.

- **Labels**: Positive, Negative, Neutral
- **Source**: Specify the source if publicly available (e.g., Kaggle, Hugging Face Datasets).

Before running the notebook, ensure the dataset is properly loaded and formatted.

---

## Requirements

Before running the notebook, ensure you have the following libraries installed:

- Python 3.7+
- PyTorch
- Transformers
- Datasets (Hugging Face)
- UMAP (for dimensionality reduction)
- scikit-learn

Install the required packages using:

```bash
pip install torch transformers datasets umap-learn scikit-learn
```

---

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/sovit-nayak/Finetuning-DistilBert.git
    cd Finetuning-DistilBert
    ```

2. Open the notebook:
    ```bash
    jupyter notebook Finetuning_LLM_DistilBert.ipynb
    ```

3. Follow the instructions in the notebook to:
   - Load and preprocess the Poem Sentiment Dataset.
   - Tokenize the data using the Hugging Face tokenizer.
   - Fine-tune the DistilBERT model on the dataset.
   - Evaluate the model using metrics like accuracy and F1-score.
   - Visualize hidden states and sentiment separation using UMAP.

---

## Key Features

- **Tokenization:** Efficient preprocessing using Hugging Face's `tokenizer`.
- **Fine-Tuning:** Train DistilBERT on the Poem Sentiment Dataset with options for hyperparameter tuning.
- **Feature Extraction:** Extract hidden states for downstream tasks.
- **Dimensionality Reduction:** Use UMAP to visualize sentiment separation in high-dimensional space.
- **Evaluation:** Measure model performance using metrics like accuracy, precision, recall, and F1-score.

---

## Results

After fine-tuning, the model demonstrates its ability to classify the sentiment of poems effectively.

- **Training Accuracy**: 86% 
- **Validation Accuracy**: 86% 
- **F1-Score**: 85% 

Include any additional results like confusion matrices or training curves.

---

## Visualizations

Visualizations play a key role in understanding model performance. Include images generated during the notebook run.

### UMAP Visualization of Hidden States
![UMAP Visualization](/Users/sovitnayak/Desktop/Finetuning-DistilBert/img/umap.png)

### Training Loss and Validation Accuracy
![Training Metrics](/Users/sovitnayak/Desktop/Finetuning-DistilBert/img/acc.png)

---

## Acknowledgments

This project leverages the following open-source libraries and datasets:

- Hugging Face Transformers
- PyTorch
- UMAP-learn
- Poem Sentiment Dataset

Special thanks to the contributors of these libraries and datasets.

