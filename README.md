# Drug-Related Text Classification Model

A fine-tuned BERT model designed for classifying drug-related text with high precision. This model is trained on synthetic data and optimized for identifying drug-related content in textual data.

**Model ID:** `ankoor123/drug2`  
**Platform:** Hugging Face Hub  
**Base Model:** BERT  
**Task:** Text Classification  

---

## Overview

This model fine-tunes BERT to accurately classify text into drug-related and non-drug-related categories. Built on synthetic training data, it provides robust classification performance for content moderation, medical text analysis, and research applications.

### Key Features

- **Fine-tuned BERT architecture** for domain-specific performance
- **Synthetic training data** for consistent, unbiased classification
- **Production-ready** with optimized inference speed
- **Easy integration** via Hugging Face Transformers library
- **Detailed documentation** for implementation and evaluation

---

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | `bert-base-uncased` |
| **Model Type** | Text Classification |
| **Training Data** | Synthetic drug-related and non-drug-related text |
| **Languages** | English |
| **Input** | Raw text (sentences or documents) |
| **Output** | Binary classification (Drug/Non-Drug) |
| **Framework** | PyTorch |

---

## Usage

### Quick Start

```python
from transformers import pipeline

# Load the classifier
classifier = pipeline("text-classification", model="ankoor123/drug2")

# Classify text
result = classifier("This medication treats hypertension effectively")
print(result)
# Output: [{'label': 'LABEL_1', 'score': 0.95}]
```

### Advanced Usage with Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ankoor123/drug2")
model = AutoModelForSequenceClassification.from_pretrained("ankoor123/drug2")

# Prepare input
text = "The patient was prescribed ibuprofen for pain relief"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()

print(f"Predicted class: {predicted_class_id}, Confidence: {confidence:.4f}")
```

### Batch Processing

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="ankoor123/drug2")

texts = [
    "Aspirin is commonly used for headaches",
    "The weather is sunny today",
    "Metformin helps manage blood sugar levels"
]

results = classifier(texts, batch_size=8)
for text, result in zip(texts, results):
    print(f"{text} -> {result}")
```

---

## Input/Output Specification

### Input Format
- **Type:** String (raw text)
- **Length:** Up to 512 tokens (BERT limit)
- **Language:** English
- **Format:** Free-form text (sentences, paragraphs, or documents)

### Output Format
- **Label:** `LABEL_0` (Non-Drug) or `LABEL_1` (Drug)
- **Score:** Confidence probability (0.0 to 1.0)

**Example Output:**
```json
{
  "label": "LABEL_1",
  "score": 0.9823
}
```

---

## Training Data

The model was trained on **synthetic drug-related text** generated to represent:

- **Drug mentions** (prescription drugs, over-the-counter medications, illicit substances)
- **Medical contexts** (dosage information, side effects, uses)
- **Non-drug contexts** (general topics, unrelated discussions)

### Data Characteristics
- Balanced class distribution
- Diverse vocabulary and sentence structures
- Real-world language patterns
- Synthetic generation ensures privacy and consistency

---

## Performance

The model achieves strong classification performance on held-out synthetic validation sets:

- **Accuracy:** High precision for both drug and non-drug classifications
- **Speed:** Fast inference suitable for real-time applications
- **Robustness:** Effective across varying text lengths and writing styles

*For detailed performance metrics, refer to the original GitHub repository.*

---

## Limitations

- **Synthetic Training Data:** Performance on real-world data may vary depending on domain-specific terminology
- **Language:** Optimized for English text; performance on other languages not tested
- **Context:** Single-text classification; does not consider broader document context in multi-turn conversations
- **Bias:** Inherits limitations from BERT and training data distribution

---

## Use Cases

✅ **Recommended Applications:**
- Content moderation and filtering
- Medical text analysis and categorization
- Drug-related mention detection in research papers
- Social media monitoring for health-related content
- Chatbot responses to drug-related queries

⚠️ **Not Recommended For:**
- Clinical decision-making without human review
- Legal or regulatory compliance (use domain-validated models)
- Real-time identification of illicit drug networks

---

## Installation

### Requirements
```bash
pip install transformers torch
```

### Setup
```bash
# Install the Hugging Face transformers library
pip install transformers>=4.30.0

# Optional: for GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Model Architecture

The model is based on BERT with the following modifications:

- **Encoder:** BERT base encoder layers
- **Classification Head:** Dropout + Linear layer (2 output classes)
- **Tokenizer:** BERT wordpiece tokenizer
- **Pooling:** [CLS] token representation for classification

---

## License

This model is made available under the [MIT License](https://opensource.org/licenses/MIT). Please ensure compliance with licensing requirements when using this model.

---

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@model{drug_classification_2024,
  author = {Ankoor},
  title = {Fine-Tuned BERT for Drug-Related Text Classification},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ankoor123/drug2}}
}
```

---

## Repository & Resources

- **Hugging Face Model:** https://huggingface.co/ankoor123/drug2
- **GitHub Repository:** https://github.com/Ankoor11/Fine-Tuning-BERT-for-Drug-Related-Text-Classification
- **Documentation:** See repository for training scripts, data generation, and evaluation code

---

## Support & Issues

For issues, questions, or suggestions:

1. Check existing issues on the [GitHub repository](https://github.com/Ankoor11/Fine-Tuning-BERT-for-Drug-Related-Text-Classification)
2. Create a new issue with detailed description
3. Include example text and expected vs. actual output
4. For Hugging Face-specific issues, use the model card discussion tab

---

## Disclaimer

This model is provided for research and educational purposes. Users are responsible for ensuring compliance with applicable laws and regulations when using this model. The authors assume no liability for misuse or harmful applications.

---

## Acknowledgments

Built using the [Hugging Face Transformers](https://github.com/huggingface/transformers) library and trained on synthetically generated data.
