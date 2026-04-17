# AI Content Moderation System - Enhanced Accuracy & Training Guide

## 🚀 Major Improvements

### 1. **Enhanced Detection Accuracy**
- **Expanded Label Taxonomy**: 13 detailed categories instead of 3 generic ones
  - Profanity: `explicit_profanity`, `mild_profanity`, `hate_speech`, `racial_slur`, `sexual_content`
  - Toxicity: `harassment`, `threats`, `bullying`, `discrimination`, `violence`, `self_harm`
  - Contextual: `sarcasm_detected`, `spam`, `safe`

- **Severity-Weighted Scoring**: Each category has a severity weight (0.0-1.0) that adjusts detection sensitivity
  - Critical: hate_speech (1.0), racial_slur (1.0), threats (0.95)
  - High: violence (0.9), self_harm (0.9), discrimination (0.85)
  - Medium: explicit_profanity (0.8), harassment (0.8), bullying (0.75)
  - Lower: sexual_content (0.7), mild_profanity (0.4), spam (0.3)

- **Adaptive Thresholding**: High-severity categories use lower thresholds automatically
- **Critical Overrides**: Auto-flag for scores >0.75 on critical categories regardless of user threshold

### 2. **Hybrid Ensemble AI**
- **Dual-Model Architecture**: Combines local Transformers.js (mDeBERTa-v3) + OpenAI Moderation API
- **Adaptive Weighting**: Higher severity categories get more weight from OpenAI (60-80%)
- **Sentiment Analysis**: Additional distilbert model for contextual understanding
- **Multilingual Support**: Works with Tagalog, Chinese, English, and 100+ languages

### 3. **Training & Continuous Improvement**
- **Data Collection**: Automatically records all analyses with AI predictions and user decisions
- **Export Functionality**: Download training data as JSON for model fine-tuning
- **Import Capability**: Load previously collected training data
- **Memory Management**: Keeps last 100 examples in browser memory

---

## 📚 How to Train Your AI Model

### Step 1: Collect Training Data

1. **Use the Interface**: Run multiple content analyses through the web interface
2. **Automatic Recording**: Every analysis is automatically saved with:
   - Input text (truncated to 500 chars)
   - AI prediction scores for all categories
   - Final decision (flagged/approved)
   - Analysis method used (local/openai/hybrid)
   - Timestamp

3. **Monitor Collection**: Watch the counter on the "Export Training Data" button

### Step 2: Export Training Data

```javascript
// In browser console or click the export button
const data = exportTrainingData();
```

This downloads a JSON file like:
```json
{
  "version": "1.0",
  "exportedAt": "2024-01-15T10:30:00.000Z",
  "totalExamples": 100,
  "data": [
    {
      "timestamp": "2024-01-15T10:25:00.000Z",
      "text": "Example toxic comment...",
      "aiPrediction": {
        "hate_speech": 0.85,
        "harassment": 0.72,
        "safe": 0.15
      },
      "userDecision": "flagged",
      "analysisMethod": "hybrid"
    }
  ]
}
```

### Step 3: Prepare Dataset for Fine-Tuning

Convert your exported data into a format suitable for training:

#### For Hugging Face Transformers:
```python
import json
from datasets import Dataset

# Load exported data
with open('moderation-training-data.json', 'r') as f:
    data = json.load(f)

# Convert to classification format
training_examples = []
for item in data['data']:
    # Create labels based on user decision + AI predictions
    if item['userDecision'] == 'flagged':
        # Find highest scoring violation
        max_category = max(
            [(k, v) for k, v in item['aiPrediction'].items() if k != 'safe'],
            key=lambda x: x[1]
        )[0]
        label = max_category
    else:
        label = 'safe'
    
    training_examples.append({
        'text': item['text'],
        'label': label
    })

# Create dataset
dataset = Dataset.from_list(training_examples)
dataset.save_to_disk('moderation_dataset')
```

#### For OpenAI Fine-tuning:
```python
import json

with open('moderation-training-data.json', 'r') as f:
    data = json.load(f)

training_data = []
for item in data['data']:
    example = {
        "messages": [
            {"role": "system", "content": "You are a content moderation assistant. Classify text as 'safe' or identify the primary violation category."},
            {"role": "user", "content": item['text']},
            {"role": "assistant", "content": item['userDecision']}
        ]
    }
    training_data.append(example)

# Save as JSONL
with open('openai_training.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')
```

### Step 4: Fine-Tune Your Model

#### Option A: Hugging Face mDeBERTa Fine-Tuning

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load tokenizer and model
model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=13,  # Number of categories
    id2label={i: label for i, label in enumerate([
        'explicit_profanity', 'mild_profanity', 'hate_speech', 'racial_slur',
        'sexual_content', 'harassment', 'threats', 'bullying', 'discrimination',
        'violence', 'self_harm', 'sarcasm_detected', 'spam', 'safe'
    ])},
    label2id={label: i for i, label in enumerate([
        'explicit_profanity', 'mild_profanity', 'hate_speech', 'racial_slur',
        'sexual_content', 'harassment', 'threats', 'bullying', 'discrimination',
        'violence', 'self_harm', 'sarcasm_detected', 'spam', 'safe'
    ])}
)

# Load dataset
dataset = load_from_disk('moderation_dataset')

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./moderation_model',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model
trainer.save_model('./moderation_model')
tokenizer.save_pretrained('./moderation_model')
```

#### Option B: OpenAI Fine-tuning

```bash
# Upload training file
openai api files.create -f openai_training.jsonl -p fine-tune

# Create fine-tuning job
openai api fine_tunes.create -t <file-id> -m gpt-3.5-turbo

# Monitor progress
openai api fine_tunes.follow -i <fine-tune-id>
```

### Step 5: Deploy Your Custom Model

#### For Local Deployment (Transformers.js):
```javascript
// Convert to ONNX format
python -m transformers.onnx --model=./moderation_model ./moderation_model_onnx

// Use in your web app
import { pipeline } from '@xenova/transformers';
const classifier = await pipeline('zero-shot-classification', './moderation_model_onnx');
```

#### For Server Deployment:
```javascript
// server.js - Add custom model endpoint
app.post('/api/moderate-custom', async (req, res) => {
    const text = req.body.input;
    const result = await customClassifier(text);
    res.json(result);
});
```

---

## 🎯 Best Practices for Training

### Data Quality
1. **Diverse Examples**: Include content in multiple languages (Tagalog, Chinese, English, etc.)
2. **Balanced Classes**: Ensure roughly equal representation of all violation types
3. **Edge Cases**: Include borderline cases to improve threshold calibration
4. **Context Variety**: Different contexts (social media, comments, chats, forums)

### Minimum Dataset Sizes
- **Basic Improvement**: 100-500 examples
- **Good Performance**: 500-2,000 examples
- **Production Quality**: 2,000-10,000+ examples

### Continuous Learning Loop
1. Deploy initial model
2. Collect user corrections/overrides
3. Export new training data monthly
4. Retrain model quarterly
5. A/B test new versions before full deployment

---

## 🔧 Advanced Configuration

### Adjust Severity Weights
Edit `CATEGORY_SEVERITY` in `index.html`:
```javascript
const CATEGORY_SEVERITY = {
    'hate_speech': 1.0,      // Increase for stricter detection
    'mild_profanity': 0.3,   // Decrease for more leniency
    // ... adjust others
};
```

### Modify Ensemble Weights
In `runHybridAnalysis()`:
```javascript
const openaiWeight = 0.5 + (severity * 0.3); // Adjust multiplier
```

### Custom Critical Thresholds
```javascript
const criticalLabels = ['hate_speech', 'racial_slur', 'threats', 'self_harm'];
const CRITICAL_THRESHOLD = 0.75; // Lower for stricter, higher for lenient
```

---

## 📊 Performance Metrics to Track

After deploying your trained model, monitor:
- **Precision**: % of flagged content that truly violates policies
- **Recall**: % of actual violations caught by the system
- **F1 Score**: Balance between precision and recall
- **False Positive Rate**: Innocent content incorrectly flagged
- **False Negative Rate**: Violating content missed
- **Multilingual Accuracy**: Performance across different languages

---

## 🛠️ Troubleshooting

### Low Accuracy on Specific Categories
- **Solution**: Collect more training examples for underperforming categories
- **Tip**: Use data augmentation (paraphrasing, translation back-and-forth)

### Too Many False Positives
- **Solution**: Increase threshold slider, reduce severity weights for sensitive categories
- **Tip**: Add more "safe" examples that resemble violating content

### Slow Performance
- **Solution**: Reduce model size, implement caching, use server-side inference
- **Tip**: Pre-filter obvious cases with simple rules before AI analysis

---

## 📞 Support & Resources

- **Hugging Face Documentation**: https://huggingface.co/docs/transformers
- **OpenAI Fine-tuning Guide**: https://platform.openai.com/docs/guides/fine-tuning
- **Transformers.js**: https://huggingface.co/docs/transformers.js
- **mDeBERTa-v3 Model**: https://huggingface.co/microsoft/mdeberta-v3-base

---

*Last Updated: January 2024*
*Version: 2.0 - Enhanced Accuracy & Training Edition*
