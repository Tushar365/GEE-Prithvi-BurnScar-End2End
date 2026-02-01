# Prithvi EO 2.0 Model Comparison Notebook

Comprehensive evaluation and comparison between pretrained foundation model and your fine-tuned burn severity model.

**Author**: Tushar Thokdar

## 📋 Overview

This notebook provides a complete analysis comparing:

1. **Foundation Model**: Pretrained Prithvi EO 2.0 (trained on general satellite tasks)
2. **Fine-tuned Model**: Your specialized model for burn severity classification

**Outputs**:

- ✅ Quantitative metrics (F1, accuracy, precision, recall)
- ✅ Per-class performance breakdown
- ✅ Confusion matrices
- ✅ Side-by-side visual predictions
- ✅ Automated verdict on fine-tuning effectiveness

## 🎯 Purpose

**Why compare?**

- Validate that fine-tuning actually improved performance
- Understand which severity classes benefited most
- Identify remaining weaknesses
- Justify model deployment decisions

## 🚀 Quick Start

### Prerequisites

```python
# Required imports (should already be in your training notebook)
from your_training_module import (
    FireSeverityDataModule,
    PrithviFireSegmentation,
    compute_class_weights
)
```

### Basic Usage

```python
from prithvi_model_comparison import main, ComparisonConfig

# Update paths
ComparisonConfig.DATA_DIR = "/path/to/prithvi_data_final_withdelta"
ComparisonConfig.FINETUNED_CHECKPOINT = "/path/to/best_model.ckpt"
ComparisonConfig.NUM_FRAMES = 3  # 3 for delta, 2 for baseline

# Run complete comparison
main()
```

## 📊 What You'll Get

### 1. Metric Comparison Table

```
📊 METRIC COMPARISON
================================================================
Metric               Foundation    Fine-tuned    Improvement
----------------------------------------------------------------
Accuracy             0.3586        0.6993        +34.07%
Macro F1             0.1160        0.6218        +50.58%
Weighted F1          0.2035        0.7015        +49.80%
Burned F1            0.0133        0.5553        +54.20%

Class                Foundation    Fine-tuned    Improvement
----------------------------------------------------------------
Unburned             0.5289        0.9397        +41.08%
Low Severity         0.0000        0.5528        +55.28%
Moderate-Low         0.0000        0.4461        +44.61%
Moderate-High        0.0000        0.4351        +43.51%
High Severity        0.0509        0.7352        +68.43%
```

### 2. Classification Reports

Detailed precision, recall, F1-score for each class for both models.

### 3. Confusion Matrices

Side-by-side heatmaps showing:

- Where foundation model gets confused
- How fine-tuning fixes these issues
- Remaining problem areas

### 4. Visual Comparisons

For each sample:

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Post-Fire  │   Ground    │ Foundation  │ Fine-tuned  │
│     RGB     │    Truth    │   Predict   │   Predict   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 5. Automated Verdict

```
💡 VERDICT
================================================================

✅ Fine-tuning provides HUGE improvement!
Overall improvement: +50.58% (Macro F1)

📈 Best improvement: High Severity (+68.43%)

🔍 INTERPRETATION
================================================================
The pretrained foundation model was trained on general satellite imagery
tasks (crop classification, land cover mapping) but NOT on burn severity.

Your fine-tuning specialized it for fire severity mapping, which explains
the performance difference (especially for burn severity classes 1-4).

✅ Your fine-tuning was very effective!
```

## ⚙️ Configuration

### Essential Settings

```python
class ComparisonConfig:
    # Paths - UPDATE THESE
    DATA_DIR = "/content/data/prithvi_data_final_withdelta"
    FINETUNED_CHECKPOINT = "/content/checkpoints/best_model.ckpt"

    # Model parameters - MUST MATCH YOUR TRAINING
    NUM_CLASSES = 5
    NUM_FRAMES = 3  # 3 for delta dataset, 2 for baseline
    IGNORE_INDEX = 255

    # Evaluation settings
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    VAL_SPLIT = 0.2

    # Visualization
    NUM_SAMPLES = 6  # Number of visual examples
    FIGURE_DPI = 150
```

### Class Configuration

```python
# Severity classes
CLASS_NAMES = [
    'Unburned',
    'Low Severity',
    'Moderate-Low',
    'Moderate-High',
    'High Severity'
]

# Visualization colors (green → red gradient)
CLASS_COLORS = [
    '#1a9850',  # Green
    '#fee08b',  # Yellow
    '#fdae61',  # Light Orange
    '#f46d43',  # Orange
    '#a50026'   # Dark Red
]
```

## 📈 Understanding Results

### Macro F1 Score

**What it means**:

- Average F1 across all classes (equal weight)
- Best metric for imbalanced datasets
- Range: 0.0 (worst) to 1.0 (perfect)

**Interpretation**:

- < 0.5: Poor performance
- 0.5-0.7: Moderate performance
- 0.7-0.85: Good performance
- \> 0.85: Excellent performance

### Burned F1 Score

**What it means**:

- F1 averaged over burn severity classes only (1-4)
- Excludes "Unburned" class
- More relevant for fire mapping applications

**Why it matters**:

- Unburned areas are easier to classify
- Burned F1 shows how well the model distinguishes severity levels
- Critical for damage assessment

### Per-Class F1

**Look for**:

- Low F1 on specific classes → that class needs work
- Balanced F1 across classes → good generalization
- High F1 on extreme classes (Unburned, High) → model might be oversimplifying

### Confusion Matrix

**How to read**:

- **Diagonal**: Correct predictions (higher is better)
- **Off-diagonal**: Misclassifications

**Common patterns**:

- Foundation model: Often predicts neighboring classes
- Fine-tuned model: Sharper diagonal, fewer off-diagonal errors

## 🔍 Interpreting the Verdict

### Excellent (>20% improvement)

```
🎉 Fine-tuning provides HUGE improvement!
```

- Model learned burn-specific features very well
- Ready for deployment
- Consider publishing results

### Very Good (10-20% improvement)

```
✅ Fine-tuning provides significant improvement.
```

- Strong performance gain
- Model is production-ready
- May benefit from minor hyperparameter tuning

### Good (5-10% improvement)

```
👍 Fine-tuning provides moderate improvement.
```

- Decent gain from fine-tuning
- Consider more epochs or data augmentation
- Still valuable improvement

### Marginal (0-5% improvement)

```
⚠️  Fine-tuning provides minimal improvement.
```

- Check: learning rate, epochs, data quality
- Foundation model may be sufficient
- Try different decoder or loss function

### Poor (negative improvement)

```
❌ Fine-tuning hurt performance!
```

- **Something is wrong** - investigate:
  - Data labels quality
  - Class imbalance handling
  - Learning rate (too high?)
  - Overfitting (too many epochs?)

## 📊 Visualization Guide

### Per-Class F1 Comparison

**Bar chart showing**:

- Blue bars: Foundation model
- Green bars: Fine-tuned model
- Values labeled on bars

**Look for**:

- Green bars consistently higher → successful fine-tuning
- Similar heights → fine-tuning didn't help much
- Blue higher than green → something wrong

### Confusion Matrices

**Left (Foundation)**: Before fine-tuning
**Right (Fine-tuned)**: After fine-tuning

**Good signs**:

- Darker diagonal on fine-tuned
- Less confusion between neighboring classes
- Reduced off-diagonal values

**Warning signs**:

- Both matrices look similar → minimal learning
- New confusions appear → overfitting or data issues

### Side-by-Side Predictions

**4-panel layout**:

1. **Post-Fire RGB**: Actual satellite image
2. **Ground Truth**: Labeled severity
3. **Foundation**: Pretrained model prediction
4. **Fine-tuned**: Your model prediction

**What to check**:

- Fine-tuned closer to ground truth?
- Spatial coherence (smooth regions)?
- Correct severity gradients?

## 🛠️ Advanced Usage

### Custom Metrics

```python
from prithvi_model_comparison import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(config)

# Evaluate on specific subset
evaluator.evaluate_dataloader(model, subset_loader)

# Get custom metrics
metrics = evaluator.compute_metrics()

# Add your own analysis
high_severity_mask = evaluator.targets == 4
high_severity_accuracy = (
    evaluator.preds[high_severity_mask] == 4
).mean()
print(f"High severity recall: {high_severity_accuracy:.4f}")
```

### Save Results

```python
import json

# Export metrics to JSON
results = main()

with open('comparison_results.json', 'w') as f:
    json.dump({
        'foundation': {
            'macro_f1': float(results['foundation']['metrics']['macro_f1']),
            'burned_f1': float(results['foundation']['metrics']['burned_f1']),
            'per_class_f1': results['foundation']['metrics']['per_class_f1'].tolist()
        },
        'finetuned': {
            'macro_f1': float(results['finetuned']['metrics']['macro_f1']),
            'burned_f1': float(results['finetuned']['metrics']['burned_f1']),
            'per_class_f1': results['finetuned']['metrics']['per_class_f1'].tolist()
        }
    }, f, indent=2)
```

### Generate Report

```python
# Save visualizations
import matplotlib.pyplot as plt

# After running main(), save all figures
for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(f'comparison_figure_{i}.png', dpi=300, bbox_inches='tight')
```

## 📝 Best Practices

### 1. Run After Training Completes

Don't compare mid-training - wait for final checkpoint.

### 2. Use Validation Set

Never use training set for comparison (biased results).

### 3. Multiple Checkpoints

Compare best, last, and mid-training checkpoints:

```python
checkpoints = [
    "epoch_10.ckpt",
    "epoch_20.ckpt",
    "best_model.ckpt"
]

for ckpt in checkpoints:
    ComparisonConfig.FINETUNED_CHECKPOINT = ckpt
    main()
```

### 4. Document Results

Save metrics and visualizations for your records.

### 5. Cross-Reference with Training Logs

Compare test metrics with validation metrics during training.

## 🐛 Troubleshooting

### Issue: "Foundation model predicts all one class"

**Cause**: Model hasn't seen diverse data

**Solution**: This is expected - shows fine-tuning value!

### Issue: "Both models perform similarly"

**Possible causes**:

- Fine-tuning didn't train (check loss curves)
- Learning rate too low
- Not enough epochs
- Data too easy (foundation sufficient)

**Solutions**:

- Verify training actually ran
- Check training logs
- Try more challenging data

### Issue: "Fine-tuned worse than foundation"

**Red flag!** Check:

- ✅ Correct checkpoint loaded?
- ✅ NUM_FRAMES matches dataset?
- ✅ Data corruption?
- ✅ Overfitting (train vs val loss diverge)?

### Issue: "CUDA out of memory"

```python
# Reduce batch size
ComparisonConfig.BATCH_SIZE = 4

# Or evaluate in smaller chunks
# Modify evaluate_dataloader to process fewer batches at once
```

### Issue: "Visualizations look wrong"

Check:

- RGB channel order (should be RGB not BGR)
- Value range (should be [0, 1])
- Frame indexing (post-fire is frame 1)

## 📚 Output Interpretation Examples

### Example 1: Excellent Fine-tuning

```
Macro F1:  0.5234 → 0.7891  (+26.57%)
Burned F1: 0.4123 → 0.7234  (+31.11%)

Verdict: 🎉 Fine-tuning provides HUGE improvement!
```

**What this means**:

- Model learned burn patterns very well
- Ready for production use
- Significant value added by fine-tuning

### Example 2: Class-Specific Issues

```
Per-Class F1:
  Unburned:      0.89 → 0.92  (+3%)
  Low:           0.54 → 0.68  (+14%)
  Moderate-Low:  0.41 → 0.63  (+22%)
  Moderate-High: 0.32 → 0.51  (+19%)
  High:          0.18 → 0.23  (+5%)
```

**What this means**:

- High severity still challenging (only +5%)
- Need more high severity examples
- Or adjust class weights

### Example 3: Minimal Improvement

```
Macro F1: 0.6234 → 0.6456  (+2.22%)

Verdict: ⚠️  Fine-tuning provides minimal improvement.
```

**What to do**:

- Review training curves
- Check if foundation model is "good enough"
- Consider different architecture or loss

## 🎓 Using Results

### For Research Papers

- Include metric tables
- Show confusion matrices
- Highlight key improvements
- Discuss failure cases

### For Stakeholders

- Focus on overall improvement %
- Show visual examples (side-by-side)
- Explain practical impact
- Provide confidence intervals

### For Model Selection

- If improvement < 5%: Consider using foundation model (simpler)
- If improvement > 10%: Deploy fine-tuned model
- If improvement > 20%: Strong case for custom model

## 📄 License

This comparison tool is provided for research and educational purposes.

---

**Last Updated**: January 29, 2026  
**Version**: 1.0.0
