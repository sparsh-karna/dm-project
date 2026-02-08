# IoT Block-Storage Acceptance — Data Mining Case Study

A comparative analysis of statistical tools (R, Weka, JMP) applied to an IoT blockchain block-storage acceptance dataset from IEEE DataPort.

## Dataset

**Source:** [IEEE DataPort — IoT Nodes Block Storage Acceptance Dataset](https://dx.doi.org/10.21227/eyc1-a041)

- 30,000 records, 11 attributes
- Binary classification: can an IoT edge node safely store an additional blockchain block?

## Repository Structure

```
dataset/              Raw and cleaned CSV data + documentation
clean_dataset.py      Python script for data cleaning and encoding
r/                    R analysis script and outputs (EDA plots, model summaries, ROC curves)
jmp/                  JMP EDA and modelling outputs
weka/                 Weka EDA and modelling outputs (J48, Random Forest, Logistic Regression)
report/               LaTeX report source and compiled PDF
```

## Models Compared

| Model               | R Accuracy | Weka Accuracy | AUC (R) |
|----------------------|-----------|---------------|---------|
| Logistic Regression  | 75.7%     | 86.3%         | 0.761   |
| Decision Tree        | 90.0%     | 96.3%         | 0.943   |
| Random Forest        | **97.5%** | **97.3%**     | **0.997** |

## Key Findings

- **Random Forest** achieves the best performance across all three tools
- **Current Available Storage** is the single most important predictor (RF importance: 100)
- Results are consistent across R, Weka, and JMP, confirming tool-independent reliability

## Authors

- Sparsh Karna (23BDS1172)
- Lavanaya Malhotra (23BDS1169)
