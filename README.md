# 💎End-to-End ML Pipeline Diamond Price Prediction 
### Part of NTI national National Telecommunication Institute (NTI) ML intern
> Predict diamond prices using a tuned SVM model exported to ONNX.

## 📊 Model Performance & Results

### Dataset Stats (53,940 diamonds after cleaning)

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| **carat** | 0.798 | 0.474 | 0.20 | 0.40 | 0.70 | 1.04 | 5.01 |
| **depth** | 61.75 | 1.43 | 43.0 | 61.0 | 61.8 | 62.5 | 79.0 |
| **table** | 57.46 | 2.23 | 43.0 | 56.0 | 57.0 | 59.0 | 95.0 |
| **price** | $3,932 | $3,989 | $326 | $950 | $2,401 | $5,324 | $18,823 |
| **x** | 5.73 | 1.12 | 0.00 | 4.71 | 5.70 | 6.54 | 10.74 |
| **y** | 5.73 | 1.14 | 0.00 | 4.72 | 5.71 | 6.54 | 58.90 |
| **z** | 3.54 | 0.71 | 0.00 | 2.91 | 3.53 | 4.04 | 31.80 |

✅ Outliers removed  
✅ Engineered `Volume = x * y * z`  
✅ Ordinal encoding for `cut`, `color`, `clarity`  
✅ Log-transformed `price` for modeling

---

### 🧪 Model Comparison (5-Fold CV on Train/Val)

| Model | Train MSE | Val MSE | Winner? |
|-------|-----------|---------|---------|
| **CART** | 0.030730 | 0.031826 | ❌ |
| **SVM**  | 0.011478 | **0.011904** | ✅ |
| **LR**   | 0.072757 | 0.072798 | ❌ |
| **KNN**  | 0.016377 | 0.024888 | ❌ |

> 🏆 **SVM selected** — lowest validation error

---

### ⚙️ Best SVM Hyperparameters

```python
SVR(
  C=0.2637,
  epsilon=0.0968,
  kernel='rbf'
)
```

✅ Final Val MSE: **0.0119**  
✅ Pipeline: `MinMaxScaler` → `SVR`

---

### 📦 Model Export & Format

- **Format**: ONNX (Open Neural Network Exchange)
- **File**: `svm_diamonds_pipeline.onnx`
- **Size**: ~XX KB (optimized for web/backend)
- **Input**: `[carat, cut, color, clarity, depth, table, Volume]`
- **Output**: `log(price)` → converted to USD with `np.expm1()`

---


## 🤝 Contributing

PRs welcome! For major changes, please open an issue first.

---

## 📄 License

MIT © yahya mohamed



