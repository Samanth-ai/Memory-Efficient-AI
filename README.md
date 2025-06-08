# 🧠 PyTorch Model Optimization: Techniques for Memory-Efficient Deep Learning

This repository demonstrates **state-of-the-art techniques** for reducing the **memory footprint** of large deep learning models in PyTorch — without relying on abstraction libraries like Hugging Face or `bitsandbytes`.

Starting from a **72MB baseline `BigNet` model**, this project implements:

- ✅ Half-Precision (FP16)  
- ✅ LoRA (Low-Rank Adaptation)  
- ✅ 4-Bit Quantization  
- ✅ QLoRA (Quantized LoRA)

---

## 📌 Table of Contents

- [🔧 Techniques Implemented](#-techniques-implemented)
- [📊 Results & Benchmarks](#-results--benchmarks)
- [⚙️ Installation & Setup](#-installation--setup)
- [📦 Bundling the Project](#-bundling-the-project)

---

## 🔧 Techniques Implemented

### 1. Half-Precision (FP16)

Reduces memory usage by **50%** by converting weights to `float16`.

> 💡 Implemented using a custom `HalfLinear` layer for type casting during the forward pass.

---

### 2. LoRA (Low-Rank Adaptation)

Enables **parameter-efficient fine-tuning** by training only low-rank matrices `lora_a` and `lora_b` per linear layer.

> ✅ Reduces gradient memory requirements  
> ✅ Built on top of the half-precision model

---

### 3. 4-Bit Quantization

Implements a custom `Linear4Bit` layer using **block-wise quantization**:

- Stores group max values in `float16`
- Normalizes and quantizes to 4-bit (0–15)
- Packs two values per `int8` byte

> 📉 Achieves up to **7× memory reduction** compared to `float32`

---

### 4. QLoRA (Quantized LoRA)

Combines **4-bit quantization** with **LoRA**, allowing fine-tuning of highly compressed models.

> 🔁 Near 4-bit memory efficiency  
> ✅ Supports gradient updates via LoRA layers

---

## 📊 Results & Benchmarks

### 1. Model Statistics

Run the `stats.py` script to compare model memory and parameters:

```bash
python3 -m src.stats bignet half_precision low_precision lora qlora
```

**Example Output:**

| Model          | Trainable Params | Non-Trainable Params | Total Params | Theoretical Memory | Actual Memory | Forward Memory | Backward Memory |
|----------------|------------------|-----------------------|---------------|---------------------|----------------|----------------|-----------------|
| bignet         | 18.90 M          | 0.00 M                | 18.90 M       | 72.11 MB            | 72.11 MB       | 8.12 MB        | 80.23 MB        |
| half_precision | 0.01 M           | 18.89 M               | 18.90 M       | 36.07 MB            | 36.07 MB       | 0.00 MB        | 0.04 MB         |
| low_precision  | 0.03 M           | 10.62 M               | 10.65 M       | 11.36 MB            | 11.36 MB       | 0.00 MB        | 0.04 MB         |
| lora           | 2.00 M           | 18.90 M               | 20.90 M       | 36.08 MB            | 36.08 MB       | 0.04 MB        | 8.86 MB         |
| qlora          | 1.48 M           | 10.65 M               | 12.13 M       | 12.60 MB            | 12.60 MB       | 0.04 MB        | 11.13 MB        |

### 2. Compare Model Outputs
Ensure the optimized models are numerically close to the baseline bignet:

```bash
python3 -m src.compare bignet half_precision lora qlora
```

**Example Output:**

Comparing bignet and half_precision
 - Max difference: 0.0015
 - Mean difference: 0.0001

Comparing bignet and lora
 - Max difference: 0.0015
 - Mean difference: 0.0001

   
### 3. Fine-Tuning Test
Run a small training loop on a dummy classification task:
```bash
python3 -m src.fit lora
# or
python3 -m src.fit qlora
```


## ⚙️ Installation & Setup

We recommend using Miniconda for environment management.

Create Environment
```bash
conda create --name model_optimization python=3.12 -y
conda activate model_optimization
```

Install PyTorch
Follow the official installation guide for your system:
🔗 https://pytorch.org/get-started/locally/

Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Bundling the Project

Package everything into a ZIP file for distribution:

python3 bundle.py src project_bundle
➡️ Outputs project_bundle.zip in the root directory.
