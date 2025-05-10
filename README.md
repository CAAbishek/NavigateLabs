# 🧠 AI Inference Optimization Project

This project demonstrates performance optimization for AI models by applying dynamic quantization. It focuses on *text-based medical diagnosis* and *image-based chest X-ray analysis* using pretrained models from Hugging Face. The goal is to measure and compare inference performance—*latency, **memory usage, and **throughput*—before and after quantization.

---

## 🔍 Project Objectives

- Use pretrained models from Hugging Face for both *text* and *image* classification tasks.
- Apply *dynamic quantization* (torch.quantization.quantize_dynamic) to optimize models.
- Compare the original and quantized models in terms of:
  - *Latency* (inference time)
  - *Memory usage*
  - *Throughput*

---

## 🧰 Tech Stack

- 🐍 Python
- 🤗 Hugging Face Transformers
- 🧠 PyTorch
- 📦 TorchVision 
- 📈 plotly / Streamlit 
- 🖼 PIL
- 🧪 psutil (for memory profiling)
- ⏱ time (for latency measurement)

---

## 📦 Pretrained Models Used

### 🔤 Text-based Diagnosis
- *Model*: bert-base-uncased
 - *Task*: Medical symptom to diagnosis classification

### 🖼 Image-based Diagnosis
- *Model*: ViT pretrained from torchvision
- *Task*: Chest X-ray image classification

---

## ⚙ Installation

```bash
git clone https://github.com/CAAbishek/NavigateLabs.git
cd ai-inference-optimization


# Install dependencies
pip install -r requirement.txt
