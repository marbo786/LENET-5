<div align="center">

# 🧠 LeNet-5 Handwritten Digit Recognizer

**A interactive digit recognition web app powered by the classic LeNet-5 CNN architecture**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-brightgreen?style=for-the-badge)](http://yann.lecun.com/exdb/mnist/)

🚀 **[Try it Live →](https://lenet-5-zedlezgpachnhzcn4po5ld.streamlit.app/)**

</div>

---

## ✨ What It Does

Draw any digit (0–9) on the interactive canvas and watch the trained LeNet-5 model predict it when you select **Predict Digit**. No setup needed — just open the app and start drawing!

---

## 🏗️ Architecture

LeNet-5 is one of the pioneering convolutional neural networks, introduced by Yann LeCun et al. This implementation uses a modernized variant with **Batch Normalization** and **ReLU activations**.

```
Input (1×32×32)
     │
     ▼
┌─────────────────────────────┐
│  Layer 1 — Conv Block       │
│  Conv2d(1→6, 5×5)           │
│  BatchNorm → ReLU           │
│  MaxPool(2×2)               │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 2 — Conv Block       │
│  Conv2d(6→16, 5×5)          │
│  BatchNorm → ReLU           │
│  MaxPool(2×2)               │
└──────────────┬──────────────┘
               │
               ▼
         Flatten (400)
               │
     ┌─────────┴─────────┐
     │  FC: 400 → 120    │
     │       ReLU        │
     │  FC: 120 → 84     │
     │       ReLU        │
     │  FC:  84 → 10     │
     └─────────┬─────────┘
               │
               ▼
       Output (10 classes)
```

| Layer | Type | Output Shape |
|-------|------|-------------|
| Input | — | `1 × 32 × 32` |
| Layer 1 | Conv + BN + ReLU + Pool | `6 × 14 × 14` |
| Layer 2 | Conv + BN + ReLU + Pool | `16 × 5 × 5` |
| Flatten | — | `400` |
| FC1 | Linear + ReLU | `120` |
| FC2 | Linear + ReLU | `84` |
| FC3 | Linear | `10` |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🖥️ Frontend | Streamlit + streamlit-drawable-canvas |
| 🤖 Model | PyTorch (LeNet-5) |
| 🔢 Dataset | MNIST Handwritten Digits |
| 🖼️ Image Processing | PIL + NumPy |

---

## 📁 Project Structure

```
LENET-5/
├── app.py               # Streamlit web interface + prediction logic
├── model.py             # LeNet-5 architecture definition (PyTorch)
├── lenet5_mnist.pth     # Pre-trained model weights (MNIST)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🎮 How to Use

1. **Open the app** in your browser
2. **Draw a digit** (0–9) on the black canvas using your mouse
3. **Select Predict Digit** to see the predicted class and confidence
4. **Clear** the canvas and try another digit!

---

## 📚 Background

LeNet-5 was originally proposed by **Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner** in their landmark 1998 paper *"Gradient-Based Learning Applied to Document Recognition"*. It was one of the first CNNs to demonstrate the power of end-to-end learning for image classification, achieving near-human accuracy on MNIST.

This implementation modernizes the original by replacing sigmoid activations with **ReLU** and adding **Batch Normalization** for faster, more stable training.

---

## 👤 Author

**Mohsin Saeed**
📧 [marboo786@gmail.com](mailto:marboo786@gmail.com)
🐙 [github.com/marbo786](https://github.com/marbo786)

---

<div align="center">

Built with ❤️ using PyTorch & Streamlit

⭐ If you found this useful, please consider starring the repo!

</div>
