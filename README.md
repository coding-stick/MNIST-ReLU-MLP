# Neural Network Visualizer – MNIST Classification

This project is an **interactive visualizer** for a **multi-layer perceptron (MLP)** that classifies handwritten digits from the **MNIST dataset**.  
It trains the model in real-time while showing:

- The **weights** between neurons, updating as learning progresses
- The **pixel input grid** for the current sample
- The model’s **predictions** for each digit class
- The **target label** for the sample being trained

---

## Features

- **Live Training Visualization** – Watch weights change after every batch.
- **Pixel Grid Display** – All 784 input pixels shown in sync with the current sample.
- **Neural Network Diagram** – Neurons and connections drawn in layers.
- **Prediction Output** – Real-time output activations and predicted digit.
- **Interactive Controls** – Start/stop training on key press.

---

## Model Architecture

The model is a simple **MLP (Multi-Layer Perceptron)** with ReLU activations:

Input Layer: 784 neurons (28x28 pixels)
Hidden Layer 1: 128 neurons, ReLU
Hidden Layer 2: 64 neurons, ReLU
Output Layer: 10 neurons (digits 0–9), Softmax

yaml
Copy
Edit

---

## Requirements

- Python 3.8+
- PyTorch
- Pygame
- Matplotlib
- NumPy

Install dependencies:

```bash
pip install torch pygame matplotlib numpy pandas scikit-learn
```

## Controls

T – Toggle training on/off

ESC / Close window – Exit

## How It Works
Loads the MNIST dataset using PyTorch’s DataLoader.
<img width="405" height="381" alt="Shotcut_00_00_50_533" src="https://github.com/user-attachments/assets/ee05f51d-bedd-47ed-ada6-e97fc0c728c3" />

Builds an MLP for classification.

Updates weights in real-time and visualizes them as colored lines.

Displays current pixel input in a 28x28 grid.

Shows prediction scores and target label for each sample.

## License
This project is open-source under the MIT License.
