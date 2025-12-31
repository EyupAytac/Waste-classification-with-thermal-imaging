# Thermal-Based Automated Material Classification for Solid Waste

[![Conference](https://img.shields.io/badge/Status-Research%20Prototype-blue)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<br />

<div align="center">
  <b>Eyüp Enes Aytaç</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>Erkani Mert Tosun</b>
  <br />
  <br />
  <img src="https://placehold.co/800x400/EEE/31343C?text=Figure+1:+System+Setup+(Thermal+Camera+%2B+Heating+Unit)" alt="System Setup" width="800"/>
  <br />
  <em>Figure 1: The experimental setup consisting of active heating lamps and a FLIR T420 thermal camera.</em>
</div>

---

## 📄 Abstract
> **Abstract—** The growing demand for efficient waste management and recycling has increased the need for automated and cost-effective material classification systems. Conventional recycling processes rely heavily on manual sorting, making material classification a key bottleneck for automation. In this study, a thermal-based material classification approach is proposed that exploits the temporal thermal response of solid waste objects. The system consists of an active heating unit with three thermal lamps and a FLIR T420 thermal camera. Objects are first heated in a controlled manner and then allowed to cool passively, while their temperature evolution is recorded over time. The resulting thermal image sequences capture material-specific temporal dynamics. These spatiotemporal features are modeled using a convolutional neural network combined with a long short-term memory network, enabling joint spatial feature extraction and temporal behavior analysis. Experimental results obtained from a laboratory-collected dataset demonstrate that the proposed method provides an effective and practical solution for automated solid waste material classification.

**Index Terms—** Thermal imaging, solid waste classification, recycling automation, temporal dynamics, CNN–LSTM.

---

## 🧩 Proposed Framework

### 1. The Physical Setup
Our approach utilizes "Active Thermography." Unlike standard visual cameras, our system relies on the **thermal retention properties** of different materials (e.g., how fast plastic cools down vs. metal).



### 2. Network Architecture (CNN-LSTM)
To capture both the visual shape (Spatial) and the cooling rate (Temporal), we utilize a hybrid Deep Learning architecture:
* **CNN (Convolutional Neural Network):** Extracts spatial features from individual thermal frames.
* **LSTM (Long Short-Term Memory):** Analyzes the sequence of frames to understand how the temperature changes over time.



---

## 📊 Experimental Results

The model was evaluated on a custom laboratory-collected dataset containing various waste materials.

| Material Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Metal** | 0.95 | 1.00 | 0.98 |
| **Plastic** | 0.94 | 0.89 | 0.91 |
| **Carton** | 0.94 | 0.94 | 0.94 |


*(Note: These are placeholder metrics. Please update with your actual validation results.)*

---

## 🛠 Installation & Usage

```bash
# Clone the repository
git clone [https://github.com/YourUsername/thermal-waste-classification.git](https://github.com/YourUsername/thermal-waste-classification.git)

# Navigate to directory
cd thermal-waste-classification

# Install dependencies
pip install -r requirements.txt
