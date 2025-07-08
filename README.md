
# 🌫️ DeFogger: Hybrid Deep Learning for Multi-Fog Detection

> A real-time, multi-modal CNN-LSTM framework for fog classification using RGB imagery and LIDAR sequences.

![architecture](assets/architecture_diagram.png)

## 📌 Overview

**FogSight** is a research-grade hybrid deep learning system designed for robust fog classification across diverse environments. The system integrates spatial features from CNNs, temporal patterns via LSTM, and handcrafted fog descriptors to detect and classify:

- ☀️ Clear
- 🌁 Homogeneous Fog
- 🌫️ Inhomogeneous Fog
- ☁️ Sky Fog

## 🔍 Key Highlights

- 🚀 100% test accuracy across 4 fog types
- 🧠 CNN-LSTM hybrid with physics-inspired features
- 🎯 Real-time deployment capability (25–37 FPS)
- 🔁 Adversarial & domain-adaptive augmentation
- 🧪 Comparative study with ViT-LSTM & Dual-CNN FogNet

## 🧠 Architecture

```
            +---------------------+
            |     RGB Image       |
            |   ResNet-50 (CNN)   |
            +---------+-----------+
                      |
               [2048-d feature]
                      |
          +-----------+-----------+
          |                       |
 [LIDAR Sequence]        [Fog Descriptors]
 BiLSTM (128x2)            MLP Encoder
    [Temporal]              [DCP, Variance...]
          \                      //
           \      +-------------+
            \     |  Fusion MLP |
             \    +-------------+
              \         |
              [Softmax Classifier]
                    |
                Prediction
```

## 🗃️ Dataset Sources

| Dataset        | Type            | Samples  |
|----------------|-----------------|----------|
| FRIDA2         | Synthetic       | ~1,056   |
| RESIDE+RTTS    | Synthetic/Real  | 27,000+  |
| D-Hazy         | Synthetic        | 1,449    |
| MultiFog       | Real + LIDAR     | ✔️        |

> Total Samples: 28,397 (Train: 70%, Val: 15%, Test: 15%)

## 🛠️ Setup

```bash
git clone https://github.com/YashM-235/fogsight.git
cd fogsight
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Train the model
python train.py --model cnn_lstm --data ./data --epochs 50

# Evaluate the model
python evaluate.py --checkpoint ./checkpoints/model.pt
```

## 📊 Results

| Model           | Accuracy | FPS  | Notes                      |
|-----------------|----------|------|----------------------------|
| CNN-LSTM        | 100%     | 25   | Hybrid model               |
| ViT-LSTM        | 58.82%   | 17   | Weak on fog textures       |
| Dual-CNN FogNet | 100%     | 37   | Lightweight real-time model|

## 🔬 Fog Descriptors

- Grayscale Variance
- Dark Channel Prior (DCP)
- Edge Attenuation Score
- Contrast Energy

## 🎓 Citation

If you use this code or dataset, please cite our proposed paper:

```
Yash Mehta, Prabhat Kumar, Dileep Yadav, "Fog Detection Using Hybrid Deep Learning Models: A Multi-Modal Approach with CNN-LSTM Architectures", IEEE.
```

## 📄 License

This project is licensed under the MIT License.

## 📫 Contact

For queries or collaborations, reach out to:

**Yash Mehta**  
📧 yash.dlw@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/yash-mehta-402239163) | [GitHub](https://github.com/YashM-235)
