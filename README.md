# ML-model
---

# 🧠 Anomaly Detection using Generative Adversarial Networks (ADGAN)

An implementation of **ADGAN (Anomaly Detection using Generative Adversarial Networks)** — a deep learning approach that learns the distribution of *normal* images and detects anomalies by identifying deviations from it.

---

## 🚀 Overview

This project uses a **GAN-based model** to perform unsupervised anomaly detection on the **MVTec AD dataset**.
The **Generator** learns to reconstruct “normal” samples, and the **Discriminator** learns to distinguish real from generated data.
The **reconstruction difference** between the original and generated image forms the **anomaly map**, highlighting unusual regions.

🔹 **Use case examples:**
Industrial defect detection · Medical imaging irregularities · Fraud or outlier detection

---

## 🧩 Architecture

| Component             | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| **Generator (G)**     | Learns to recreate normal images from noise or latent vectors |
| **Discriminator (D)** | Distinguishes real vs generated images                        |
| **Loss Function**     | Binary Cross-Entropy (adversarial loss)                       |
| **Optimization**      | Adam optimizer (learning rate = 0.0002)                       |

🌀 The model is trained adversarially — G tries to fool D, and D tries to catch G’s fakes.

---

## 🧮 Evaluation Metrics

Although GANs are primarily qualitative, several metrics can be used to evaluate performance:

* 🧾 **Reconstruction Error (L1/L2)** – measures pixel-wise difference
* 🎯 **Precision, Recall, F1-Score** – for labeled anomaly data
* ⚖️ **Discriminator Accuracy** – indicates GAN balance
* 💡 **FID (Fréchet Inception Distance)** – evaluates realism of generated images (for future work)

---

## 📂 Dataset

**MVTec Anomaly Detection (AD) Dataset**

> Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).
> *MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
> [🔗 Dataset Link](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

Each object category (e.g., `bottle`, `capsule`, `leather`, `wood`) contains:

* ✅ **Normal images** for training
* ⚠️ **Defective images** for testing

---

## ⚙️ Requirements

```bash
pip install torch torchvision opencv-python matplotlib tqdm
```

---

## 🧠 How It Works

1. **Train on normal data only** → Model learns what “normal” looks like.
2. **Generate reconstructed image** → G tries to reproduce input.
3. **Compute anomaly map** → Difference = possible defect region.
4. **Evaluate and visualize** → Red regions = higher anomaly score.

---

## 📊 Results

✅ GAN learns to reconstruct “normal” samples
⚠️ Anomaly regions show up as red in the heatmap
📈 Performance improves with more epochs and better architectures

---

## ⚠️ Limitations

* Needs a large amount of *normal* data
* GANs can be unstable during training
* Reconstruction-based methods might miss subtle defects

---

## 🔮 Future Work

* Implement **Autoencoder-GAN hybrid** for better reconstructions
* Add **FID score** and **ROC-AUC** for quantitative benchmarking
* Explore **conditional GANs (cGANs)** for category-specific anomaly learning
* Integrate **real-time defect detection pipeline**

---

## 📚 Citations

1. Goodfellow, I. et al. (2014). *Generative Adversarial Nets*, NeurIPS. [📄 Paper](https://papers.nips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)
2. Bergmann, P. et al. (2019). *MVTec AD Dataset*, CVPR. [📄 Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html)
3. Schlegl, T. et al. (2017). *Unsupervised Anomaly Detection with GANs*, IPMI. [📄 Paper](https://arxiv.org/abs/1703.05921)

---

## 🧑‍💻 Author

**Yasesh Chandra**
B.Tech CSE (Data Science) | GITAM University
Exploring AI/ML & Computer Vision 🚀

---
