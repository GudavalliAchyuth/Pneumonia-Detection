# 🫁 Pneumonia Detection from Chest X-Rays

An automated medical image classification system built to detect pneumonia from chest X-ray scans using a custom deep learning network and Explainable AI (Grad-CAM).

## 📄 Abstract
*(Prepared for Minor Project Submission)*

The rapid and accurate diagnosis of pneumonia is critical for effective medical intervention. This project introduces an automated diagnostic tool utilizing a custom Convolutional Neural Network (CNN) to classify chest X-ray images. Built using TensorFlow and Keras, the deep learning model is trained on a validated dataset of pediatric chest X-rays categorized into "Normal" and "Pneumonia" classes. To handle the intricacies of medical imaging, the system employs controlled data augmentation and feature extraction techniques to identify opacities and lung abnormalities without compromising anatomical structures. Additionally, the system incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to visually highlight the pathological regions driving the model's decision. The final model achieves high diagnostic accuracy, demonstrating the viability of integrating AI-driven computer vision systems into clinical workflows to reduce diagnostic bottlenecks.

## 🛠️ Technologies Used
* **Deep Learning Framework:** TensorFlow / Keras (Custom Sequential CNN)
* **Explainable AI (XAI):** Grad-CAM (Gradient-weighted Class Activation Mapping)
* **Web Interface:** Streamlit
* **Computer Vision & Processing:** OpenCV, PIL, Matplotlib
* **Machine Learning Tools:** Scikit-Learn

## ⚠️ Prerequisites (Crucial for TensorFlow)
Because this project utilizes TensorFlow 2.x, it requires a specific Python environment to run successfully.
* **Required Python Version:** Python 3.8, 3.9, 3.10, or 3.11.
* **Warning:** If you are using Python 3.12 or newer, the `pip install` command will fail because TensorFlow has not released stable binaries for those versions yet. 

**Recommended Setup:**
If you have a newer version of Python installed, please create an isolated Python 3.10 virtual environment before installing the requirements:
```bash
py -3.10 -m venv venv
.\venv\Scripts\activate

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GudavalliAchyuth/Pneumonia-Detection.git
   cd Pneumonia-Detection

2. Install required dependencies:
     pip install -r requirements.txt
3. Usage & Execution:
    Phase 1: Train the CNN Model: python scripts/train_pneumonia.py
    Phase 2: Command Line (CLI) Inference: python scripts/predict_xray.py path/to/sample_xray.jpeg
    Phase 3: Launch the Streamlit Web Application:streamlit run app.py
