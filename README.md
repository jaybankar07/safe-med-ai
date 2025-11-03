💊 Safe-Med AI
A CNN-based System for Detecting Medicine Packaging Seal Integrity using Mobile Imagery
📘 Overview

Safe-Med AI is a deep learning–powered application designed to automatically detect tampered or broken medicine packaging seals using mobile imagery.
The project uses a Convolutional Neural Network (EfficientNetB0) trained on a custom dataset of medicine packages labeled as Defect or No Defect.
It’s deployed as an interactive Streamlit web app, allowing users to upload photos and receive instant predictions with confidence scores.

🎯 Problem Statement

Manual visual inspection of medicine packaging is:

Time-consuming

Error-prone

Often impossible in rural or resource-limited settings

This can lead to unsafe or contaminated medicines being distributed to patients.
Safe-Med AI automates this process using Artificial Intelligence, ensuring patient safety and trust.

🧠 Key Features

Real-time AI detection using mobile or desktop cameras

Binary classification: Defect vs. No Defect

CNN-based model (EfficientNetB0) trained on custom dataset

Strong data augmentation for small dataset generalization

Deployed via Streamlit Cloud for global access

Confidence scoring and clear visual feedback

⚙️ Tech Stack
Component	Technology
Model Architecture	EfficientNetB0 (Keras / TensorFlow)
Frameworks	TensorFlow, Keras, Streamlit
Languages	Python
Deployment	Streamlit Cloud
Data Augmentation	ImageDataGenerator (Keras)
Evaluation Metrics	Accuracy, Precision, Recall, F1, AUC-ROC
🚀 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/jaybankar07/safe-med-ai.git
cd safe-med-ai

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

4️⃣ Open in your browser
http://localhost:8501

🌐 Live App

Once deployed on Streamlit Cloud, you can access the live app here:
👉 Safe-Med AI Live Demo

(Replace the link above after deployment)

📊 Model Performance Summary
Metric	Score
Accuracy	~54% (on validation set)
F1-Score (No Defect)	0.66
F1-Score (Defect)	0.32
AUC-ROC	Good general separation but more data needed

🧩 Model performance can improve with dataset expansion and fine-tuning.

🔮 Future Work

Expand dataset for more robust learning

Add Grad-CAM explainability to visualize model focus areas

Deploy mobile version using TensorFlow Lite

Enable batch image testing for hospitals or distributors

Integrate into pharmacy dashboard systems

👩‍💻 Author

Siddhi Nanasaheb Hon
Department of Artificial Intelligence & Data Science
Sanjivani University, School of Engineering & Technology

Guided by Dr. Ajay Shankar

🏆 Acknowledgment

This project was developed as part of the academic work under the course Deep Learning (IPR Presentation).
Special thanks to the mentors and peers who contributed to dataset creation and testing.

📜 License

This project is released under the MIT License — feel free to use and adapt for educational or research purposes.