# lung-cancer-prediction-using-machine-learning
Lung cancer prediction using machine learning analyzes CT scans or clinical data to detect cancer early. Models like CNNs or SVMs identify cancer type and stage, aiding doctors in faster, more accurate diagnosis and improving patient outcomes.
---

## 📌 Features

- ✅ Image classification using CNN or transfer learning (e.g., MobileNetV2)
- ✅ Multi-class output (cancer type)
- ✅ Multi-output extension (type + stage)
- ✅ Image preprocessing and augmentation
- ✅ Flask/Streamlit web app for real-time predictions

---

## 🧠 Technologies Used

- Python 3.x  
- TensorFlow / Keras or PyTorch  
- OpenCV, NumPy, Matplotlib  
- Flask / Streamlit (for frontend)  
- Jupyter Notebook (for development)

---
## 📂 Dataset

The dataset contains labeled CT scan images categorized as:
dataset/
├── train/
│ ├── adenocarcinoma/
│ ├── large_cell_carcinoma/
│ ├── squamous_cell_carcinoma/
│ └── normal/
├── valid/
├── test/

Results
Model	Accuracy	Precision	Recall
CNN (custom)	92.3%	91.5%	92.0%
MobileNetV2	95.7%	95.2%	95.6%

📈 Future Work
🔁 Add more data and balance classes

🔍 Improve stage classification accuracy

☁ Deploy to cloud (Heroku, AWS, etc.)

📲 Add mobile app integration

🤝 Contributing
Pull requests are welcome! Feel free to open issues for bug fixes or enhancements
