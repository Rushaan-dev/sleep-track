# 🧠 Sleep Track — Non-Invasive Sleep Monitoring System

A real-time EEG-based sleep staging system that classifies 
sleep stages using custom biomedical hardware and machine learning.
Built as a Final Year Project at FAST-NUCES, Lahore (2025-2026).

---

## 🔬 What This Does

Records raw EEG brain signals during sleep, processes them,
extracts frequency band features, and classifies each 30-second 
epoch into a sleep stage — without any invasive procedure.

---

## 🏗️ System Pipeline

Raw EEG Signal
      ↓
BioAmp EXG Pill (Hardware Amplification)
      ↓
Arduino Uno (512Hz Data Acquisition)
      ↓
Python Serial Reader (CSV Storage)
      ↓
Signal Preprocessing (Notch + Bandpass Filter)
      ↓
Feature Extraction (Delta, Theta, Alpha, Beta bands)
      ↓
ML Classifier (Random Forest / SVM)
      ↓
Sleep Stage Output (Wake / Light / Deep / REM)

---

## 💤 Sleep Stages Detected

|     Stage      | Brain Wave |    Frequency    |
|----------------|------------|-----------------|
|     Awake      |   Beta     |    14-30 Hz     |
| Relaxed/Drowsy |   Alpha    |    8-13 Hz      |
|   Light Sleep  |   Theta    |    4-7 Hz       |
|   Deep Sleep   |   Delta    |    0.5-3 Hz     |

---

## 🔧 Hardware Used

- **BioAmp EXG Pill** — Biomedical amplifier for EEG signal acquisition
- **Arduino Uno** — Analog to digital conversion at 512Hz sampling rate
- **EEG Electrodes** — Non-invasive scalp electrodes

---

## 🧪 Tech Stack

|         Area      |             Tools                 |
|-------------------|-----------------------------------|
| Signal Processing |        Python, SciPy, NumPy       |
| Machine Learning  | Scikit-learn (Random Forest, SVM) |
| Data Analysis     |        Pandas, Matplotlib         |
|    Dashboard      |            Streamlit              |
|    Hardware       |      Arduino, BioAmp EXG Pill     |

---

## 📁 Repository Structure
```
sleep-track/
│
├── hardware/
│   └── arduino_eeg.ino        # Arduino data acquisition code
│
├── preprocessing/
│   └── eeg_preprocessing.py   # Signal filtering + feature extraction
│
├── ml/
│   └── sleep_classifier.py    # ML model training + evaluation
│
├── dashboard/
│   └── app.py                 # Streamlit web dashboard
│
├── results/
│   └── eeg_analysis.png       # Visualization outputs
│
└── README.md
```

---

## 📊 Results

> ML results and accuracy metrics will be updated here 
> after model training is complete.

| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| Random Forest | TBD      | TBD      |
| SVM           | TBD      | TBD      |

---

## 📚 Dataset

- **Custom Dataset:** 3-4 hours of EEG recorded using BioAmp EXG Pill
for model training and validation

---

## 🚀 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python preprocessing/eeg_preprocessing.py

# Train ML model
python ml/sleep_classifier.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 👨‍💻 Author

**Muhammad Rushaan Abbas**  
BS Computer Engineering — FAST-NUCES, Lahore  
📧 rushaan935@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rushan-abbas-b45945357)

---

## 🏫 Acknowledgements

This project was developed as a Final Year Project at 
FAST-NUCES Lahore under the supervision of Abeer Bashir.
