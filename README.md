# Emotion Detector

The notebook trains and saves the text pipeline. FastAPI loads that same pickle
file and Streamlit sends user text to the API.

## 1. Install packages

Run these commands from this folder:

```powershell
pip install -r requirements.txt
```

## 2. Train and save the model

Open `Train_Model.ipynb` in VS Code, select a Python kernel, and choose
**Run All**. The last cell saves `Emotion_Detector_Model.pkl` in this folder.

