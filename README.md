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

## 3. Start the API

In a terminal from this folder:

```powershell
python -m uvicorn emotion_api:app --reload
```

The API documentation is available at <http://127.0.0.1:8000/docs>.

## 4. Start Streamlit

Open a second terminal in this folder:

```powershell
python -m streamlit run app.py
```

Keep the API terminal running while you use the Streamlit app.
