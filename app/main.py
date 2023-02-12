import joblib
import pandas as pd
from fastapi import FastAPI

from entry import Entry

app = FastAPI()
model = joblib.load("stroke_classifier.joblib")
optimal_proba_cutoff = 0.11613977711204293


def predict_stroke(model, entry):
    """Get data from entry object as a dict"""
    new_entry = pd.DataFrame.from_dict(entry.get_entry_dict())

    """ Predict new data based on threshold """
    predicted_proba = model.predict_proba(new_entry)
    prediction = [
        1 if i >= optimal_proba_cutoff else 0 for i in predicted_proba[:, -1]]

    return {
        "stroke": True if prediction[0] == 1 else False,
        "stroke_prob": list(predicted_proba[:, -1])[0],
        "data": {key: val[0] for key, val in entry.get_entry_dict().items()},
    }


@app.get("/")
def get_root():
    return {"message": "Welcome to the stroke prediction API"}


@app.post("/stroke_prediction_query/")
async def predict_stroke_query(entry: Entry):
    return predict_stroke(model, entry)
