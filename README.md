# Stroke Prediction Project

Strokes are one of the leading causes of death and disability worldwide. Early detection and prediction of stroke risk can lead to better treatment outcomes and improved patient prognoses. This project aims to analyse how machine learning can help with that.

Machine learning algorithms can be used to analyze large amounts of data to identify patterns and correlations that may be indicative of an increased risk of stroke. This information can then be used to design targeted prevention strategies, such as lifestyle modifications, to reduce an individual's risk of stroke.

The goal of this project is to analyse the data from [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and create a model which can predict whether ths "stroke" column in the dataset.

## EDA and building model
The exploratory data analysis and model creation can be found in 325.ipynb 

## Deployed model
The final model was deployed with FastAPI which can be run as this WITHOUT docker:

go to /app folder
```
uvicorn main:app --reload
```

The API can be tested in http://localhost:8000/docs

Running with docker:
```
docker build -t myimage .
```

and then 
```
docker run -d --name mycontainer -p 80:80 myimage
```