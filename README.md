# Stroke Prediction Project

Strokes are one of the leading causes of death and disability worldwide. Early detection and prediction of stroke risk can lead to better treatment outcomes and improved patient prognoses. This project aims to analyse how machine learning can help with that.

Machine learning algorithms can be used to analyze large amounts of data to identify patterns and correlations that may be indicative of an increased risk of stroke. This information can then be used to design targeted prevention strategies, such as lifestyle modifications, to reduce an individual's risk of stroke.

The goal of this project is to analyse the data from [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and create a model which can predict whether ths "stroke" column in the dataset.

## About Notebook 

This Jupyter notebook is an analysis and prediction of stroke based on a Kaggle dataset. The notebook explores the dataset, performs some data preprocessing and visualization, and then uses machine learning algorithms to predict whether a person is likely to have a stroke.

The dataset used in this notebook is the "Healthcare Dataset Stroke Data" available on Kaggle. It contains over 5,000 records of patient data from various regions of India, collected between 2018 and 2019. The dataset includes various features related to the patients, such as age, gender, hypertension, heart disease, smoking status, BMI, and more.

### Getting Started

To get started with this notebook, you will need to have Jupyter Notebook installed on your machine. You will also need to clone or download this repository and extract the files to a local directory.

Once you have the files locally, open the "325.ipynb" file in Jupyter Notebook to view the analysis and prediction process.

### Notebook Content

The notebook is divided into different sections, each serving a specific purpose. Here is a brief overview of what each section does:

* **Importing Libraries and Dataset** - This section imports the necessary Python libraries and loads the dataset into the notebook.
* **Data Exploration and Visualization**- This section explores the dataset and visualizes the distribution of various features using different plots, such as bar charts, histograms, and pie charts.
* **Data Preprocessing** - This section preprocesses the data by converting categorical variables into numerical ones, filling missing values, and normalizing the data.
* **Feature Selection** - This section selects the most important features that influence the prediction of stroke, using techniques like correlation matrix and feature importance.
* **Machine Learning Modeling** - This section applies different machine learning algorithms to the preprocessed data and compares their accuracy, using techniques like cross-validation and confusion matrix.
* **Final Model and Prediction** - This section uses the selected features to train a final machine learning model and uses it to make predictions on new data.

### Conclusion

This notebook provides a thorough analysis of the "Healthcare Dataset Stroke Data" available on Kaggle and applies machine learning algorithms to predict the likelihood of stroke based on various patient features. It can be used as a starting point for anyone interested in analyzing and predicting health-related data, and can be easily extended to include other datasets and algorithms.


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