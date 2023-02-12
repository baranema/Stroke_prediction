FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./app /app
WORKDIR /app
RUN pip install sklearn joblib
RUN pip install pandas
RUN pip install imblearn
RUN pip install xgboost