# Churn Prediction Project

## Overview
This project predicts customer churn using a machine learning model.

## Installation
```bash
bash setup.sh
```

## Running the Pipeline
```bash
python scripts/pipeline.py
python scripts/train_model.py
```

## Running the API
```bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

## Running the Streamlit App
```bash
python scripts/app.py
```
