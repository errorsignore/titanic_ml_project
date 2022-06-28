# Titanic ML Project

Repo used for training purposes at [Indicum](https://indicium.tech/).

## What should I read before cloning this repo?

This is an example project to understand how to work with Kedro and MLFlow for machine learning projects. If you are new to Kedro and MLFlow, I highly recommend that you follow these getting started tutorials:

 * [Kedro](https://kedro.readthedocs.io/en/stable/02_get_started/01_prerequisites.html)
 * [MLFlow (with kedro-mlflow)](https://kedro-mlflow.readthedocs.io/en/stable/source/03_getting_started/index.html)

Also, to understand why they exist I suggest this [video](https://www.youtube.com/watch?v=ZPxuohy5SoU&ab_channel=PyData). 

## Data used in this project

In this project, there is one datasets:

 * [Titanic Dataset](https://www.kaggle.com/competitions/titanic/data). The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck, which we will use to create our models.

## Goal of the project

The mains goal of this project is to help understand how Kedro and MLFLow make your life a lot easier when working in a machine learning project that needs to be reproducible, maintainable e with tracking of the models metrics.

The idea here is to classify a Pokemon type based on his statics, such as Life, Attack, Defense. This idea was inspired by this [Kaggle notebook](https://www.kaggle.com/ericazhou/pokemon-type-classification).

## Project dependencies

This project was created using Kedro 0.17.7. To install, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```
### Pipeline structure

The pipeline structure of this project is based on what we think is the optimal way of working in ML projects. Here we have a brief description of what expect in each pipeline.

  1) **Pre-processing pipeline**: basic data manipulation such as rename columns, filter rows, select, cast data type.
  2) **Data engineering pipeline**: create new features, handling the missing data, here we have more complex data manipulations
  3) **Data science pipeline**: train models, while testing different techniques and hyperparameters tuning. If you are working on a project with multiple objectives, I suggest you rename your pipeline based on your model. Let's suppose we wanted also to predict the pokemon attack. Then, we would have a pipeline named predict_primary_type and predict_attack if this was another objective, and the data science pipeline would be extinct.
  4) **Model metrics pipeline**: a pipeline to make report of our precision and register our metrics using MLFlow
  5) **Indicium predictor**: the optimal way that we developed at [Indicium](https://indicium.tech/) to use our models in API.