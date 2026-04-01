# UrbanSound8K Audio Classification API

This project is an end-to-end audio classification system using Deep Learning and FastAPI.

## Overview
The model is trained on the UrbanSound8K dataset to classify environmental sounds such as dog bark, siren, drilling, etc.

## Features
- Upload audio file (.wav)
- Predict sound category
- Returns confidence score
- FastAPI-based deployment

## Tech Stack
- Python
- TensorFlow
- FastAPI
- Librosa
- NumPy

## Project Structure
- `app.py` → FastAPI backend
- `model_training.ipynb` → model training
- `urbansound8k_transformer_model.h5` → trained model
- `requirements.txt` → dependencies
