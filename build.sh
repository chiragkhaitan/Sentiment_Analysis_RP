#!/bin/bash
pip install -r requirements.txt
python download_nltk.py
python train_model.py
