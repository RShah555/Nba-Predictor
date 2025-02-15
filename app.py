from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Fetch historical data (mock example)
def fetch_nba_data():
    url = "https://dashboard.api-football.com/#"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None
