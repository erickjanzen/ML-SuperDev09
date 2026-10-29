import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv('apple_quality.csv')

X = df.iloc[:,:-1].values
y = df.iloc[:,:-1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42)

modelos = {'naive':GaussianNB(), 'tree': DecisionTreeClassifier(),
           'forest': RandomForestClassifier(100),
           'knn':KNeighborsClassifier(5),'svm':SVC()}

for modelo in modelos.values():
    modelo.fit(X_train, y_train)