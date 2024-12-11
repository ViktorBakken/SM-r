import pandas as pd
import numpy as np
import os
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 42

#Load data
df = pd.read_csv(os.path.abspath("../data/labeled_training_data.csv"))

print(np.mean(df['increase_stock'] == 0))

#Select all columns except the last
X = df.iloc[:, :-1]
#Select label column
y = df['increase_stock']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=seed)

baseline = DummyClassifier(strategy='uniform', random_state=seed)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dummy', baseline)])


pipeline.fit(X_train,y_train)

y_pred = baseline.predict(X_test)
y_proba = baseline.predict_proba(X_test)[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1-Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test Recall-Weighted: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test Precision-Weighted: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

