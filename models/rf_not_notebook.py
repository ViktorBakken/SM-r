import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 42

#Load data
df = pd.read_csv(os.path.abspath("../data/labeled_training_data.csv"))

#Select all columns except the label
X = df.iloc[:, :-1]

#Create new features
X['day_or_night'] = X['hour_of_day'].apply(lambda x: 1 if 8 <= x < 21 else 0)

X['normal_day'] = (~((X['summertime'] == 1) | (X['holiday'] == 1) | (X['weekday'] == 0))).astype(int)

X['cold'] = X['temp'].apply(lambda x: 1 if x <= 8 else 0)

X['atemp'] = (243.04 * (np.log(X['humidity']/100)
                        + (17.625 * X['dew']) / (243.04 + X['dew']))) / (17.625 - np.log(X['humidity']/100)
                        - (17.625 * X['dew']) / (243.04 + X['dew']))

#Remove bad features
X = X.drop(['snow', 'snowdepth', 'holiday', 'visibility', 'precip', 'dew'], axis=1)

X.info()

#Select label column
y = df['increase_stock']


#Stratify train and test split so that data keeps the same proportion of classes
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed, stratify=y)

#Use stratified cross validation since data is unbalanced
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

#Create model and set variables that should not be tuned
model = RandomForestClassifier( 
    random_state=seed,
    # n_estimators=145,
    # max_depth=24,
    class_weight= 'balanced',
    # min_samples_leaf=1,
    # min_samples_split=2,
    bootstrap=True,
    max_features='sqrt',
    criterion='gini',
    n_jobs=-1
)

#Create a autoscaler pipeline to normalize data
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', model)])

#The hyperparameters to tune
param_dists = {
    'rf__n_estimators': np.linspace(90, 150, num=30, dtype=int),      
    'rf__max_depth': np.linspace(15, 80, num=20, dtype=int),              
    'rf__min_samples_split': np.linspace(2, 10, num=8, dtype=int),      
    'rf__min_samples_leaf': np.linspace(1, 8, num=8, dtype=int), 
    # 'max_features' : ['sqrt', 'log2'],  
    # 'criterion' : ['gini', 'entropy', 'log_loss'],                             
    # 'bootstrap': [True, False], 
    # 'class_weight' : ['balanced_subsample', 'balanced']                                      
}

#Use Random search with cross validation to tune hyperparameters
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dists,
    n_iter=3000,
    scoring='f1_weighted',
    refit=True, 
    cv=skf,
    n_jobs=-1,
    verbose=3,
    return_train_score=False,
    random_state=seed,
)

random_search.fit(X_train, y_train)

print("Best Parameters:")
print(random_search.best_params_)
print(f"\nBest Score: {random_search.best_score_:.4f}")

#Save the best model
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

#Print metrics (Weighted F1 is the important one)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1-Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test Recall-Weighted: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test Precision-Weighted: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

rf_classifier = best_model.named_steps['rf']

#Retrieve feature importances
importances = rf_classifier.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

#Plot using Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest Classifier')
plt.show()