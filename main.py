import pandas as pd
import sqlite3
import sklearn

# connect to database
conn = sqlite3.connect("hospital.db")

query = """
SELECT * FROM patients
WHERE discharge_disposition_id NOT IN (11,19,20,21);
"""
df = pd.read_sql(query, conn)

df.fillna({'race':'Unknown'}, inplace=True)

df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

features = ['age','time_in_hospital','num_medications','num_lab_procedures','number_diagnoses','readmitted']
df = df[features]

df = pd.get_dummies(df, columns=['age'], drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['time_in_hospital','num_medications','num_lab_procedures','number_diagnoses']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Train‑Test Split

from sklearn.model_selection import train_test_split

X = df.drop(columns=['readmitted'])
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# identify 5 factors
import numpy as np

coef = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
}).sort_values(by='Importance', key=np.abs, ascending=False)

print(coef.head(5))

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model.fit(X_res, y_res)
y_pred = model.predict(X_test)

import seaborn as sns
import matplotlib.pyplot as plt

# Readmission distribution
sns.countplot(x='readmitted', data=df)
plt.show()

# Feature importance (top 5)
sns.barplot(x=coef['Importance'][:5], y=coef['Feature'][:5])
plt.show()