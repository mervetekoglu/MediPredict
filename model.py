import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# CSV 
df = pd.read_csv('data/dataset.csv')

# Temizlik
df.columns = df.columns.str.strip()
df = df.fillna(0)

# collect alll symptoms
symptom_columns = [col for col in df.columns if col != 'Disease']
all_symptoms = []
for col in symptom_columns:
    all_symptoms.extend(df[col].unique())

all_symptoms = list(set([s.strip() for s in all_symptoms if isinstance(s, str) and s != '0']))
all_symptoms.sort()

# Generate a binary vector for each row.
def encode_symptoms(row):
    row_symptoms = [str(row[col]).strip() for col in symptom_columns if str(row[col]).strip() != '0']
    return [1 if s in row_symptoms else 0 for s in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms

# disease tickets
le = LabelEncoder()
y = le.fit_transform(df['Disease'].str.strip())

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('symptoms_list.pkl', 'wb') as f:
    pickle.dump(all_symptoms, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("The model was successfully trained and registered!")
print(f"Total number of symptoms: {len(all_symptoms)}")
print(f"Total number of diseases:  {len(le.classes_)}")