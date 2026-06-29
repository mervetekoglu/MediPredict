from flask import Flask, render_template, request, redirect, url_for, session
import pyodbc
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import pandas as pd

# CSV 
desc_df = pd.read_csv('data/symptom_Description.csv')
prec_df = pd.read_csv('data/symptom_precaution.csv')

desc_df.columns = desc_df.columns.str.strip()
prec_df.columns = prec_df.columns.str.strip()

app = Flask(__name__)
app.secret_key = 'medipredict_secret_key'
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('symptoms_list.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# SQL Server 
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER=/opt/homebrew/lib/libmsodbcsql.17.dylib;'
        'SERVER=localhost,1433;'
        'DATABASE=medipredict;'
        'UID=sa;'
        'PWD=Merve123!;'
        'TrustServerCertificate=yes;'
    )
    return conn
    

# home page
@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

# login page
@app.route('/login.html')
def login_html():
    return redirect(url_for('login'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", email)
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Email veya şifre hatalı!')
    
    return render_template('login.html')

# register page
@app.route('/register.html')
def register_html():
    return redirect(url_for('register'))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                username, email, password
            )
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except:
            conn.close()
            return render_template('register.html', error='Bu email zaten kayıtlı!')
    
    return render_template('register.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# history page
@app.route('/history')
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    selected_symptoms = request.form.getlist('symptoms')

    input_vector = [1 if s in selected_symptoms else 0 for s in all_symptoms]
    input_array = np.array(input_vector).reshape(1, -1)

    prediction = model.predict(input_array)
    disease = le.inverse_transform(prediction)[0]

    proba = model.predict_proba(input_array).max()
    if proba >= 0.75:
        risk = 'High'
    elif proba >= 0.50:
        risk = 'Medium'
    else:
        risk = 'Low'

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (user_id, symptoms, predicted_disease, risk_level) VALUES (?, ?, ?, ?)",
        session['user_id'],
        ', '.join(selected_symptoms),
        disease,
        risk
    )
    conn.commit()
    conn.close()

    desc_row = desc_df[desc_df['Disease'].str.strip() == disease]
    description = desc_row['Description'].values[0] if len(desc_row) > 0 else "No description available."

    prec_row = prec_df[prec_df['Disease'].str.strip() == disease]
    if len(prec_row) > 0:
        precautions = [prec_row.iloc[0][f'Precaution_{i}'] for i in range(1, 5) if pd.notna(prec_row.iloc[0][f'Precaution_{i}'])]
    else:
        precautions = []

    return render_template('index.html',
                         disease=disease,
                         risk=risk,
                         selected_symptoms=selected_symptoms,
                         description=description,
                         precautions=precautions)
@app.route('/history.html')
def history_html():
    return redirect(url_for('history'))
@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        session['user_id']
    )
    predictions = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
    