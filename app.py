# app.py
import os
import sqlite3
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
from fer import FER

# ===============================
# 1. Flask setup
# ===============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# 2. Initialize emotion detector
# ===============================
detector = FER(mtcnn=True)  # Uses MTCNN for better face detection

# ===============================
# 3. Initialize SQLite database
# ===============================
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    matric_no TEXT,
                    department TEXT,
                    emotion TEXT,
                    image_path TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# ===============================
# 4. Flask Routes
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    matric_no = request.form['matric_no']
    department = request.form['department']
    image_file = request.files['image']

    if image_file:
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        # Read image with OpenCV
        img = cv2.imread(filepath)

        # Detect emotions
        results = detector.detect_emotions(img)

        if results:
            # Get the most prominent emotion
            top_emotion, score = detector.top_emotion(img)
            emotion = top_emotion
        else:
            emotion = "no face detected"

        # Save to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, matric_no, department, emotion, image_path) VALUES (?, ?, ?, ?, ?)",
                  (name, matric_no, department, emotion, filepath))
        conn.commit()
        conn.close()

        message = f"You look {emotion}!" if emotion != "no face detected" else "Couldn't detect a face. Try again?"
        return render_template('index.html', message=message, image_path=filepath)

    else:
        return render_template('index.html', message="Please upload an image.")

# ===============================
# 5. Run the app
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
