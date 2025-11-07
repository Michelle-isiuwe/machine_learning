# app.py
import os
import sqlite3
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# ===============================
# 1. Flask setup
# ===============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# 2. Load the trained PyTorch model
# ===============================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load model weights
model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load("face_emotionModel.pth", map_location='cpu'))
model.eval()

# Emotion labels (same order as training dataset folders)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ===============================
# 3. Define image transformation
# ===============================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# ===============================
# 4. Initialize SQLite database
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
# 5. Flask Routes
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

        # Load and preprocess image
        img = Image.open(filepath)
        img_tensor = transform(img).unsqueeze(0)

        # Predict emotion
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTIONS[predicted.item()]

        # Save user info to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, matric_no, department, emotion, image_path) VALUES (?, ?, ?, ?, ?)",
                  (name, matric_no, department, emotion, filepath))
        conn.commit()
        conn.close()

        message = f"You look {emotion}! 😊"
        return render_template('index.html', message=message, image_path=filepath)

    else:
        return render_template('index.html', message="Please upload an image.")

# ===============================
# 6. Run the app
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
