# Handwritten Digit Classifier
CNN-based digit recognition (0–9) using TensorFlow/Keras + Flask + Web UI.

---

## Project Structure
```
digit_classifier/
├── train.py              # CNN training on MNIST
├── app.py                # Flask inference API
├── requirements.txt      # Python dependencies
├── static/
│   └── index.html        # Web UI (draw or upload)
├── digit_model.keras     # Saved model (after training)
└── training_history.png  # Accuracy/loss plot (after training)
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Step 1 — Train the Model

```bash
python train.py
```

- Downloads MNIST automatically (~11 MB)
- Trains a CNN with data augmentation, BatchNorm, and Dropout
- Saves the best checkpoint to `digit_model_best.keras`
- Saves final model to `digit_model.keras`
- Saves a training plot to `training_history.png`
- **Expected accuracy: ~99.3% on the test set**

---

## Step 2 — Start the API Server

```bash
python app.py
```

Server starts at `http://localhost:5000`

### API Endpoints

| Method | Endpoint   | Description              |
|--------|-----------|--------------------------|
| GET    | /health   | Health check             |
| POST   | /predict  | Classify a digit image   |

**POST /predict** — Request body:
```json
{ "image": "<base64-encoded PNG or JPEG>" }
```

**Response:**
```json
{
  "digit": 7,
  "confidence": 0.9984,
  "probabilities": [0.0001, 0.0002, ..., 0.9984, ...]
}
```

---

## Step 3 — Open the Web UI

With the Flask server running, open your browser at:
```
http://localhost:5000
```

Features:
- **Draw tab** — draw a digit with your mouse/finger on the canvas
- **Upload tab** — upload a PNG/JPEG of a handwritten digit
- Live bar chart of probabilities for all 10 digits
- Adjustable brush size

---

## Model Architecture

```
Input (28×28×1)
   ↓  Data Augmentation (rotation, zoom, shift)
   ↓  Conv2D(32) + BN + Conv2D(32) + BN + MaxPool + Dropout(0.25)
   ↓  Conv2D(64) + BN + Conv2D(64) + BN + MaxPool + Dropout(0.25)
   ↓  Conv2D(128) + BN + MaxPool + Dropout(0.25)
   ↓  GlobalAveragePooling2D
   ↓  Dense(256) + BN + Dropout(0.5)
   ↓  Dense(10, softmax)
```

---

## Tips for Best Results

- Draw digits large, centered in the canvas
- For uploaded images: plain white/light background with a dark digit works best
- If the model gets your uploaded image wrong, it may be inverted — the preprocessor auto-detects and flips it
- For doctor-handwriting specifically, you can fine-tune on custom data by modifying `train.py`

---

## Fine-tuning on Custom Data

To add your own digit images, place them in folders `data/0/`, `data/1/`, ..., `data/9/` and use:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory('data/', target_size=(28,28),
    color_mode='grayscale', class_mode='categorical', subset='training')
```

Then fine-tune the saved model:
```python
model = keras.models.load_model('digit_model.keras')
model.fit(train_gen, epochs=10)
model.save('digit_model_finetuned.keras')
```
