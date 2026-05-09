import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset

# ------------------------
# Load dataset
# ------------------------

X, y = load_dataset()

# ------------------------
# Normalize
# ------------------------

X = X / np.max(X)

# ------------------------
# One Hot Encoding
# ------------------------

y = to_categorical(y)

# ------------------------
# Split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------
# Build Model
# ------------------------

model = Sequential([

    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')

])

# ------------------------
# Compile
# ------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------
# Train
# ------------------------

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ------------------------
# Save Model
# ------------------------

model.save("models/digit_model.keras")

print("Model saved successfully!")