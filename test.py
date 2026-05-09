from dl_model import predict_digit

# ------------------------
# Audio File
# ------------------------

audio_file = "dataset/0/0_george_8.wav"

# ------------------------
# Predict
# ------------------------

digit, confidence = predict_digit(audio_file)

# ------------------------
# Print Result
# ------------------------

print("Predicted Digit:", digit)

print(f"Confidence: {confidence:.2%}")