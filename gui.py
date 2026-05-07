import customtkinter as ctk
import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import threading
import time
import os

# ==============================
# APP SETTINGS
# ==============================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.geometry("650x500")
app.title("Spoken Digit Recognition")

# ==============================
# VARIABLES
# ==============================

fs = 44100
duration = 5
recording = None
is_recording = False

# ==============================
# CREATE RECORDINGS FOLDER
# ==============================

if not os.path.exists("recordings"):
    os.makedirs("recordings")

# ==============================
# RECORD AUDIO FUNCTION
# ==============================

def start_recording():

    thread = threading.Thread(target=record_audio)

    thread.start()

# ------------------------------

def record_audio():

    global recording
    global is_recording

    is_recording = True

    record_btn.configure(state="disabled")

    progress_bar.set(0)

    for i in range(duration, 0, -1):

        status_label.configure(
            text=f"🎤 Recording... {i}"
        )

        progress_bar.set(
            (duration - i) / duration
        )

        app.update()

        time.sleep(1)

    # Actual Recording
    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    # Save WAV
    write(
        "recordings/test.wav",
        fs,
        recording
    )

    # Finish UI
    progress_bar.set(1)

    status_label.configure(
        text="✅ Recording Saved Successfully!"
    )

    record_btn.configure(state="normal")

    is_recording = False

# ==============================
# PLAY AUDIO
# ==============================

def play_audio():

    try:

        pygame.mixer.init()

        pygame.mixer.music.load(
            "recordings/test.wav"
        )

        pygame.mixer.music.play()

        status_label.configure(
            text="▶️ Playing Audio..."
        )

    except:

        status_label.configure(
            text="❌ No Recording Found"
        )

# ==============================
# HEADER
# ==============================

title = ctk.CTkLabel(
    app,
    text="🎙 Spoken Digit Recognition",
    font=("Arial", 30, "bold")
)

title.pack(pady=25)

# ==============================
# DESCRIPTION
# ==============================

description = ctk.CTkLabel(
    app,
    text="Record your voice and test the AI model",
    font=("Arial", 16),
    text_color="gray"
)

description.pack(pady=5)

# ==============================
# MAIN FRAME
# ==============================

main_frame = ctk.CTkFrame(
    app,
    width=500,
    height=300,
    corner_radius=25
)

main_frame.pack(pady=30)

# ==============================
# RECORD BUTTON
# ==============================

record_btn = ctk.CTkButton(
    main_frame,
    text="🎤 Start Recording",
    command=start_recording,
    width=250,
    height=55,
    corner_radius=20,
    font=("Arial", 18, "bold")
)

record_btn.pack(pady=30)

# ==============================
# PLAY BUTTON
# ==============================

play_btn = ctk.CTkButton(
    main_frame,
    text="▶️ Play Recording",
    command=play_audio,
    width=250,
    height=55,
    corner_radius=20,
    font=("Arial", 18, "bold")
)

play_btn.pack(pady=10)

# ==============================
# PROGRESS BAR
# ==============================

progress_bar = ctk.CTkProgressBar(
    main_frame,
    width=350,
    height=20,
    corner_radius=20
)

progress_bar.pack(pady=25)

progress_bar.set(0)

# ==============================
# STATUS LABEL
# ==============================

status_label = ctk.CTkLabel(
    main_frame,
    text="🟢 Ready",
    font=("Arial", 17)
)

status_label.pack(pady=10)

# ==============================
# FOOTER
# ==============================

footer = ctk.CTkLabel(
    app,
    text="AI DSP Project • ERU",
    font=("Arial", 13),
    text_color="gray"
)

footer.pack(side="bottom", pady=15)

# ==============================
# RUN APP
# ==============================

app.mainloop()