import os
import pygame
import threading
import cv2
import numpy as np
import re
import speech_recognition as sr
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from gtts import gTTS
import google.generativeai as genai

genai.configure(api_key='AIzaSyDlDLNXXLb7XQK8zLwGCpt2BwSjRlcbq3k')
model = genai.GenerativeModel('gemini-1.5-flash-latest')

captured_image = None
speech_text = None
is_running = False  

app = Flask(__name__)
socketio = SocketIO(app)

def emit_status_update(message):
    socketio.emit('status_update', {'message': message})

def capture_image_from_webcam():
    global captured_image
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    emit_status_update("Webcam started.")
    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        captured_image = frame
        cv2.imshow('Webcam', captured_image)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            emit_status_update("Image captured.")
            break  

    cap.release()
    cv2.destroyAllWindows()

def recognize_speech():
    global speech_text
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        emit_status_update("Microphone started.")
        
        while is_running:
            try:
                audio = recognizer.listen(source, timeout=5)
                speech_text = recognizer.recognize_google(audio)
                emit_status_update(f"Recognized Speech: {speech_text}")
                process_query(speech_text) 
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Could not understand audio.")
                continue
            except sr.RequestError as e:
                print(f"Request error: {e}")
                continue

def query_gemini_with_image(image, user_query):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_bytes = buffer.tobytes()

        response = model.generate_content(
            [user_query, {"mime_type": "image/jpeg", "data": img_bytes}]
        )

        return response.text
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None

def process_query(user_query):
    global captured_image
    if user_query and ("image" in user_query or "picture" in user_query or "photo" in user_query):
        if captured_image is not None:
            response_text = query_gemini_with_image(captured_image, user_query)
            if response_text:
                response_text = get_first_n_tokens(response_text)

                tts_thread = threading.Thread(target=speak_text, args=(response_text,))
                tts_thread.start()

                
                socketio.emit('query_response', {'query': user_query, 'response': response_text})

        else:
            emit_status_update("No image captured. Please capture an image using 'c'.")
    else:
        try:
            response_text = model.generate_content([user_query]).text
            if response_text:
                response_text = get_first_n_tokens(response_text)

                tts_thread = threading.Thread(target=speak_text, args=(response_text,))
                tts_thread.start()

                socketio.emit('query_response', {'query': user_query, 'response': response_text})
        except Exception as e:
            print(f"Error querying Gemini: {e}")


def get_first_n_tokens(response_text, n=15):
    cleaned_response = re.sub(r'[^a-zA-Z0-9\s,?!]', '', response_text)
    tokens = cleaned_response.split()
    truncated_tokens = tokens[:n]
    truncated_response = ' '.join(truncated_tokens)
    return truncated_response + '.' if truncated_response else ""

def speak_text(text):
    if text:
        tts = gTTS(text=text, lang='en')
        audio_file = os.path.join(os.path.expanduser('~'), 'Documents', 'response.mp3')
        tts.save(audio_file)
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        os.remove(audio_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_process():
    global is_running
    if not is_running:
        is_running = True
        webcam_thread = threading.Thread(target=capture_image_from_webcam)
        webcam_thread.start()

        speech_thread = threading.Thread(target=recognize_speech)
        speech_thread.start()

        return jsonify({"status": "Started"})
    else:
        return jsonify({"status": "Already running"})

@app.route('/end', methods=['POST'])
def end_process():
    global is_running
    if is_running:
        is_running = False
        return jsonify({"status": "Stopped"})
    else:
        return jsonify({"status": "Not running"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
