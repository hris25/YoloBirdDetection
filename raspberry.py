import os
import cv2
import time
import numpy as np
import pandas as pd
import requests
from ultralytics import YOLO
from datetime import datetime
import RPi.GPIO as GPIO

# Configuration du buzzer (GPIO 17)
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")  # Veillez à ce que ce modèle soit bien disponible

# Créer le dossier de sortie s’il n’existe pas
os.makedirs("videos", exist_ok=True)
os.makedirs("output", exist_ok=True)

def detect_bird_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    dict_results = {
        'frame_idx' : [],
        'birds_in_frame' : [],
        'prob_min' : [],
        'prob_max' : [],
        'prob_avg' : []
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        classes = results.boxes.cls
        scores = results.boxes.conf
        bird_scores = [float(scores[i]) for i in range(len(classes)) if int(classes[i]) == 14]
        birds_in_frame = len(bird_scores)

        if birds_in_frame > 0:
            prob_min = min(bird_scores)
            prob_max = max(bird_scores)
            prob_avg = sum(bird_scores) / birds_in_frame
        else:
            prob_min = prob_max = prob_avg = 0.0

        dict_results['frame_idx'].append(frame_idx)
        dict_results['birds_in_frame'].append(birds_in_frame)
        dict_results['prob_min'].append(prob_min)
        dict_results['prob_max'].append(prob_max)
        dict_results['prob_avg'].append(prob_avg)

        annotated_frame = results.plot()
        cv2.imwrite(os.path.join("output", f"frame_{frame_idx:04d}.jpg"), annotated_frame)
        frame_idx += 1

    cap.release()

    df_result = pd.DataFrame(dict_results)
    output = df_result[df_result['birds_in_frame']==df_result['birds_in_frame'].max()].to_dict('records')[0]
    return output

def send_alert(video_path):
    url = "https://server-agriproject.onrender.com/api/detections"
    files = {"video": open(video_path, "rb")}
    try:
        response = requests.post(url, files=files)
        print("[INFO] Alerte envoyée :", response.status_code)
    except Exception as e:
        print("[ERROR] Échec de l'envoi de l'alerte:", e)

def record_video(duration=5):
    filename = f"videos/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    return filename

def main_loop():
    try:
        while True:
            video_file = record_video(duration=5)
            print("[INFO] Vidéo capturée :", video_file)

            result = detect_bird_in_video(video_file)
            print("[INFO] Résultat :", result)

            if result['birds_in_frame'] > 10:
                print("[ALERTE] Trop d'oiseaux détectés. Activation du buzzer.")
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(BUZZER_PIN, GPIO.LOW)

                send_alert(video_file)

            time.sleep(1)  # pause avant la prochaine boucle

    except KeyboardInterrupt:
        print("[INFO] Arrêt du système.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main_loop()