import os
import cv2
import time
import numpy as np
import pandas as pd
import requests
from ultralytics import YOLO
import subprocess
from datetime import datetime
import RPi.GPIO as GPIO
import mimetypes
import json


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

def send_alert(video_path, media_path):
    # URL de l'API
    url = "https://server-agriproject.onrender.com/api/detections"

    
    # Vérifier si les fichiers existent
    if not os.path.exists(media_path):
        print(f"Erreur: Le fichier {media_path} n'existe pas")
        return
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier {video_path} n'existe pas")
        return
    
    print(f"Taille du fichier media: {os.path.getsize(media_path)} bytes")
    print(f"Taille du fichier tram: {os.path.getsize(video_path)} bytes")
    
    # Déterminer les types MIME
    media_mime = mimetypes.guess_type(media_path)[0] or 'image/jpeg'
    video_mime = mimetypes.guess_type(video_path)[0] or 'video/mp4'
    
    print(f"Type MIME media: {media_mime}")
    print(f"Type MIME video: {video_mime}")
    
    # Préparer les fichiers pour l'envoi avec les types MIME corrects
    files = {
        'media': ('media', open(media_path, 'rb'), media_mime),
        'video': ('video', open(video_path, 'rb'), video_mime)
    }
    
    # Données supplémentaires
    data = {
        'systeme_id': 2
    }
    
    # Headers pour le débogage
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Python/Requests',
        'Connection': 'keep-alive'
    }
    
    # Configuration de la session
    session = requests.Session()
    session.max_redirects = 5
    
    try:
        print("Envoi de la requête...")
        # Faire la requête POST avec timeout et retry
        max_retries = 3
        retry_delay = 5  # secondes
        
        for attempt in range(max_retries):
            try:
                response = session.post(
                    url, 
                    files=files, 
                    data=data,
                    headers=headers,
                    timeout=120,  # 2 minutes de timeout
                    allow_redirects=True
                )
                
                print(f"Tentative {attempt + 1}/{max_retries}")
                print(f"Code de statut: {response.status_code}")
                print(f"Headers de réponse: {dict(response.headers)}")
                
                if response.status_code == 201:
                    print("Succès! Détection créée avec succès")
                    print("Réponse:", response.json())
                    break
                elif response.status_code == 502:
                    print("Erreur 502 détectée, nouvelle tentative...")
                    if attempt < max_retries - 1:
                        print(f"Attente de {retry_delay} secondes avant la prochaine tentative...")
                        time.sleep(retry_delay)
                        continue
                else:
                    print(f"Erreur: {response.status_code}")
                    try:
                        print("Détails:", response.json())
                    except json.JSONDecodeError:
                        print("Contenu brut de la réponse:", response.text)
                    break
                    
            except requests.exceptions.Timeout:
                print(f"Timeout sur la tentative {attempt + 1}")
                if attempt < max_retries - 1:
                    print(f"Attente de {retry_delay} secondes avant la prochaine tentative...")
                    time.sleep(retry_delay)
                continue
                
    except requests.exceptions.ConnectionError:
        print("Erreur: Impossible de se connecter au serveur. Vérifiez votre connexion internet.")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête: {e}")
    except Exception as e:
        print(f"Erreur inattendue: {e}")
    finally:
        # Fermer les fichiers
        files['media'][1].close()
        files['tram'][1].close()
        session.close()

def record_video(duration=5):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    h264_path = f"videos/tmp_{timestamp}.h264"
    mp4_path = f"videos/detection_{timestamp}.mp4"

    # Commande pour enregistrer la vidéo avec libcamera-vid
    record_cmd = [
        "libcamera-vid",
        "-t", str(duration * 1000),  # durée en ms
        "-o", h264_path,
        "--width", "1920",
        "--height", "1080",
        "--framerate", "30"
    ]

    try:
        print(f"[INFO] Enregistrement vidéo avec libcamera-vid ({duration}s)...")
        subprocess.run(record_cmd, check=True)

        # Conversion en MP4 avec ffmpeg
        convert_cmd = [
            "ffmpeg",
            "-framerate", "30",
            "-i", h264_path,
            "-c", "copy",
            mp4_path
        ]
        subprocess.run(convert_cmd, check=True)

        os.remove(h264_path)
        print(f"[INFO] Vidéo enregistrée et convertie : {mp4_path}")
        return mp4_path

    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] Lors de l'enregistrement ou la conversion : {e}")
        return None

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

                send_alert(f'videos/{video_file}', f'output/frame_{result['frame_idx']:04d}.jpg')

            time.sleep(1)  # pause avant la prochaine boucle

    except KeyboardInterrupt:
        print("[INFO] Arrêt du système.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main_loop()