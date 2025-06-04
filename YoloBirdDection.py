from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np

video_path = "video.mp4"
def detect_bird_in_video(video_path):
    # Charger le modèle YOLO
    model = YOLO("yolo11n.pt")  # Remplace "yolo11n.pt" s’il n’existe pas
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
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
    
        results = model(frame)[0]  # Résultat YOLOv8
    
        # Extraire les classes et scores
        classes = results.boxes.cls
        scores = results.boxes.conf
    
        # Filtrer uniquement les oiseaux (classe 14 dans COCO)
        bird_scores = [float(scores[i]) for i in range(len(classes)) if int(classes[i]) == 14]
        birds_in_frame = len(bird_scores)
    
        if birds_in_frame > 0:
            prob_min = min(bird_scores)
            prob_max = max(bird_scores)
            prob_avg = sum(bird_scores) / birds_in_frame
        else:
            prob_min = prob_max = prob_avg = 0.0
    
        # Affichage console
        #print(f"Frame {frame_idx:04d} - Birds: {birds_in_frame}, Min: {prob_min:.2f}, Max: {prob_max:.2f}, Avg: {prob_avg:.2f}")
        dict_results['frame_idx'].append(frame_idx)
        dict_results['birds_in_frame'].append(birds_in_frame)
        dict_results['prob_min'].append(prob_min)
        dict_results['prob_max'].append(prob_max)
        dict_results['prob_avg'].append(prob_avg)
    
        # Sauvegarde image annotée
        annotated_frame = results.plot()
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), annotated_frame)
    
        frame_idx += 1
    
    df_result = pd.DataFrame(dict_results)
    output = df_result[df_result['birds_in_frame']==df_result['birds_in_frame'].max()].to_dict('records')[0]
    
    cap.release()
    #print("Traitement terminé.")

    print(f"Name:    frame_{output['frame_idx']:04d}.jpg")
    return 'The output is : ' + str(output)
print(detect_bird_in_video(video_path))