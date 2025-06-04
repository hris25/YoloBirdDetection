import requests
import os
import json
import time
import mimetypes

def test_detection_api():
    # URL de l'API
    url = "https://server-agriproject.onrender.com/api/detections"
    
    # Chemin vers les fichiers
    media_path = "output/frame_0112.jpg"
    tram_path = "video.mp4"
    
    # Vérifier si les fichiers existent
    if not os.path.exists(media_path):
        print(f"Erreur: Le fichier {media_path} n'existe pas")
        return
    if not os.path.exists(tram_path):
        print(f"Erreur: Le fichier {tram_path} n'existe pas")
        return
    
    print(f"Taille du fichier media: {os.path.getsize(media_path)} bytes")
    print(f"Taille du fichier tram: {os.path.getsize(tram_path)} bytes")
    
    # Déterminer les types MIME
    media_mime = mimetypes.guess_type(media_path)[0] or 'image/jpeg'
    tram_mime = mimetypes.guess_type(tram_path)[0] or 'video/mp4'
    
    print(f"Type MIME media: {media_mime}")
    print(f"Type MIME tram: {tram_mime}")
    
    # Préparer les fichiers pour l'envoi avec les types MIME corrects
    files = {
        'media': ('media', open(media_path, 'rb'), media_mime),
        'tram': ('tram', open(tram_path, 'rb'), tram_mime)
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

if __name__ == "__main__":
    test_detection_api() 