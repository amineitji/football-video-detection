import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import os

# Créer les dossiers pour les frames et les labels s'ils n'existent pas
if not os.path.exists('data/frames'):
    os.makedirs('data/frames')
if not os.path.exists('data/labels'):
    os.makedirs('data/labels')

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Chemin de la vidéo
video_path = 'data/videos/test.mp4'  # Remplacer par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

# Classe "person" pour YOLO
CLASS_ID_PERSON = 0

# Couleurs de référence pour les équipes
team_1_color = np.array([0, 0, 255])  # Rouge, exemple pour Team 1 (en BGR)
team_2_color = np.array([255, 255, 255])  # Blanc, exemple pour Team 2 (en BGR)

# Initialiser un dictionnaire pour stocker les trackers pour chaque objet détecté
trackers = {}

def get_dominant_color(image, k=2):
    """Applique K-Means pour obtenir la couleur dominante de la personne détectée."""
    # Redimensionner l'image pour accélérer le traitement
    image = cv2.resize(image, (50, 50))

    # Réorganiser l'image en un tableau 2D de pixels (longueur * largeur, 3)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Appliquer K-Means pour la segmentation
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixel_values)

    # Trouver la couleur dominante (la plus fréquente)
    dominant_color = kmeans.cluster_centers_[kmeans.labels_.mean().round().astype(int)]

    return dominant_color

def determine_team(dominant_color, team_1_color, team_2_color):
    """Détermine si une personne appartient à Team 1 ou Team 2 en fonction de la couleur dominante."""
    dist_team1 = np.linalg.norm(dominant_color - team_1_color)
    dist_team2 = np.linalg.norm(dominant_color - team_2_color)

    if dist_team1 < dist_team2:
        return "team_1"
    else:
        return "team_2"

frame_id = 0  # Pour compter les frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Obtenir les dimensions de la frame
    height, width, _ = frame.shape
    
    # Utiliser YOLO pour faire des prédictions sur la frame
    results = model.predict(frame, imgsz=640, conf=0.4)
    
    # Si des objets sont détectés, ajouter des trackers pour les nouveaux objets
    for result in results[0].boxes:
        class_id = int(result.cls)
        if class_id == CLASS_ID_PERSON:  # Filtrer pour ne prendre que les personnes
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            
            # Créer un tracker KCF pour cette personne si ce n'est pas déjà fait
            if result.id not in trackers:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                trackers[result.id] = tracker
    
    # Mettre à jour chaque tracker et déterminer l'équipe à laquelle la personne appartient
    for obj_id, tracker in list(trackers.items()):
        success, bbox = tracker.update(frame)
        if success:
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h

            # Extraire la région correspondant à la personne
            roi = frame[y1:y2, x1:x2]

            # Trouver la couleur dominante de la personne détectée
            dominant_color = get_dominant_color(roi)
            
            # Déterminer à quelle équipe appartient la personne
            team = determine_team(dominant_color, team_1_color, team_2_color)
            
            # ID de la classe en fonction de l'équipe
            class_id = 0 if team == "team_1" else 1
            
            # Sauvegarder les annotations YOLO
            label_file_path = f"data/labels/frame_{frame_id}.txt"
            with open(label_file_path, 'w') as f:
                # Convertir les coordonnées en format YOLO (normalisé)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = w / width
                bbox_height = h / height
                
                # Sauvegarder l'annotation
                label = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                f.write(label)
            
            # Dessiner les boîtes englobantes sur la frame
            team_label = 'team_1' if class_id == 0 else 'team_2'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id} - {team_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            # Si le tracker ne fonctionne plus, supprimer l'objet du suivi
            del trackers[obj_id]

    # Sauvegarder les frames dans data/frames
    frame_file_path = f"data/frames/frame_{frame_id}.jpg"
    cv2.imwrite(frame_file_path, frame)

    # Afficher la vidéo avec les objets suivis (facultatif)
    cv2.imshow('YOLOv8 + KCF + Team Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Libérer la vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

print("Toutes les frames ont été traitées et enregistrées.")
