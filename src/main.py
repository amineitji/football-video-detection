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

# Obtenir les informations de la vidéo pour créer une vidéo de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Sortie vidéo
output_video = cv2.VideoWriter('data/output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Classe "person" pour YOLO
CLASS_ID_PERSON = 0

# Dictionnaire pour stocker les équipes attribuées par personne (utilisé pour garder les joueurs dans la même équipe)
player_teams = {}

def get_dominant_color_no_green(image, k=5):
    """Applique K-Means pour obtenir la couleur dominante en supprimant le vert du classificateur et en utilisant plusieurs clusters."""
    # Convertir l'image de BGR à HSV pour identifier la couleur verte
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les limites pour la couleur verte en HSV
    lower_green = np.array([35, 40, 40])  # Limite inférieure pour la couleur verte
    upper_green = np.array([85, 255, 255])  # Limite supérieure pour la couleur verte

    # Créer un masque pour filtrer les pixels verts
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Remplacer les pixels verts par du noir (ou toute autre couleur neutre)
    image[green_mask != 0] = [0, 0, 0]  # Remplacement des pixels verts par du noir

    # Redimensionner l'image pour accélérer le traitement
    image_resized = cv2.resize(image, (50, 50))

    # Réorganiser l'image en un tableau 2D de pixels (longueur * largeur, 3)
    pixel_values = image_resized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Appliquer K-Means avec plusieurs clusters pour capter plus de nuances
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixel_values)

    # Trouver la couleur dominante (la plus fréquente)
    dominant_colors = kmeans.cluster_centers_  # Prendre les couleurs dominantes de tous les clusters
    
    return dominant_colors

def assign_team(player_id, dominant_colors, team_assignment, team_1_color, team_2_color):
    """Attribue une équipe à un joueur en fonction des couleurs dominantes."""
    if player_id not in player_teams:
        # Utiliser une métrique pour évaluer la proximité des couleurs dominantes avec les couleurs des équipes
        distances_to_team_1 = np.linalg.norm(dominant_colors - team_1_color, axis=1)
        distances_to_team_2 = np.linalg.norm(dominant_colors - team_2_color, axis=1)
        
        # La distance la plus faible indique à quelle équipe le joueur appartient
        if np.min(distances_to_team_1) < np.min(distances_to_team_2):
            player_teams[player_id] = 0  # team_1
        else:
            player_teams[player_id] = 1  # team_2
    return player_teams[player_id]

# Couleurs de référence pour les équipes (par exemple, rouge pour team_1, bleu pour team_2)
team_1_color = np.array([220, 20, 60])  # Rouge
team_2_color = np.array([30, 144, 255])  # Bleu

# Initialisation des variables pour stocker les couleurs dominantes et boîtes englobantes
dominant_colors = []
person_boxes = []
player_ids = []  # Liste des IDs des joueurs

frame_id = 0  # Pour compter les frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Créer une copie de la frame pour effacer les annotations de la frame précédente
    frame_copy = frame.copy()  # IMPORTANT: on réinitialise la frame ici
    
    # Obtenir les dimensions de la frame
    height, width, _ = frame.shape
    
    # Utiliser YOLO pour faire des prédictions sur la frame
    results = model.predict(frame, imgsz=640, conf=0.4)
    
    # Parcourir les détections de YOLO
    for result in results[0].boxes:
        class_id = int(result.cls)
        if class_id == CLASS_ID_PERSON:  # Ne garder que les personnes
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            
            # Extraire la région correspondant à la personne
            roi = frame[y1:y2, x1:x2]

            # Trouver les couleurs dominantes de la personne détectée en supprimant le vert
            dominant_colors = get_dominant_color_no_green(roi)

            # Ajouter la couleur dominante et la boîte englobante à la liste
            person_boxes.append((x1, y1, x2, y2))
            player_ids.append(result.id)  # Utiliser l'ID unique attribué par YOLO

    # Si suffisamment de personnes sont détectées, effectuer le clustering
    if len(player_ids) >= 2:  # Au moins deux personnes doivent être détectées

        # Afficher les résultats sur la vidéo
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            player_id = player_ids[i]
            assigned_team = assign_team(player_id, dominant_colors, None, team_1_color, team_2_color)

            # Attribuer l'équipe basée sur l'historique
            team = 'team_1' if assigned_team == 0 else 'team_2'

            # Dessiner les boîtes englobantes et l'équipe sur la frame
            color = (0, 255, 0) if team == 'team_1' else (0, 0, 255)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, f'{team}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Sauvegarder l'annotation dans data/labels
            label_file_path = f"data/labels/frame_{frame_id}.txt"
            with open(label_file_path, 'a') as f:
                # Convertir les coordonnées en format YOLO (normalisé)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                # ID de la classe en fonction de l'équipe
                class_id = 0 if team == "team_1" else 1

                # Sauvegarder l'annotation
                label = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                f.write(label)

        # Sauvegarder la frame avec les annotations dans data/frames
        frame_file_path = f"data/frames/frame_{frame_id}.jpg"
        cv2.imwrite(frame_file_path, frame_copy)

        # Ajouter la frame annotée à la vidéo de sortie
        output_video.write(frame_copy)

    # Afficher la vidéo avec les équipes détectées en temps réel
    cv2.imshow('YOLOv8 + Clustering Teams', frame_copy)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

    # Réinitialiser les listes pour éviter l'accumulation des données
    person_boxes.clear()
    player_ids.clear()

# Libérer la vidéo et fermer les fenêtres
cap.release()
output_video.release()
cv2.destroyAllWindows()

print("Clustering des équipes terminé, les frames et les annotations ont été enregistrées, et la vidéo de sortie a été générée.")
