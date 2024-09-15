import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Chemin de la vidéo
video_path = 'data/videos/test.mp4'
output_video_path = 'data/videos/output_with_teams.mp4'
cap = cv2.VideoCapture(video_path)

# Classe "person" pour YOLO
CLASS_ID_PERSON = 0

# Tracker les personnes et leur équipe
person_team_tracker = defaultdict(lambda: {"color": None, "team": None})

# Fonction pour appliquer K-Means en excluant la couleur verte et noire
def get_dominant_color_no_green_or_black(image, k=2, threshold=40):
    # Convertir l'image en HSV pour isoler la couleur verte
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Définir les limites pour la couleur verte
    lower_green = np.array([35, 40, 40])  # Limite inférieure
    upper_green = np.array([85, 255, 255])  # Limite supérieure
    
    # Créer un masque pour filtrer les pixels verts
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Remplacer les pixels verts par du noir
    image[green_mask != 0] = [0, 0, 0] 

    # Redimensionner l'image pour accélérer le traitement
    image_resized = cv2.resize(image, (50, 50))
    
    # Réorganiser l'image en un tableau 2D de pixels
    pixel_values = image_resized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Appliquer K-Means
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixel_values)

    # Filtrer les couleurs proches du noir
    valid_colors = [center for center in kmeans.cluster_centers_ if np.linalg.norm(center) > threshold]

    # Si aucune couleur valide, retourner un gris par défaut
    dominant_color = valid_colors[0] if valid_colors else np.array([128, 128, 128])
    
    return dominant_color

# Réouvrir la vidéo pour analyse et enregistrement
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Définir le writer pour la vidéo de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_id = 0
team_labels = {0: "Team_1", 1: "Team_2"}
team_colors = {0: (0, 0, 255), 1: (255, 0, 0)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prédictions pour la frame
    results = model.predict(frame, imgsz=640, conf=0.4)

    # Liste pour les boxes et couleurs dominantes
    person_boxes = []
    frame_dominant_colors = []

    for idx, result in enumerate(results[0].boxes):
        class_id = int(result.cls)
        if class_id == CLASS_ID_PERSON:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            person_boxes.append([x1, y1, x2, y2])

            # Extraire la région de la personne
            roi = frame[y1:y2, x1:x2]

            # Couleur dominante
            dominant_color = get_dominant_color_no_green_or_black(roi)

            # Stocker la couleur
            frame_dominant_colors.append(dominant_color)

    if frame_dominant_colors:
        # Vérifier les couleurs et assigner aux équipes en fonction des frames précédentes
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            dominant_color = frame_dominant_colors[idx]
            best_match = None
            min_distance = float('inf')

            # Chercher une personne proche en couleur dans les frames précédentes
            for person_id, info in person_team_tracker.items():
                color_dist = np.linalg.norm(dominant_color - info['color'])
                if color_dist < min_distance:
                    best_match = person_id
                    min_distance = color_dist

            # Si un match est trouvé, assigner la même équipe
            if best_match is not None and min_distance < 50:  # seuil pour une correspondance de couleur
                team_idx = person_team_tracker[best_match]['team']
            else:
                # Sinon, appliquer K-Means pour déterminer l'équipe
                if frame_id == 0:
                    kmeans_teams = KMeans(n_clusters=2, random_state=0)
                    kmeans_teams.fit(frame_dominant_colors)
                    team_idx = kmeans_teams.labels_[idx]
                else:
                    team_idx = 0 if len(frame_dominant_colors) % 2 == 0 else 1

                # Enregistrer cette personne dans le tracker
                person_team_tracker[frame_id] = {"color": dominant_color, "team": team_idx}

            # Dessiner la personne avec son label d'équipe
            team = team_labels[team_idx]
            team_color = team_colors[team_idx]
            cv2.putText(frame, team, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, team_color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)

    # Écrire la frame dans le fichier de sortie
    out.write(frame)

    frame_id += 1

cap.release()
out.release()
print(f"[INFO] Vidéo complète enregistrée dans '{output_video_path}' avec les labels d'équipe.")
