import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import KMeans
import os

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Chemin de la vidéo (Remplacer par le chemin de votre vidéo)
video_path = 'data/videos/test.mp4'
cap = cv2.VideoCapture(video_path)

# Classe "person" pour YOLO
CLASS_ID_PERSON = 0

# Initialisation pour la frame à utiliser
frame_id = 0  # Compteur de frames

# Sélectionner la frame sur laquelle vous voulez effectuer la détection (par exemple, frame 10)
target_frame_id = 10

# Fonction pour appliquer K-Means en excluant la couleur verte
def get_dominant_color_no_green(image, k=2):
    """Applique K-Means pour obtenir la couleur dominante, tout en supprimant le vert du classificateur."""
    # Convertir l'image de l'espace BGR à HSV pour mieux isoler la couleur verte
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Définir les limites pour la couleur verte (tonalité de l'herbe)
    lower_green = np.array([35, 40, 40])  # Plage inférieure pour la couleur verte (HSV)
    upper_green = np.array([85, 255, 255])  # Plage supérieure pour la couleur verte (HSV)
    
    # Créer un masque pour filtrer les pixels verts
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Remplacer les pixels verts par du noir (ou toute autre couleur neutre)
    image[green_mask != 0] = [0, 0, 0]  # Remplacer par du noir pour ignorer les verts

    # Redimensionner l'image pour accélérer le traitement
    image_resized = cv2.resize(image, (50, 50))
    
    # Réorganiser l'image en un tableau 2D de pixels (longueur * largeur, 3)
    pixel_values = image_resized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Appliquer K-Means pour la segmentation
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixel_values)

    # Trouver la couleur dominante (la plus fréquente)
    dominant_color = kmeans.cluster_centers_[0]  # Prendre la couleur dominante du cluster principal
    
    return dominant_color, kmeans, image_resized, green_mask  # On retourne aussi l'image modifiée et le masque

# Attendre d'arriver à la frame cible
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id == target_frame_id:
        # Afficher la frame cible
        print(f"[INFO] Affichage de la frame {target_frame_id}")
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Target Frame {target_frame_id}')
        plt.show()
        
        # Utiliser YOLO pour faire des prédictions sur la frame
        print(f"[INFO] Utilisation de YOLO pour la détection sur la frame {frame_id}...")
        results = model.predict(frame, imgsz=640, conf=0.4)

        # Détails sur les détections
        print(f"[INFO] Nombre de détections : {len(results[0].boxes)}")
        
        # Parcourir les détections de YOLO
        for idx, result in enumerate(results[0].boxes):
            class_id = int(result.cls)
            if class_id == CLASS_ID_PERSON:
                print(f"[INFO] Détection d'une personne à la position {idx} avec les coordonnées : {result.xyxy[0]}")

                # On va traiter le premier objet détecté (person)
                x1, y1, x2, y2 = map(int, result.xyxy[0])

                # Extraire la région correspondant à la personne
                print(f"[INFO] Extraction de la région d'intérêt (ROI) pour la personne...")
                roi = frame[y1:y2, x1:x2]

                # Trouver la couleur dominante de la personne détectée sans prendre en compte le vert
                dominant_color, kmeans, modified_image, green_mask = get_dominant_color_no_green(roi)

                # Visualisation de la région sans le vert
                print(f"[INFO] Visualisation de la personne sans les pixels verts et K-Means...")
                plt.figure(figsize=(15, 10))
                
                # Afficher la personne originale
                plt.subplot(2, 2, 1)
                plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                plt.title('Person ROI (Original)')

                # Afficher la personne sans le vert
                plt.subplot(2, 2, 2)
                plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
                plt.title('Person ROI (No Green)')

                # Visualisation du masque vert
                plt.subplot(2, 2, 3)
                plt.imshow(green_mask, cmap='gray')
                plt.title('Green Mask')

                # Visualisation K-Means
                plt.subplot(2, 2, 4)
                centers = np.uint8(kmeans.cluster_centers_)
                labels = kmeans.labels_
                segmented_image = centers[labels.flatten()]
                segmented_image = segmented_image.reshape(modified_image.shape)
                plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                plt.title('K-Means Segmentation')

                plt.show()

                # Terminer après avoir traité le premier objet
                #break
    frame_id += 1

cap.release()
print("[INFO] Fin du traitement.")
