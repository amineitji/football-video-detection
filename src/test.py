import cv2
import numpy as np

# Chemins pour la vidéo d'entrée et la vidéo de sortie finale avec annotation
input_video_path = 'data/videos/test_1.mp4'
output_annotated_video_path = 'data/videos/test_1_with_ball.mp4'

# Charger la vidéo d'entrée
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
else:
    # Paramètres pour la vidéo de sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_annotated_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # Fin de la vidéo

        # Conversion en espace HSV pour détecter le vert et masquer le terrain
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Plage pour la couleur verte du terrain (à ajuster si nécessaire)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Inverser le masque pour garder tout sauf le terrain vert
        non_green_mask = cv2.bitwise_not(mask)
        
        # Appliquer le masque inversé pour isoler les éléments non verts (incluant le ballon)
        non_green_areas = cv2.bitwise_and(frame, frame, mask=non_green_mask)
        
        # Conversion en niveaux de gris pour la détection de cercle
        gray_non_green = cv2.cvtColor(non_green_areas, cv2.COLOR_BGR2GRAY)
        
        # Détection de cercles potentiels (ballon) avec HoughCircles
        circles = cv2.HoughCircles(
            gray_non_green,
            cv2.HOUGH_GRADIENT,
            dp=1.2,  # Résolution
            minDist=30,  # Distance minimale entre cercles détectés
            param1=50,  # Seuil de Canny
            param2=30,  # Accumulateur pour la détection de cercles (ajustable)
            minRadius=5,  # Rayon minimal du cercle (à ajuster selon la taille du ballon)
            maxRadius=30  # Rayon maximal du cercle (à ajuster selon la taille du ballon)
        )
        
        # Si un cercle est détecté, dessiner le cercle et l'annotation
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles[:1]:  # Prendre uniquement le premier cercle détecté
                # Dessiner le cercle en jaune
                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                # Ajouter l'annotation "ball"
                cv2.putText(frame, "ball", (x + r + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Écrire le cadre annoté dans la vidéo de sortie
        out.write(frame)

    # Libérer les ressources
    cap.release()
    out.release()
    print("Annotation du ballon terminée, vidéo enregistrée.")
