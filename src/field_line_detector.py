import cv2
import numpy as np

class FieldLineDetector:
    def __init__(self, video_path, output_video_path):
        self.video_path = video_path
        self.output_video_path = output_video_path

    def process_video(self):
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Définir le codec et créer le writer pour la vidéo de sortie
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Appliquer une simple détection de lignes
            processed_frame = self._simple_line_detection(frame)

            # Ajouter la frame modifiée à la nouvelle vidéo
            out.write(processed_frame)

        cap.release()
        out.release()
        print(f"[INFO] Vidéo complète enregistrée dans '{self.output_video_path}' avec une détection de lignes simple.")

    def _simple_line_detection(self, frame):
        # Convertir l'image en niveaux de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer un flou pour réduire le bruit
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Appliquer la détection des bords avec Canny
        edges = cv2.Canny(blurred_frame, 50, 150)

        # Appliquer la transformée de Hough pour détecter les lignes
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=100,  # Seuil de base pour détecter des lignes claires
            minLineLength=50,  # Longueur minimale des lignes à détecter
            maxLineGap=10  # Écart maximum entre deux points pour relier une ligne
        )

        # Dessiner les lignes détectées en rouge sur l'image originale
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Lignes en rouge (BGR: (0, 0, 255)), épaisseur 2

        # Retourner l'image originale avec les lignes rouges superposées
        return frame
