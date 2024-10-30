import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from detection_football import FootballDetection

class TeamTracker(FootballDetection):
    def __init__(self, model_path='yolov10n.pt', video_path='football_video.mp4', output_path='data/videos/res.mp4'):
        super().__init__(model_path, video_path)
        self.reference_colors = None  # Couleurs moyennes de référence pour chaque équipe
        self.output_path = output_path
        self.all_color_data = []  # Stocker les couleurs de chaque frame
        self.all_player_boxes = []  # Stocker les boîtes de chaque frame

    def process_frame_for_teams(self, frame, results):
        color_data = []
        player_boxes = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == self.classes_to_detect['player']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_area = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
                    if player_area.size == 0:
                        continue

                    # Convertir la zone en HSV pour filtrer les tons verts
                    hsv_area = cv2.cvtColor(player_area, cv2.COLOR_BGR2HSV)

                    # Définir une plage de couleur pour le vert
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                    mask_green = cv2.inRange(hsv_area, lower_green, upper_green)

                    # Exclure les pixels verts
                    non_green_pixels = player_area[mask_green == 0]
                    if non_green_pixels.size == 0:
                        continue

                    # Calculer la couleur moyenne sans les pixels verts
                    color_data.append(non_green_pixels.mean(axis=0))
                    player_boxes.append(box)

        self.all_color_data.append(color_data)
        self.all_player_boxes.append(player_boxes)

    def adjust_labels(self):
        # Combiner toutes les données de couleur collectées
        combined_color_data = [color for frame_colors in self.all_color_data for color in frame_colors]
        
        # Appliquer PCA pour réduire la dimensionnalité
        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(combined_color_data)

        # Utiliser K-means pour diviser en deux équipes
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(transformed_data)
        
        # Stocker les labels pour chaque frame
        labels_per_frame = []
        start = 0
        for frame_colors in self.all_color_data:
            end = start + len(frame_colors)
            labels_per_frame.append(labels[start:end])
            start = end

        return labels_per_frame

    def track_teams(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialiser le VideoWriter pour sauvegarder le résultat
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Traitement de chaque frame et collecte des données
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            self.process_frame_for_teams(frame, results)
            out.write(frame)  # Écrire chaque frame brute pour la taille et l'initialisation de la vidéo

        # Libérer la capture après avoir collecté toutes les données
        cap.release()

        # Obtenir les labels ajustés pour chaque frame
        labels_per_frame = self.adjust_labels()

        # Relire la vidéo pour appliquer les labels ajustés
        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Obtenir les labels et boîtes de cette frame
            labels = labels_per_frame[frame_index]
            boxes = self.all_player_boxes[frame_index]

            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                team_color = (255, 0, 0) if labels[i] == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
                team_label = f"Team_{labels[i] + 1}"
                cv2.putText(frame, team_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

            # Sauvegarder la frame avec les labels ajustés
            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
