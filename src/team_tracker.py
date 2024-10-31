import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from detection_football import FootballDetection

class TeamTracker(FootballDetection):
    def __init__(self, model_path='yolov10n.pt', video_path='football_video.mp4', output_final_path='data/videos/res_final.mp4', output_ball_filter_path='data/videos/res_ball_filter.mp4'):
        super().__init__(model_path, video_path)
        self.output_final_path = output_final_path
        self.output_ball_filter_path = output_ball_filter_path
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

                    hsv_area = cv2.cvtColor(player_area, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])
                    mask_green = cv2.inRange(hsv_area, lower_green, upper_green)
                    non_green_pixels = player_area[mask_green == 0]
                    if non_green_pixels.size == 0:
                        continue

                    color_data.append(non_green_pixels.mean(axis=0))
                    player_boxes.append(box)

        self.all_color_data.append(color_data)
        self.all_player_boxes.append(player_boxes)

    def adjust_labels(self):
        combined_color_data = [color for frame_colors in self.all_color_data for color in frame_colors]
        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(combined_color_data)
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(transformed_data)
        
        labels_per_frame = []
        start = 0
        for frame_colors in self.all_color_data:
            end = start + len(frame_colors)
            labels_per_frame.append(labels[start:end])
            start = end

        return labels_per_frame

    def detect_field_lines(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
        green_frame = cv2.bitwise_and(frame, frame, mask=mask_green)

        gray = cv2.cvtColor(green_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        line_frame = frame.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Lignes blanches

        return line_frame
    

    def detect_ball(self, frame, out_ball_filter):
        # Définir la zone d'intérêt (ROI) aux 2/3 inférieurs de l'image
        height, width = frame.shape[:2]
        roi = frame[int(height / 3):, :]  # Zone d'intérêt : les 2/3 inférieurs de l'image

        # Appliquer un filtre de couleur pour isoler le ballon dans la zone d'intérêt
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Plage de couleur pour le ballon (ajustez en fonction de la couleur du ballon)
        lower_ball = np.array([0, 0, 200])  # Pour détecter des tons blancs
        upper_ball = np.array([180, 50, 255])
        mask_ball = cv2.inRange(hsv_roi, lower_ball, upper_ball)
        
        # Détection des contours dans le masque de couleur
        contours, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Créer une image noire pour dessiner le contour du ballon le plus circulaire
        ball_contour_frame = np.zeros_like(mask_ball)
        
        best_circle = None
        max_circularity = 0
        
        # Trouver le contour le plus circulaire dans les 2/3 inférieurs
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue  # Éviter la division par zéro

            # Calcul de la circularité
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # Filtrer les contours pour détecter les cercles parfaits uniquement
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)
            
            # Vérifier le rapport d'aspect pour éviter les cercles "allongés"
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Vérification de la circularité, du rapport d'aspect et de la taille par rapport aux joueurs
            if circularity > max_circularity and circularity > 0.7 and 0.9 < aspect_ratio < 1.1:
                # Ajuster la position du ballon détecté pour la position dans l'image d'origine
                y += int(height / 3)  # Réajustement de la position pour le ROI
                for box in self.all_player_boxes[-1]:  # Vérifier les boîtes des joueurs dans la frame actuelle
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_width, player_height = x2 - x1, y2 - y1
                    if radius < min(player_width, player_height) / 4:  # Le ballon doit être au moins 4 fois plus petit
                        max_circularity = circularity
                        best_circle = (x, y, radius, contour)
        
        # Dessiner et enregistrer le meilleur cercle détecté, si trouvé
        if best_circle:
            x, y, radius, best_contour = best_circle
            cv2.drawContours(ball_contour_frame, [best_contour], -1, 255, thickness=2)
            out_ball_filter.write(ball_contour_frame)
            return (x, y, radius)  # Retourner la position et le rayon du ballon détecté
        
        return None




    def track_teams_with_ball_lines_and_filter(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_final = cv2.VideoWriter(self.output_final_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_ball_filter = cv2.VideoWriter(self.output_ball_filter_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            self.process_frame_for_teams(frame, results)

        cap.release()

        labels_per_frame = self.adjust_labels()

        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            labels = labels_per_frame[frame_index]
            boxes = self.all_player_boxes[frame_index]

            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                team_color = (255, 0, 0) if labels[i] == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
                team_label = f"Team_{labels[i] + 1}"
                cv2.putText(frame, team_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

            ball_position = self.detect_ball(frame, out_ball_filter)
            if ball_position:
                x, y, radius = ball_position
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
                cv2.putText(frame, 'Ball', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            line_frame = self.detect_field_lines(frame)
            frame = cv2.addWeighted(frame, 0.8, line_frame, 0.5, 0)

            out_final.write(frame)
            frame_index += 1

        cap.release()
        out_final.release()
        out_ball_filter.release()
        cv2.destroyAllWindows()