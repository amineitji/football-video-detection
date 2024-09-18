import cv2
import numpy as np
from ultralytics import YOLO
from TeamTracker import TeamTracker

class VideoProcessor:
    def __init__(self, video_path, output_video_path):
        self.model = YOLO("yolov8n.pt")
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.team_tracker = TeamTracker()
        self.CLASS_ID_PERSON = 0

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, imgsz=640, conf=0.4)
            person_boxes, frame_dominant_colors = self._process_frame(results, frame)

            if frame_dominant_colors:
                for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
                    dominant_color = frame_dominant_colors[idx]
                    team_idx = self.team_tracker.track_person(frame_id, dominant_color, person_boxes, frame_dominant_colors)
                    team_label = self.team_tracker.team_labels[int(team_idx)]
                    team_color = self.team_tracker.team_colors[int(team_idx)]

                    cv2.putText(frame, team_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, team_color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)

            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()

    def _process_frame(self, results, frame):
        person_boxes = []
        frame_dominant_colors = []

        # Détecter les lignes du terrain
        touchline_top, touchline_bottom = self.detect_touchlines(frame)
        line_left, line_middle, line_right = self.detect_vertical_lines(frame)

        for idx, result in enumerate(results[0].boxes):
            class_id = int(result.cls)
            if class_id == self.CLASS_ID_PERSON:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                person_boxes.append([x1, y1, x2, y2])

                roi = frame[y1:y2, x1:x2]
                dominant_color = self.team_tracker.get_dominant_color_no_green_or_black(roi)
                frame_dominant_colors.append(dominant_color)

                # Estimer la position orthogonale du joueur
                player_x = (x1 + x2) / 2  # Prendre le centre du rectangle pour la position x
                player_y = (y1 + y2) / 2  # Prendre le centre du rectangle pour la position y

                x_on_field = self.estimate_x_on_field(player_x, line_left, line_middle, line_right)
                y_on_field = self.estimate_y_on_field(player_y, touchline_top, touchline_bottom)

                print(f"Position du joueur sur le terrain (x, y): ({x_on_field}, {y_on_field})")

        return person_boxes, frame_dominant_colors

    def detect_touchlines(self, frame):
        """
        Détecte les lignes de touche sur le terrain (longueurs) en utilisant la transformée de Hough.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=200, maxLineGap=20)

        touchlines = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if abs(y2 - y1) < 10:  # On filtre les lignes presque horizontales
                        touchlines.append((x1, y1, x2, y2))

        # Tri des lignes par leur coordonnée y pour obtenir les lignes supérieure et inférieure
        touchlines = sorted(touchlines, key=lambda l: l[1])
        if len(touchlines) >= 2:
            return touchlines[0], touchlines[-1]  # Lignes de touche en haut et en bas
        return None, None

    def detect_vertical_lines(self, frame):
        """
        Détecte les lignes verticales importantes du terrain (ligne de surface de réparation gauche, centrale et droite).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        vertical_lines = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if abs(x2 - x1) < 10:  # Filtrer les lignes presque verticales
                        vertical_lines.append((x1, y1, x2, y2))

        # Filtrage et tri des lignes détectées
        vertical_lines = sorted(vertical_lines, key=lambda l: l[0])
        if len(vertical_lines) >= 3:
            return vertical_lines[0], vertical_lines[len(vertical_lines) // 2], vertical_lines[-1]  # Gauche, milieu, droite
        return None, None, None

    def estimate_y_on_field(self, player_y, touchline_top, touchline_bottom):
        """
        Estime la position y du joueur en proportion sur le terrain.
        """
        if not touchline_top or not touchline_bottom:
            return None

        total_height = touchline_bottom[1] - touchline_top[1]
        y_relative = (player_y - touchline_top[1]) / total_height
        return y_relative * 68  # 68m est la hauteur du terrain de football

    def estimate_x_on_field(self, player_x, line_left, line_middle, line_right):
        """
        Estime la position x du joueur par rapport aux trois lignes verticales (gauche, milieu, droite).
        """
        if not line_left or not line_middle or not line_right:
            return None

        if player_x <= line_middle[0]:
            total_width_left = line_middle[0] - line_left[0]
            x_relative = (player_x - line_left[0]) / total_width_left
            return x_relative * 52.5  # 52.5m pour la moitié du terrain
        else:
            total_width_right = line_right[0] - line_middle[0]
            x_relative = (player_x - line_middle[0]) / total_width_right
            return 52.5 + (x_relative * 52.5)  # 52.5m + la proportion de l'autre moitié
