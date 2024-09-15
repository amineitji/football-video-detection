import cv2
from ultralytics import YOLO
from team_tracker import TeamTracker

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
                    team_label = self.team_tracker.team_labels[int(team_idx)]  # Correction : conversion en entier
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

        for idx, result in enumerate(results[0].boxes):
            class_id = int(result.cls)
            if class_id == self.CLASS_ID_PERSON:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                person_boxes.append([x1, y1, x2, y2])

                roi = frame[y1:y2, x1:x2]
                dominant_color = self.team_tracker.get_dominant_color_no_green_or_black(roi)
                frame_dominant_colors.append(dominant_color)

        return person_boxes, frame_dominant_colors
