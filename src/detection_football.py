# detection_football.py

from ultralytics import YOLO
import cv2

class FootballDetection:
    def __init__(self, model_path='yolov10n.pt', video_path='football_video.mp4'):
        # Charger le modèle YOLO personnalisé
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.classes_to_detect = {
            'player': 0,  # Classe ID pour les joueurs
            'ball': 1,    # Classe ID pour le ballon
            'line': 2     # Classe ID pour les lignes du terrain
        }

    def detect_objects(self):
        # Charger la vidéo
        cap = cv2.VideoCapture(self.video_path)

        # Boucle pour traiter chaque frame de la vidéo
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Appliquer la détection sur chaque frame
            results = self.model(frame)

            # Parcourir les résultats de détection
            for result in results:
                boxes = result.boxes.cpu().numpy()  # Extraire les boîtes
                for box in boxes:
                    class_id = int(box.cls[0])  # Classe détectée
                    if class_id in self.classes_to_detect.values():
                        # Récupérer les coordonnées et la confiance
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        label = self.get_label(class_id)

                        # Dessiner la boîte de détection
                        color = (0, 255, 0) if label == 'player' else (0, 0, 255) if label == 'ball' else (255, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Afficher la frame avec les détections
            cv2.imshow('Football Detection', frame)

            # Quitter avec la touche 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()

    def get_label(self, class_id):
        # Retourner le label basé sur l'ID de classe
        for label, cid in self.classes_to_detect.items():
            if class_id == cid:
                return label
        return 'unknown'
