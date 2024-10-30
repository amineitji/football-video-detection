# ball_annotation_tool.py

import cv2
import os
import random
import shutil

class BallAnnotationTool:
    def __init__(self, video_path, output_folder, class_id=0, frame_skip=10, train_ratio=0.8):
        self.video_path = video_path
        self.output_folder = output_folder
        self.class_id = class_id  # ID pour la classe du ballon
        self.frame_skip = frame_skip  # Sauter des frames pour réduire les annotations
        self.train_ratio = train_ratio  # Ratio pour train/val split
        self.bbox = []
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.frame_count = 0
        self.current_frame = None
        self.frame_name = None

        # Créer les dossiers d'organisation
        self.setup_folders()

    def setup_folders(self):
        # Dossiers principaux pour les images et labels
        self.images_train_folder = os.path.join(self.output_folder, "images", "train")
        self.images_val_folder = os.path.join(self.output_folder, "images", "val")
        self.labels_train_folder = os.path.join(self.output_folder, "labels", "train")
        self.labels_val_folder = os.path.join(self.output_folder, "labels", "val")
        
        os.makedirs(self.images_train_folder, exist_ok=True)
        os.makedirs(self.images_val_folder, exist_ok=True)
        os.makedirs(self.labels_train_folder, exist_ok=True)
        os.makedirs(self.labels_val_folder, exist_ok=True)

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                frame_copy = self.current_frame.copy()
                cv2.rectangle(frame_copy, self.start_point, self.end_point, (255, 0, 0), 2)
                cv2.imshow("Ball Annotator", frame_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            cv2.rectangle(self.current_frame, self.start_point, self.end_point, (255, 0, 0), 2)
            self.bbox.append((self.start_point, self.end_point))

    def save_annotations(self, train=True):
        if not self.bbox:
            print(f"Aucune annotation pour {self.frame_name}. Frame ignorée.")
            return

        # Dossier de destination (train ou val) pour l'image et le label
        image_folder = self.images_train_folder if train else self.images_val_folder
        label_folder = self.labels_train_folder if train else self.labels_val_folder

        # Sauvegarder l'image
        frame_path = os.path.join(image_folder, self.frame_name)
        cv2.imwrite(frame_path, self.current_frame)

        # Calcul des coordonnées normalisées au format YOLO
        frame_height, frame_width = self.current_frame.shape[:2]
        annotation_path = os.path.join(label_folder, self.frame_name.replace(".jpg", ".txt"))
        with open(annotation_path, "w") as f:
            for (start, end) in self.bbox:
                x1, y1 = start
                x2, y2 = end
                x_center = (x1 + x2) / 2 / frame_width
                y_center = (y1 + y2) / 2 / frame_height
                width = abs(x2 - x1) / frame_width
                height = abs(y2 - y1) / frame_height
                f.write(f"{self.class_id} {x_center} {y_center} {width} {height}\n")
        print(f"Annotation sauvegardée pour {self.frame_name} dans {'train' if train else 'val'}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sauter les frames pour réduire les annotations nécessaires
            if self.frame_count % self.frame_skip == 0:
                self.bbox = []
                self.current_frame = frame.copy()
                self.frame_name = f"frame_{self.frame_count}.jpg"
                print(f"Annoter le ballon dans la frame : {self.frame_name}")

                cv2.imshow("Ball Annotator", self.current_frame)
                cv2.setMouseCallback("Ball Annotator", self.draw_rectangle)

                # Contrôles utilisateur
                print("Appuyez sur 's' pour sauvegarder, 'r' pour réinitialiser, 'q' pour quitter")

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):  # Sauvegarder l'annotation
                        # Déterminer si la frame va dans train ou val
                        train = random.random() < self.train_ratio
                        self.save_annotations(train=train)
                        break
                    elif key == ord("r"):  # Réinitialiser les annotations
                        self.bbox = []
                        self.current_frame = frame.copy()
                        cv2.imshow("Ball Annotator", self.current_frame)
                    elif key == ord("q"):  # Quitter le programme
                        print("Annotation interrompue.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Spécifiez le chemin de la vidéo et le dossier de sortie pour les annotations
    video_path = "data/videos/test_1.mp4"
    output_folder = "data/ball_dataset"
    class_id = 0  # ID de la classe pour le ballon
    frame_skip = 10  # Annoter une frame tous les 10 frames (ajustable)

    annotator = BallAnnotationTool(video_path, output_folder, class_id, frame_skip)
    annotator.run()
