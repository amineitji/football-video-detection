from ultralytics import YOLO

# Charger et configurer le modèle
model = YOLO('yolov10n.pt')  # Modèle YOLO nano (léger)

# Lancer l'entraînement
model.train(data='config/ball_data.yaml', epochs=100, imgsz=640)
