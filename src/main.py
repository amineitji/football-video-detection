from video_processor import VideoProcessor
from field_line_detector import FieldLineDetector

if __name__ == "__main__":
    # Détection des joueurs avec les équipes
    video_path = 'data/videos/test.mp4'
    output_video_path = 'data/videos/output_with_teams.mp4'

    video_processor = VideoProcessor(video_path, output_video_path)
    video_processor.process_video()

    print(f"[INFO] Vidéo complète enregistrée dans '{output_video_path}' avec les labels d'équipe.")

    # Détection des lignes du terrain
    output_line_video_path = 'data/videos/output_with_lines.mp4'
    line_detector = FieldLineDetector(video_path, output_line_video_path)
    line_detector.process_video()

    print(f"[INFO] Vidéo complète enregistrée dans '{output_line_video_path}' avec les lignes du terrain détectées.")
