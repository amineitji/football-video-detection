from VideoProcessor import VideoProcessor
from field_line_detector import FieldLineDetector

if __name__ == "__main__":
    # Détection des joueurs avec les équipes
    video_path = 'data/videos/test.mp4'
    output_video_path = 'data/videos/output_with_teams.mp4'

    video_processor = VideoProcessor(video_path, output_video_path)
    video_processor.process_video()

    print(f"[INFO] Vidéo complète enregistrée dans '{output_video_path}' avec les labels d'équipe.")