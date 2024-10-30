from team_tracker import TeamTracker

def main():
    # Initialiser le suivi des équipes avec le modèle YOLO personnalisé et la vidéo de football
    team_tracker = TeamTracker(model_path='yolov10n.pt', video_path='data/videos/test_1.mp4')
    team_tracker.track_teams()

if __name__ == '__main__':
    main()
