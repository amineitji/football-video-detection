import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import cv2

class TeamTracker:
    def __init__(self, num_teams=2):
        self.person_team_tracker = defaultdict(lambda: {"color": None, "team": None})
        self.team_labels = {0: "Team_1", 1: "Team_2"}
        self.team_colors = {0: (0, 0, 255), 1: (255, 0, 0)}
        self.num_teams = num_teams

    def get_dominant_color_no_green_or_black(self, image, k=2, threshold=40):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        image[green_mask != 0] = [0, 0, 0]

        image_resized = cv2.resize(image, (50, 50))
        pixel_values = image_resized.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(pixel_values)

        valid_colors = [center for center in kmeans.cluster_centers_ if np.linalg.norm(center) > threshold]
        dominant_color = valid_colors[0] if valid_colors else np.array([128, 128, 128])
        return dominant_color

    def assign_team(self, frame_dominant_colors, frame_id):
        kmeans_teams = KMeans(n_clusters=self.num_teams, random_state=0)
        kmeans_teams.fit(frame_dominant_colors)
        return kmeans_teams.labels_

    def track_person(self, frame_id, dominant_color, person_boxes, frame_dominant_colors):
        best_match = None
        min_distance = float('inf')

        for person_id, info in self.person_team_tracker.items():
            color_dist = np.linalg.norm(dominant_color - info['color'])
            if color_dist < min_distance:
                best_match = person_id
                min_distance = color_dist

        if best_match is not None and min_distance < 50:
            team_idx = self.person_team_tracker[best_match]['team']
        else:
            # Si aucun match, utiliser KMeans mais retourner un index unique pour cette personne
            if frame_id == 0:
                kmeans_teams = KMeans(n_clusters=self.num_teams, random_state=0)
                kmeans_teams.fit(frame_dominant_colors)
                team_idx = int(kmeans_teams.labels_[len(person_boxes) - 1])  # Correction ici : récupère un seul label
            else:
                team_idx = 0 if len(frame_dominant_colors) % 2 == 0 else 1

            self.person_team_tracker[frame_id] = {"color": dominant_color, "team": team_idx}

        return team_idx
