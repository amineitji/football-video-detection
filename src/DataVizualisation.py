import matplotlib.pyplot as plt
from mplsoccer import Pitch
import matplotlib.animation as animation
import os

class DataVisualization:
    def __init__(self):
        # Initialisation de la classe
        pass
    
    def plot_players(self, data):
        """
        Fonction pour afficher la position des joueurs sur un terrain de football en utilisant les coordonnées {x, y, color}.
        
        Paramètres :
        data (list of dict): Liste de dictionnaires avec les coordonnées x, y et la couleur du joueur.
                            Exemple: [{'x': 10, 'y': 20, 'color': 'blue'}, {'x': 30, 'y': 40, 'color': 'red'}, ...]
        """
        # Créer le terrain de football avec mplsoccer
        pitch = Pitch(pitch_type='opta', axis=True, label=True)
        fig, ax = pitch.draw()
        
        # Dessiner les joueurs sur le terrain
        for player in data:
            x = player['x']
            y = player['y']
            color = player['color']
            ax.scatter(x, y, c=color, label=f'Joueur ({x}, {y})', edgecolors='black', s=100)  # Taille réduite
        
        # Ajouter des étiquettes et un titre
        ax.set_title('Position des joueurs sur le terrain')
        
        # Afficher le graphique
        plt.show()

    def animate_players(self, frames_data, output_file='data/videos/animation.gif'):
        """
        Fonction pour générer une animation qui montre le déplacement des joueurs au cours du temps.
        
        Paramètres :
        frames_data (list of list of dict): Liste de frames, où chaque frame est une liste de dictionnaires représentant les joueurs
                                            avec 'x', 'y', 'color' et 'id' comme attributs.
        output_file (str): Chemin de sortie pour l'animation en .gif.
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Créer le terrain de football avec mplsoccer
        pitch = Pitch(pitch_type='opta', axis=True, label=True)
        fig, ax = pitch.draw()

        # Initialisation des points pour les joueurs
        scatters = []
        for frame in frames_data[0]:
            scatter = ax.scatter(frame['x'], frame['y'], c=frame['color'], edgecolors='black', s=100)  # Taille réduite
            scatters.append(scatter)

        def update(frame_data):
            """Fonction appelée à chaque étape de l'animation pour mettre à jour les positions."""
            for i, player in enumerate(frame_data):
                scatters[i].set_offsets([player['x'], player['y']])
                scatters[i].set_color(player['color'])
            return scatters

        # Créer l'animation
        anim = animation.FuncAnimation(
            fig, update, frames=frames_data, interval=100, blit=True, repeat=False  # Intervalle réduit à 100ms pour plus de fluidité
        )

        # Enregistrer l'animation en GIF
        anim.save(output_file, writer='pillow')

        print(f"Animation enregistrée dans {output_file}")

# Génération de frames avec moins de différence pour plus de fluidité
frames_data = [
    [{'x': 30, 'y': 50, 'color': 'blue', 'id': 1}, {'x': 70, 'y': 40, 'color': 'red', 'id': 2}, {'x': 20, 'y': 70, 'color': 'green', 'id': 3}],
    [{'x': 31, 'y': 51, 'color': 'blue', 'id': 1}, {'x': 69, 'y': 41, 'color': 'red', 'id': 2}, {'x': 21, 'y': 69, 'color': 'green', 'id': 3}],
    [{'x': 32, 'y': 52, 'color': 'blue', 'id': 1}, {'x': 68, 'y': 42, 'color': 'red', 'id': 2}, {'x': 22, 'y': 68, 'color': 'green', 'id': 3}],
    [{'x': 33, 'y': 53, 'color': 'blue', 'id': 1}, {'x': 67, 'y': 43, 'color': 'red', 'id': 2}, {'x': 23, 'y': 67, 'color': 'green', 'id': 3}],
    [{'x': 34, 'y': 54, 'color': 'blue', 'id': 1}, {'x': 66, 'y': 44, 'color': 'red', 'id': 2}, {'x': 24, 'y': 66, 'color': 'green', 'id': 3}],
    [{'x': 35, 'y': 55, 'color': 'blue', 'id': 1}, {'x': 65, 'y': 45, 'color': 'red', 'id': 2}, {'x': 25, 'y': 65, 'color': 'green', 'id': 3}],
    [{'x': 36, 'y': 56, 'color': 'blue', 'id': 1}, {'x': 64, 'y': 46, 'color': 'red', 'id': 2}, {'x': 26, 'y': 64, 'color': 'green', 'id': 3}],
    [{'x': 37, 'y': 57, 'color': 'blue', 'id': 1}, {'x': 63, 'y': 47, 'color': 'red', 'id': 2}, {'x': 27, 'y': 63, 'color': 'green', 'id': 3}],
    [{'x': 38, 'y': 58, 'color': 'blue', 'id': 1}, {'x': 62, 'y': 48, 'color': 'red', 'id': 2}, {'x': 28, 'y': 62, 'color': 'green', 'id': 3}],
    [{'x': 39, 'y': 59, 'color': 'blue', 'id': 1}, {'x': 61, 'y': 49, 'color': 'red', 'id': 2}, {'x': 29, 'y': 61, 'color': 'green', 'id': 3}],
    [{'x': 40, 'y': 60, 'color': 'blue', 'id': 1}, {'x': 60, 'y': 50, 'color': 'red', 'id': 2}, {'x': 30, 'y': 60, 'color': 'green', 'id': 3}],
]

# Création de l'objet DataVisualization et génération de l'animation
viz = DataVisualization()
viz.animate_players(frames_data)
