from utils import bfs_search, get_valid_moves
import numpy as np
import random

class BaseCat():
    def __init__(self, position):

        # Posición inicial del agente
        self.pos = position

        # Hash utilizado para calcular que movimiento seguir tras hacer una búsqueda mediante BFS
        self.diffs = {
            np.array([0, -1]).tobytes(): 0,
            np.array([0, 1]).tobytes(): 1,
            np.array([-1, 0]).tobytes(): 2,
            np.array([1, 0]).tobytes(): 3,
            np.array([0, 0]).tobytes(): 4
        }

    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        if random.random() < noise:
            return random.randint(0, 4)

        # Ruta más corta de la posición actual del gato a la posición del ratón
        route = bfs_search(lab_map, cat_pos, mouse_pos)

        # Consideramos el siguiente paso en la ruta más corta como el movimiento a seguir
        optimal_position = route[1] if len(route) > 1 else route[0]
        diff = np.array(optimal_position.coordinates) - cat_pos
        move = self.diffs.get(diff.tobytes(), 4)

        return move
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos, player):
        return
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_exploration(self, n_game):
        return

class BaseMouse():
    def __init__(self, position):

        # Posición inicial del agente
        self.pos = position

        # Hash utilizado para calcular que efecto tiene una cierta posición
        self.moves = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
            4: np.array([0, 0])
        }

    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        # también, el ratón jugará a moverse aleatoriamente si el gato está lejos
        if random.random() < noise or np.sum(np.abs(mouse_pos - cat_pos)) > lab_map.shape[0]:
            return random.randint(0, 4)

        # Movimientos posibles desde la configuración actual
        valid_moves = np.array(get_valid_moves(lab_map, mouse_pos))
        distances = []

        # Calculamos la distancia al gato desde cada posible nueva configuración
        for move in valid_moves:
            new_pos = mouse_pos + self.moves[move]
            distances.append(np.sum(np.abs(new_pos - cat_pos)))
        
        # Creamos un array con los movimientos que más nos alejan del gato
        best_moves = valid_moves[np.array(distances) == max(distances)]
        weights = [1 for _ in best_moves]

        # Excepciones, trataremos de evitar caminos sin salida, reduciendo su probabilidad
        if (mouse_pos == np.array((7,1))).all() and 0 in best_moves:
            weights[np.where(best_moves == 0)[0][0]] = 0.2
        elif (mouse_pos == np.array((4,6))).all() and 1 in best_moves:
            weights[np.where(best_moves == 1)[0][0]] = 0.2

        # Elegimos el movimiento que más nos aleja del gato
        move = random.choices(best_moves, weights = weights, k = 1)[0]
        
        return move
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos, player):
        return
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_exploration(self, n_game):
        return
