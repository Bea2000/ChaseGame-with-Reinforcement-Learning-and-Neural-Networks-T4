import numpy as np
import random
import tensorflow as tf
import os

# Funciones auxiliares que pueden resultar útiles para tu implementación
from utils import bfs_search, get_valid_moves

# Path actual de trabajo
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

class NNCat():
    def __init__(self, position, model_name = 'NNCat.h5'):
        
        # Posición inicial del agente
        self.pos = position

        # Cargamos la red del agente (que modela su político)
        # En caso de usar SciKit Learn, se debe cargar el modelo acordemente
        self.model = tf.keras.models.load_model(os.path.join(CURRENT_PATH, "data", model_name))

    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        if random.random() < noise:
            return random.randint(0, 4)

        # Calculamos el estado actual del juego
        state = np.array([cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1]])

        # NOTA: Normalmente la red espera un batch de datos a la vez y no un único vector
        # por lo que usamos el método np.expand_dims(state, axis = 0) para añadir una dimensión
        # y tener un vector de tamaño (1, 4) en lugar de (4)
        state = np.expand_dims(state, axis=0)

        # ===== COMPLETAR =====
        # Se debe retornar el movimiento que lleve a un mejor estado futuro,
        # tomando el índice de mayor probabilidad de la salida de la red
        # Podemos obtener la predicción de un modelo sobre un vector x mediante prediction = model(x)
        move = np.argmax(self.model(state))
        # =====================

        return move
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos, player):
        pass
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_exploration(self, n_game):
        pass
    
class NNMouse():
    def __init__(self, position, model_name = 'NNMouse.h5'):
        
        # Posición inicial del agente
        self.pos = position

        # Cargamos la red del agente (que modela su político)
        # En caso de usar SciKit Learn, se debe cargar el modelo acordemente
        self.model = tf.keras.models.load_model(os.path.join(CURRENT_PATH, "data", model_name))

    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        if random.random() < noise:
            return random.randint(0, 4)

        # Calculamos el estado actual del juego
        state = np.array([cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1]])

        # NOTA: Normalmente la red espera un batch de datos a la vez y no un único vector
        # por lo que usamos el método np.expand_dims(state, axis = 0) para añadir una dimensión
        # y tener un vector de tamaño (1, 4) en lugar de (4)
        state = np.expand_dims(state, axis=0)

        # ===== COMPLETAR =====
        # Se debe retornar el movimiento que lleve a un mejor estado futuro,
        # tomando el índice de mayor probabilidad de la salida de la red
        # Podemos obtener la predicción de un modelo sobre un vector x mediante prediction = model(x)
        move = np.argmax(self.model(state))
        # =====================

        return move
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos, player):
        pass
    
    # Método vacío, no se utiliza en este tipo de agente
    def update_exploration(self, n_game):
        pass