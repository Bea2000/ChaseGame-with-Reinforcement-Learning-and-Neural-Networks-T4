import numpy as np
import random
import os

# Funciones auxiliares que pueden resultar útiles para tu implementación
from utils import bfs_search, get_valid_moves

# Path actual de trabajo
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# Hiperparámetros de entrenamiento (jugar con ellos, estudiar que ocurre al cambiarlos)
CAT_MAX_EXPLORATION_RATE = 1
CAT_MIN_EXPLORATION_RATE = 0.0001
CAT_EXPLORATION_DECAY_RATE = 0.001
CAT_LR = 0.9
CAT_DISCOUNT_RATE = 0.1

MOUSE_MAX_EXPLORATION_RATE = 1
MOUSE_MIN_EXPLORATION_RATE = 0.0001
MOUSE_EXPLORATION_DECAY_RATE = 0.001
MOUSE_LR = 1
MOUSE_DISCOUNT_RATE = 0.1

class ReinforcedAgent:

    def __init__(self, position, table_name = None):

        # Posición inicial del agente
        self.pos = position

        # ===== CONSTRUCCIÓN DE LA Q-TABLE ===== #
        # Cargamos el mapa y buscamos las posiciones libres dentro de este
        free_positions = []
        lab_map = np.load(os.path.join(CURRENT_PATH, "game_map.npy"))
        for x in range(lab_map.shape[0]):
            for y in range(lab_map.shape[1]):
                if lab_map[x, y] == 0:
                    free_positions.append((x, y))

        # Diccionario que recibe una tupla del estado de juego de key y retorna el índice de su fila asociada en la Q-Table
        self.states_index = dict()
        index = 0
        for cat_pos in free_positions:
            for mouse_pos in free_positions:
                self.states_index[tuple([cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1]])] = index
                index += 1
            
        # Tasa de exploración del agente
        self.exploration_rate = 1
        
        # En caso de haber una Q-Table preexistente, utilizarla
        if table_name is None:
            self.q_table = np.zeros((index, 5))
        
        # En caso de no entregar una Q-Table, crear una llena de ceros
        else:
            self.q_table = np.load(os.path.join(CURRENT_PATH, "data", table_name))


    # Obtener la acción a ejecutar dado el estado del juego
    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Si entrenamos, considerar si explorar o explotar
        if train:
            if random.random() < self.exploration_rate:
                return random.randint(0, 4)
        
        # Si no, considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        else:
            if random.random() < noise:
                return random.randint(0, 4)
        
        # Calculamos el estado actual del juego
        state = (cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1])


        # ===== COMPLETAR =====
        # Se debe retornar el movimiento que lleve a un mejor estado futuro, basándose en la Q-Table
        # recuerda que el estado actual del juego es state
        row = self.states_index[state]
        move = self.q_table[row]
        move = np.argmax(move)
        # =====================

        return move
    
    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        pass
    
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos, player):
        state = (old_cat_pos[0], old_cat_pos[1], old_mouse_pos[0], old_mouse_pos[1])
        new_state = (new_cat_pos[0], new_cat_pos[1], new_mouse_pos[0], new_mouse_pos[1])
        if player == "cat":
            LR = CAT_LR
            DISCOUNT_RATE = CAT_DISCOUNT_RATE
        else:
            LR = MOUSE_LR
            DISCOUNT_RATE = MOUSE_DISCOUNT_RATE
        # ===== COMPLETAR =====
        # Se debe actualizar el valor asociado al par estado-acción en la Q-Table
        # recuerda que la acción jugada fue action en el estado state
        # y que el estado futuro es new_state
        row = self.states_index[state]
        new_row = self.states_index[new_state]
        reward = self.get_reward(lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos)
        self.q_table[row,action] = (1 - LR) * self.q_table[row,action] + LR * (reward + DISCOUNT_RATE * np.max(self.q_table[new_row,:]))
        # =====================
    
    def update_exploration(self, n_game):
        pass

class RLCat(ReinforcedAgent):

    def __init__(self, position, table_path = None):

        super().__init__(position, table_path)

    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        # ===== COMPLETAR =====
        # Se debe calcular el reward para la acción realizada por el agente
        # Calculamos la distancia entre la posición actual del gato y la posición actual del ratón
        old_distance = np.abs(old_cat_pos[0] - old_mouse_pos[0]) + np.abs(old_cat_pos[1] - old_mouse_pos[1])
        new_distance = np.abs(new_cat_pos[0] - new_mouse_pos[0]) + np.abs(new_cat_pos[1] - new_mouse_pos[1])
        
        if new_distance == 0:
            reward = 100
        # Si el gato se acerca al ratón, se le otorga una recompensa positiva
        elif new_distance < old_distance:
            reward = 10
        # Si el gato se aleja del ratón, se le otorga una recompensa negativa
        elif new_distance > old_distance:
            reward = -10
        # Si el gato mantiene la misma distancia con respecto al ratón, no se le otorga ninguna recompensa
        else:
            reward = 0
        # =====================
        return reward
    
    def update_exploration(self, n_game):
        # ===== COMPLETAR =====
        # Se debe actualizar la tasa de exploración del agente
        # Actualiza la tasa de exploración del agente
        self.exploration_rate = CAT_MIN_EXPLORATION_RATE + (CAT_MAX_EXPLORATION_RATE - CAT_MIN_EXPLORATION_RATE) * np.exp(-CAT_EXPLORATION_DECAY_RATE*n_game)
        # =====================

        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableCat{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableCat{n_game}.npy")
    
class RLMouse(ReinforcedAgent):
    def __init__(self, position, table_path = None):

        super().__init__(position, table_path)

    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        # ===== COMPLETAR =====
        # Se debe calcular el reward para la acción realizada por el agente
        # =====================
        # Calculamos la distancia entre la posición actual del ratón y la posición actual del gato
        old_distance = np.abs(old_mouse_pos[0] - old_cat_pos[0]) + np.abs(old_mouse_pos[1] - old_cat_pos[1])
        new_distance = np.abs(new_mouse_pos[0] - new_cat_pos[0]) + np.abs(new_mouse_pos[1] - new_cat_pos[1])
        
        if new_distance == 0:
            reward = -100
        elif new_distance!=0:
            reward = 10
        # Si el ratón se aleja del gato, se le otorga una recompensa positiva
        elif new_distance > old_distance:
            reward = 5
        # Si el ratón se acerca al gato, se le otorga una recompensa negativa
        elif new_distance < old_distance:
            reward = -5
        else:
            reward = 0

        return reward
    
    def update_exploration(self, n_game):
        # ===== COMPLETAR =====
        # Se debe actualizar la tasa de exploración del agente
        self.exploration_rate = MOUSE_MIN_EXPLORATION_RATE + (MOUSE_MAX_EXPLORATION_RATE - MOUSE_MIN_EXPLORATION_RATE) * np.exp(-MOUSE_EXPLORATION_DECAY_RATE*n_game)
        # =====================

        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableMouse{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableMouse{n_game}.npy")


## Actividad 2.4: Implementación de Q-Learning

### ¿Qué rol cumple la tasa de descuento en Q-Learning ?
# La tasa de descuento es un factor que se utiliza para determinar la importancia de las recompensas futuras.
# Un factor de descuento más alto hará que el agente considere las recompensas futuras más importantes que las recompensas inmediatas.
# Esto permite que el algoritmo de Q-Learning considere el valor acumulado de las recompensas a largo plazo y tome decisiones óptimas a largo plazo en lugar de maximizar solo las recompensas inmediatas.

### ¿Qué tasa de descuento te dio mejores resultados? ¿Por qué crees que fue así?