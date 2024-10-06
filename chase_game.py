import pygame
import numpy as np
import time
import os
import random

# Tiempo entre cada movimiento del juego
TIME_DELAY = 0.1

# Máximo de pasos por juegos
MAX_STEPS = 500

# Dimensiones de la pantalla creada
WIDTH = 500
HEIGHT = 500

# Dimensiones del tablero de juego dentro de la ventana
MAP_WIDTH, MAP_HEIGHT = WIDTH // 1.1, HEIGHT // 1.1
BORDER_WIDTH = int(MAP_WIDTH // 100)

# Colores utilizados en la simulación
WHITE = (255, 255, 255)
BLACK = (22, 22, 22)
GREY = (120, 120, 120)
LIGHT_GREY = (240, 240, 240)
BLUE_GREY = (120, 120, 140)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Detalles de la pantalla, utilizados para dibujar el mapa de juego
WIN_INFO = ((WIDTH - MAP_WIDTH) // 2, (HEIGHT - MAP_HEIGHT) // 2, MAP_WIDTH, MAP_HEIGHT)

# Path desde donde se accede al juego, utilizado para cargar el mapa
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

class ChaseGame():
    def __init__(self, visualization = True):
        # Cargamos el mapa de juego
        self.lab_map = np.load(os.path.join(CURRENT_PATH, "game_map.npy"))

        # Calculamos las posiciones libres de obstáculos
        free_positions = []
        for x in range(self.lab_map.shape[0]):
            for y in range(self.lab_map.shape[1]):
                if self.lab_map[x, y] == 0:
                    free_positions.append((x, y))

        # Ubicamos al gato en una posición aleatoria del mapa
        self.cat_pos = np.array(random.choice(free_positions))
        self.mouse_pos = self.cat_pos

        # Colocamos al ratón en una posición aleatoria del mapa, al menos a 6 pasos del gato
        while np.sum(np.abs(self.cat_pos - self.mouse_pos)) < 6:
            self.mouse_pos = np.array(random.choice(free_positions))

        # Duración del juego actual
        self.t = 0

        # Indica si tener visualización (una pantalla) o no
        self.visualization = visualization

        # Indica si la partida actual ha finalizado o no
        self.end = False
        
        if self.visualization:
            pygame.init()
            self.win = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Cat and Mouse Game")
        
        # Movimientos posibles y sus efectos en la posición del agente
        self.moves = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
            4: np.array([0, 0])
        }

    def draw_grid(self):
        # Detalles sobre el espacio a dibujar
        x_zero, y_zero, map_width, map_height = WIN_INFO
        n_rows = self.lab_map.shape[0]
        n_cols = self.lab_map.shape[1]

        # Dibujamos las líneas de la grilla
        for i in range(n_rows + 1):
            pygame.draw.line(self.win, GREY, (x_zero + i * map_width // n_rows, y_zero), (x_zero + i * map_width // n_rows, y_zero + map_height), 1)
        for j in range(n_cols + 1):
            pygame.draw.line(self.win, GREY, (x_zero, y_zero + j * map_height // n_cols), (x_zero + map_width, y_zero + j * map_height // n_cols), 1)

        # Dibujamos los bordes del mapa
        pygame.draw.line(self.win, BLACK, (x_zero, y_zero), (x_zero + map_width, y_zero), BORDER_WIDTH)
        pygame.draw.line(self.win, BLACK, (x_zero, y_zero), (x_zero, y_zero + map_height), BORDER_WIDTH)
        pygame.draw.line(self.win, BLACK, (x_zero + map_width, y_zero + map_height), (x_zero + map_width, y_zero), BORDER_WIDTH)
        pygame.draw.line(self.win, BLACK, (x_zero + map_width, y_zero + map_height), (x_zero, y_zero + map_height), BORDER_WIDTH)

    def draw_map(self):
        # Llenamos la pantalla de un color gris azulado
        self.win.fill(BLUE_GREY)

        # Detalles sobre el espacio a dibujar
        x_zero, y_zero, map_width, map_height = WIN_INFO
        n_rows = self.lab_map.shape[0]
        n_cols = self.lab_map.shape[1]

        # Dibujamos el fondo del mapa
        pygame.draw.rect(self.win, LIGHT_GREY, (x_zero, y_zero, map_width, map_height))

        # Dibujamos las paredes
        for i in range(n_rows):
            for j in range(n_cols):
                if self.lab_map[i, j] == 1:
                    pygame.draw.rect(self.win, BLACK, (x_zero + i * map_width // n_rows, y_zero + j * map_height // n_cols, map_width // n_rows, map_height // n_cols))

        # En caso de terminar la partida, dibujar al ratón y el gato encima del otro con color azul
        if self.end:
            pygame.draw.rect(self.win, BLUE, (x_zero + self.mouse_pos[0] * map_width // n_rows, y_zero + self.mouse_pos[1] * map_height // n_cols, map_width // n_rows, map_height // n_cols))
        
        # Dibujamos al gato y al ratón
        else:
            pygame.draw.rect(self.win, RED, (x_zero + self.cat_pos[0] * map_width // n_rows, y_zero + self.cat_pos[1] * map_height // n_cols, map_width // n_rows, map_height // n_cols))
            pygame.draw.rect(self.win, GREEN, (x_zero + self.mouse_pos[0] * map_width // n_rows, y_zero + self.mouse_pos[1] * map_height // n_cols, map_width // n_rows, map_height // n_cols))

        # Dibujamos la grilla sobre el mapa
        self.draw_grid()

    def game_step(self, cat_move, mouse_move):
        
        # Si la partida no ha finalizado
        if not self.end:

            # Avanzamos un paso la duración
            self.t += 1

            # Posiciones anteriores, utilizadas para revisar si pasan uno sobre otro
            old_cat_pos = self.cat_pos.copy()
            old_mouse_pos = self.mouse_pos.copy()

            # Si el movimiento del gato es válido, jugarlo
            if self.valid_move("cat", cat_move):
                self.cat_pos += self.moves.get(int(cat_move), np.array([0, 0]))
            # else:
                # print("El gato ha jugado un movimiento inválido", cat_move)
            
            # Si el movimiento del ratón es válido, jugarlo
            if self.valid_move("mouse", mouse_move):
                self.mouse_pos += self.moves.get(int(mouse_move), np.array([0, 0]))
            # else:
                # print("El ratón ha jugado un movimiento inválido:", mouse_move)

            # Si se encuentran en la misma posición, terminar el juego
            if (self.cat_pos == self.mouse_pos).all():
                self.end = True

            # Si pasaron uno sobre otro (chocan), terminar el juego
            if (self.cat_pos == old_mouse_pos).all() and (self.mouse_pos == old_cat_pos).all():
                self.cat_pos = self.mouse_pos
                self.end = True

            # Automáticamente terminar el juego a los MAX_STEPS pasos
            if self.t > MAX_STEPS - 1:
                self.end = True

            # Dibujar el nuevo estado del juego
            if self.visualization:
                self.draw_map()
                pygame.display.flip()

                # Delay hasta el siguiente paso
                time.sleep(TIME_DELAY)

    def valid_move(self, agent, move):

        # Verifica si un movimiento del gato es válido
        if agent == "cat":
            new_cat_pos = self.cat_pos + self.moves[move]
            # Si el movimiento lo lleva a una posición dentro del mapa
            if np.min(new_cat_pos) < 0 or np.max(new_cat_pos) >= self.lab_map.shape[0]:
                return False
            # Si el movimiento lo lleva a una pared
            elif self.lab_map[new_cat_pos[0], new_cat_pos[1]] == 1:
                return False
            else:
                return True
            
        # Verifica si un movimiento del ratón es válido
        elif agent == "mouse":
            new_mouse_pos = self.mouse_pos + self.moves[move]
            # Si el movimiento lo lleva a una posición dentro del mapa
            if np.min(new_mouse_pos) < 0 or np.max(new_mouse_pos) >= self.lab_map.shape[0]:
                return False
            # Si el movimiento lo lleva a una pared
            elif self.lab_map[new_mouse_pos[0], new_mouse_pos[1]] == 1:
                return False
            else:
                return True
    
    def reset(self, n_game):
        if n_game % 10 == 0:
            self.visualization = True
        else:
            self.visualization = False
        if self.visualization:
            print(f"El ratón sobrevivió durante {self.t} turnos")
        self.__init__(self.visualization)
