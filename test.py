import numpy as np
from chase_game import ChaseGame

from agents.baseline import BaseCat, BaseMouse
from agents.neural import NNCat, NNMouse
from agents.reinforced import RLCat, RLMouse

# Si deseamos o no visualización en el juego
VISUALIZATION = False

# Número de partidas a jugar
NUM_EPISODES = 100000

# Instanciamos el juego
game = ChaseGame(visualization = VISUALIZATION)

# Instanciamos los agentes
cat = BaseCat(game.cat_pos)
# mouse = BaseMouse(game.mouse_pos)

# cat = NNCat(game.cat_pos, "NNCat.h5")         # Reemplazar con el nombre del archivo de la red neuronal
# mouse = NNMouse(game.mouse_pos, "NNMouse.h5") # Reemplazar con el nombre del archivo de la red neuronal

# cat = RLCat(game.cat_pos, "QTableCat100000.npy")        # Reemplazar con el nombre del archivo de la Q-Table
mouse = RLMouse(game.mouse_pos, "QTableMouse100000.npy")  # Reemplazar con el nombre del archivo de la Q-Table

# Métricas de desempeño
mean_time = 0
total_time = 0
period_steps = 0
max_time = 0
min_time = np.inf

for n_game in range(1, NUM_EPISODES + 1):
    
    # Mientras la partida no ha acabado
    while not game.end:

        # Obtenemos la acción de cada agente
        cat_action = cat.get_action(game.lab_map, game.cat_pos, game.mouse_pos, noise = 0.01)
        
        mouse_action = mouse.get_action(game.lab_map, game.cat_pos, game.mouse_pos, noise = 0.01)
        # Jugamos el movimiento de cada agente
        game.game_step(cat_action, mouse_action)

        # Actualizamos las posiciones de cada agente
        cat.pos = game.cat_pos
        mouse.pos = game.mouse_pos

    # Cada 100 partidas, reportamos el desempeño
    if n_game % 100 == 0:
        print('Game', n_game, '| Mean Steps:', period_steps//100, '| MAX:', max_time, "| MIN:", min_time, "| Total Steps:", period_steps)
        period_steps = 0
        max_time = 0
        min_time = np.inf

    # Métricas específicas de cada periodo
    period_steps += game.t
    if game.t > max_time:
        max_time = game.t
    if game.t < min_time:
        min_time = game.t

    # Métricas generales del juego
    mean_time += 1 / n_game * (game.t - mean_time)
    total_time += game.t

    # Si la partida termina, iniciamos una nueva
    game.reset(n_game)
