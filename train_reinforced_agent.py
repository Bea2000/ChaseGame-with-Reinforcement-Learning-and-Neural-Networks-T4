import numpy as np
from chase_game import ChaseGame

from agents.baseline import BaseCat, BaseMouse
from agents.neural import NNCat, NNMouse
from agents.reinforced import RLCat, RLMouse

# Si deseamos o no visualización en el juego
VISUALIZATION = True

# Número de partidas a jugar para entrenar
NUM_EPISODES = 100000

# Instanciamos el juego
game = ChaseGame(visualization = VISUALIZATION)

# Instanciamos los agentes
# cat = BaseCat(game.cat_pos)
# mouse = BaseMouse(game.mouse_pos)

cat = NNCat(game.cat_pos, "NNCat.h5")         # Reemplazar con el nombre del archivo de la red neuronal (los agentes neuronales no aprenden en este archivo)
mouse = NNMouse(game.mouse_pos, "NNMouse.h5") # Reemplazar con el nombre del archivo de la red neuronal (los agentes neuronales no aprenden en este archivo)

# cat = RLCat(game.cat_pos)
# mouse = RLMouse(game.mouse_pos, "NNMouse.h5")  # Reemplazar con el nombre del archivo de la Q-Table (los agentes Q-Learning no aprenden en este archivo

# Métricas de desempeño
mean_time = 0
total_time = 0
period_steps = 0
max_time = 0
min_time = np.inf

for n_game in range(1, NUM_EPISODES + 1):
    
    # Mientras la partida no ha acabado
    while not game.end:

        # Guardamos las posiciones anteriores de cada agente,
        # son utilizadas para actualizar la Q-Table en Q-Learning :)
        old_cat_pos = game.cat_pos.copy()
        old_mouse_pos = game.mouse_pos.copy()

        # Obtenemos la acción de cada agente
        cat_action = cat.get_action(game.lab_map, game.cat_pos, game.mouse_pos, noise = 0.1, train = True)
        mouse_action = mouse.get_action(game.lab_map, game.cat_pos, game.mouse_pos, noise = 0.05, train = True)
        # Jugamos el movimiento de cada agente
        game.game_step(cat_action, mouse_action)

        # Actualizamos las posiciones de cada agente
        cat.pos = game.cat_pos
        mouse.pos = game.mouse_pos

        # (Exclusivo para Q-Learning), se actualiza la política de comportamiento
        # (el método para los otros agentes es vacío)
        cat.update_policy(game.lab_map, cat_action, old_cat_pos, game.cat_pos, old_mouse_pos, game.mouse_pos, "cat")
        mouse.update_policy(game.lab_map, mouse_action, old_cat_pos, game.cat_pos, old_mouse_pos, game.mouse_pos, "mouse")

    # (Exclusivo para Q-Learning), se actualiza la tasa de exploración
    # Aquí también aprovecharemos de guardar la Q-Table del agente
    cat.update_exploration(n_game)
    mouse.update_exploration(n_game)

    # Cada 100 partidas, reportamos el desempeño
    if n_game % 10 == 0:
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
