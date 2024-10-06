import numpy as np

class Cell:
    """ Clase que define a un nodo, usado en BFS """
    
    def __init__(self, coordinates, from_cell):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.parent = from_cell
    
    def __repr__(self):
        return str(self.coordinates)
    
    def __str__(self):
        return str(self.coordinates)

def get_neighboring_cells(current_cell, grid_map):
    """ Retorna los nodos aledaños vacíos en las cuatro direcciones a un cierto punto """

    neighbors = []
    map_rows = grid_map.shape[0]
    map_cols = grid_map.shape[1]

    # Para cada dirección (izquierda, derecha, abajo, arriba) creamos el vecino si su casilla esta vacía
    if current_cell.x - 1 >= 0 and grid_map[current_cell.x - 1, current_cell.y] == 0:
        neighbors.append(Cell([current_cell.x - 1, current_cell.y], current_cell))

    if current_cell.x + 1 < map_rows and grid_map[current_cell.x + 1, current_cell.y] == 0:
        neighbors.append(Cell([current_cell.x + 1, current_cell.y], current_cell))

    if current_cell.y - 1 >= 0 and grid_map[current_cell.x, current_cell.y - 1] == 0:
        neighbors.append(Cell([current_cell.x, current_cell.y - 1], current_cell))
        
    if current_cell.y + 1 < map_cols and grid_map[current_cell.x, current_cell.y + 1] == 0:
        neighbors.append(Cell([current_cell.x, current_cell.y + 1], current_cell))

    return neighbors

def bfs_search(grid_map, cat_pos, mouse_pos):
    """ Búsqueda de profundidad, utilizada para la función go_to del robot (para moverse en un mapa conocido) """

    init_cell = Cell(cat_pos, None)
    visited = {}
    visited_list = []
    stack = [init_cell]

    # Mientras tengamos nodos sin expandir
    while len(stack) > 0:
        # Expandimos un nodo
        current_cell = stack.pop(0)
        
        # Lo marcamos como visitado
        visited[str(current_cell)] = True
        visited_list.append(current_cell.coordinates)
        
        # Si el nodo es objetivo, hacemos la vuelta atrás de cómo llegamos hasta él
        if current_cell.x == mouse_pos[0] and current_cell.y == mouse_pos[1]:
            # Traceback contiene las casillas navegadas hasta llegar al objetivo
            traceback = [current_cell]
            while current_cell.parent is not None:
                traceback.insert(0, current_cell.parent)
                current_cell = current_cell.parent
            return traceback

        # En caso contrario, añadimos al stack todos los vecinos del nodo actualmente expandido
        for neighbor in get_neighboring_cells(current_cell, grid_map):
            if not visited.get(str(neighbor), False):
                stack.append(neighbor)

    # Peor de los casos, retornamos None
    return None

def get_valid_moves(lab_map, agent_pos):
    """ Movimientos posibles en el mapa para un agente en posición agent_pos """

    moves_dict = {
        0: np.array([0, -1]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([1, 0]),
        4: np.array([0, 0])
    }

    moves = []

    # Probamos cada movimiento y vemos si el punto a donde nos lleva está obstruido
    for move in moves_dict.keys():

        new_agent_pos = agent_pos + moves_dict[move]

        if np.min(new_agent_pos) < 0 or np.max(new_agent_pos) >= lab_map.shape[0]:
            continue
        elif lab_map[new_agent_pos[0], new_agent_pos[1]] == 1:
            continue
        else:
            moves.append(move)

    return moves