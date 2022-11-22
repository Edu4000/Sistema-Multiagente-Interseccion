from model import *

SIZE = mesa.visualization.UserSettableParameter(
        "slider",
        "Tamaño de mapa",
        1,
        1,
        4,
        1,
        description="Elija el tamaño del mapa de la simulación"
)
ROWS = 2
COLS = 2
STREET_SIZE = 1
INTERSECTION_SEPARATION = 20

simulation_params = {
    "num_cars": mesa.visualization.UserSettableParameter(
        "slider",
        "Numero de coches",
        10,
        10,
        200,
        1,
        description="Elija el numero de coches"
    ),
    "separation": mesa.visualization.UserSettableParameter(
        "slider",
        "Separacion de cruces",
        14,
        14,
        20,
        1,
        description="Elija la distancia entre intersecciones"
    ),
    "street_size": mesa.visualization.UserSettableParameter(
        "slider",
        "Numero de lineas por sentido",
        1,
        1,
        3,
        1,
        description="Elija el numero de lineas de transito."
    ),
    "width": ROWS,
    "height": COLS,
    "run": mesa.visualization.UserSettableParameter(
        "slider",
        "Tipo de Simulacion",
        1,
        1,
        2,
        1,
        description="Elija el tipo de simulacion."
    )
}

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
    }
    if (agent.type == 't_f'):
        portrayal["Layer"] = 0
        portrayal["Color"] = agent.status

    elif (agent.type == 'car'):
        portrayal["Layer"] = 1
        portrayal["Color"] = "black"
    
    elif (agent.type == "placeholder"):
        portrayal
        portrayal["Layer"] = 1
        portrayal["Color"] = "gray"

    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, 
                                    (ROWS + 1)*(INTERSECTION_SEPARATION + STREET_SIZE), 
                                    (COLS + 1)*(INTERSECTION_SEPARATION + STREET_SIZE), 
                                     700, 700)

chart = mesa.visualization.ChartModule(
    [
        {"Label": "Average Speed", "Color":"green"},
    ],
    canvas_height=300,
    data_collector_name="data"
)

chart2 = mesa.visualization.ChartModule(
    [
        {"Label": "Perc. Halted Cars", "Color":"red"},
    ],
    canvas_height=300,
    data_collector_name="data"
)

server = mesa.visualization.ModularServer(
    Model, [grid, chart, chart2], "Modelo de Interseccion", 
    simulation_params
)
server.port = 8521  # The default
server.launch()