# Reporte 2 - Semaforos Inteligentes

En este reporte se busca estudiar un ambiente de intersecciones con semáforos inteligentes a través del cual transitan un número arbitrario de coches a distintas velocidades. Se presume que de este sistema pueda surgir un comportamiento óptimo para que el tráfico de los coches se reduzca los más posible para todos los agentes en comparación al modelo clásico de control de semaforos.

## Modelo

Para esta simulación en `mesa`, el ambiente contara con dos tipos de agentes diferentes: `CarAgent` y `TrafficLightAgent`. Estos interactuaran en un espacio de tipo `grid` buscando que todos tengan el mejor resultado posible aplicando la siguiente heurística:
- Los semáforos darán prioridad a los coches que llegarán a ellos en menos tiempo.
- Los coches bajaran su velocidad en caso de tener un coche enfrente suyo o un semaforo rojo.


```python
import mesa
import numpy as np
from random import randint, choice
```

### `CarAgent`

Este modelo busca simular las características y comportamientos que tiene un carro en las vías de tránsito, como lo son:
- Velocidad Maxima - `max_speed`
- Velocidad - `speed`
- Movimiento - `move()`


```python
class CarAgent (mesa.Agent):
    def __init__(self, unique_id: int, model: Model, dir) -> None:
        super().__init__(unique_id, model)
        self.type = 'car'
        self.max_speed = 3
        self.speed = 3
        # 0 : derecha | 1 : izquierda | 2 : abajo | 3 : arriba
        self.vect = [(1,0),(-1,0),(0,-1),(0,1)]
        self.dir = np.array(self.vect[dir])

    def move(self):
        self.speed = self.max_speed
        grid_size = self.model.grid.width
        front_view = [
            tuple(np.mod(np.array(self.pos) + self.dir * x, grid_size))
            for x in range(1, int(self.max_speed) + 1)
        ]

        for i in range(len(front_view)):
            front = (int(front_view[i][0]), int(front_view[i][1]))
            front_agent = self.model.grid.get_cell_list_contents(front)
            if (len(front_agent) == 1):
                if (front_agent[0].type == "t_f" and front_agent[0].status != 'red'):
                    self.speed = self.max_speed
                elif (i > 1):
                    self.speed = 1
                else:
                    self.speed = 0.01
                    return

        next_pos = (tuple(np.mod(np.array(self.pos) + self.dir * int(self.speed), grid_size)))
                    
        self.model.grid.move_agent(self, next_pos)
    
    def step(self):
        self.move()
```

### `TrafficLightAgent`
Este modelo crea semaforos con sensores que detectan la velocidad a la que los coches se acercan. Esto les permite a los semaforos de una intersección determinar cual de ellos tendrá la prioridad de tránsito y los otros esperaran.

- Dirección de Prioridad - `rightToTransit`
- Cola de Prioridad - `waitingCars`
- Alcance de Visión - `visionRange`


```python
class TrafficLightAgent (mesa.Agent):
    def __init__(self, unique_id: int, model: Model, group: int, orientation: int, vision) -> None:
        super().__init__(unique_id, model)
        self.dir = 'up'
        self.type = 't_f'
        self.status = 'red'
        self.local_group = group
        self.orientation = orientation # 0 : horizontal | 1 : vertical
        self.vision = vision

    def setGreenLight(self):
        self.status = 'green'
        
    def setYellowLight(self):
        self.status = 'orange'
        
    def setRedLight(self):
        self.status = 'red'

    def update_queue(self):
        dist = 1
        for cell in self.vision:
            agent = self.model.grid.get_cell_list_contents(cell)
            if (len(agent) == 1):
                agent = agent[0]
                eta = dist / agent.speed
                self.model.priority_queue[self.local_group].append((eta, self.orientation))
            else:
                pass
            dist += 1

    def step(self):
        if (len(self.model.priority_queue[self.local_group]) == 0):
            self.setYellowLight()
        elif (self.model.priority_queue[self.local_group][0][1] == self.orientation):
            self.setGreenLight()
        else:
            self.setRedLight()
```

### `Model`

El modelo cuenta con el ambiente en el que los agentes se encuentran y para el sistema de semáforos inteligentes el ambiente es __parcialmente visible__ por parte de los agentes. Tanto los agentes carros como los semaforos, tienen la capacidad de ver algunas celdas enfrente suyas para poder actuar como __agentes racionales__, tomando la mejor acción posible en el momento y espacio donde estan.

En cada paso del modelo se siguen los siguientes pasos como estratégia de solución:
- Los semaforos ven 7 celdas frente suyo buscando coches aproximandoce.
- Los semáforos calculan el tiempo estimado en el que cada coche en su visión podría llegar a ellos.
- Se ordenan los tiempos de llegado estimado y los semáforos escogen cual vía es la que tendrá el paso.
- Los coches avanzan a velocidad máxima en caso de poder hacerlo.
    - Si hay algun otro coche o un semáforo rojo, frenaran o se detendrán dependiendo de su distancia.

Esta estatégia busca que los coches disminuyan su velocidad en la menor medida posible a la vez que se garantiza que no suceda ningun choque entre los agentes coches. Por otra parte, tambien permite la __negociación entre agentes__ de semáforo para poder determinar cual vía tendría la prioridad de flujo.


```python

class Model (mesa.Model):

    def __init__(self, num_cars, separation, street_size, width, height, run) -> None:
        self.num_cars = num_cars
        self.separation = separation
        self.street_size = street_size
        self.run = run
        self.timer = [False, 0]
        self.vision_range = 7
        self.size = (width + 1)*(separation+street_size)
        self.priority_queue = []
        self.car_spawn = [[],[],[],[]]

        # Creating grid
        self.grid = mesa.space.MultiGrid(
            width=self.size, 
            height=self.size,
            torus=False
        )

        # Creating schedulers
        self.schedule_tf = mesa.time.BaseScheduler(self)
        self.schedule_car = mesa.time.RandomActivation(self)

        # Creating traffic lights
        for i in range(width * height):
            self.priority_queue.append([])
            intersection = ((int(i/width) + 1) * (separation+street_size) , 
                            (i%width + 1) * (separation+street_size))
            for j in [1, -1]:
                center_disp = j + j * self.street_size
                for via in range (1, self.street_size+1):
                    pos = (intersection[0] + via*j, intersection[1] - center_disp)
                    vision = [(pos[0], pos[1] - j * y) for y in range(1, self.vision_range+1)]

                    if (j == 1):
                        self.car_spawn[3].append(intersection[0] + via*j)
                    else:
                        self.car_spawn[2].append(intersection[0] + via*j)

                    # for cell in vision:
                    #     self.grid.place_agent(TrafficLineAgent(f"{cell}", self), cell)

                    t_f = TrafficLightAgent(f"t_f_{i}_via_{pos}", self, i, 0, vision)
                    self.schedule_tf.add(t_f)
                    self.grid.place_agent(t_f, pos)

                    pos = (intersection[0] - center_disp, intersection[1] - via*j)
                    vision = [(pos[0] - j * y, pos[1]) for y in range(1, self.vision_range+1)]

                    if (j == 1):
                        self.car_spawn[0].append(intersection[1] - via*j)
                    else:
                        self.car_spawn[1].append(intersection[1] - via*j)

                    # for cell in vision:
                    #     self.grid.place_agent(TrafficLineAgent(f"{cell}", self), cell)

                    t_f = TrafficLightAgent(f"t_f_{i}_via_{pos}", self, i, 1, vision)
                    self.schedule_tf.add(t_f)
                    self.grid.place_agent(t_f, pos)

        # Creating cars
        for i in range(self.num_cars):
            direction = randint(0,3)
            car = CarAgent(f"car_{i}", self, direction)
            self.schedule_car.add(car)

            if (direction == 0):
                self.grid.place_agent(car, (randint(0, self.size - 1), choice(self.car_spawn[direction])))
            elif (direction == 1):
                self.grid.place_agent(car, (randint(0, self.size - 1), choice(self.car_spawn[direction])))
            elif (direction == 2):
                self.grid.place_agent(car, (choice(self.car_spawn[direction]), randint(0, self.size - 1)))
            else:
                self.grid.place_agent(car, (choice(self.car_spawn[direction]), randint(0, self.size - 1)))

         # self.grid.place_agent(car, (choice(self.car_spawn[direction]), randint(0, self.size - 1)))

        # Crating Data Collector
        self.data = mesa.DataCollector(
            {
                "Average Speed": Model.average_speed,
                "Perc. Halted Cars": Model.halt_vehicles,
                "Maximum Speed": Model.max_speed
            }
        )

    def sort_queue(self):
        for local_priority_queue in self.priority_queue:
            sorting = True
            while sorting:
                sorting = False
                for i in range(1,len(local_priority_queue)):
                    if (local_priority_queue[i-1][0] < local_priority_queue[i][0]):
                        aux = local_priority_queue[i-1]
                        local_priority_queue[i-1] = local_priority_queue[i]
                        local_priority_queue[i] = aux
                        sorting = True

    def step(self):
        if (self.run == 1):
            self.priority_queue = [[] for i in range(len(self.priority_queue))]
            for tf in self.schedule_tf.agents:
                tf.update_queue()
            self.sort_queue()
            self.schedule_tf.step()
        else:
            if (self.timer[1] > 5):
                self.timer = [not self.timer[0], 0]

            if (self.timer[0]):
                for tf in self.schedule_tf.agents:
                    if (tf.orientation == 0):
                        tf.setGreenLight()
                    else:
                        tf.setRedLight()
            else:
                for tf in self.schedule_tf.agents:
                    if (tf.orientation == 1):
                        tf.setGreenLight()
                    else:
                        tf.setRedLight()
            self.timer[1] += 1

        self.schedule_car.step()
        self.data.collect(self)

    @staticmethod
    def average_speed(model):
        return np.average([agent.speed for agent in model.schedule_car.agents])

    @staticmethod
    def halt_vehicles(model):
        halted_cars = [agent.speed for agent in model.schedule_car.agents].count(3)
        return (len(model.schedule_car.agents) - halted_cars) / len(model.schedule_car.agents)
    
    @staticmethod
    def max_speed(model):
        return np.max([agent.speed for agent in model.schedule_car.agents])
```

Al ejecutar los modelos, se pueden configurar para tener alguno de los siguientes aspectos.
|1 Carril|2 Carriles|3 Carriles|
|:--:|:--:|:--:|
|![](https://cdn.discordapp.com/attachments/890394722139512892/1044375481266491423/Diagrama.png) | ![](https://cdn.discordapp.com/attachments/890394722139512892/1044375503580180560/Diagrama2.png) | ![](https://cdn.discordapp.com/attachments/890394722139512892/1044375481765605486/Diagrama3.png) |

## Resutados

### Modelo de Semáforos Inteligentes

|<br>|1 Carril|2 Carriles|3 Carriles|
|:--:|:--:|:--:|:--:|
| 10 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044376337177133127/unknown.png) ![](https://media.discordapp.net/attachments/1044376326569730178/1044376465896116274/unknown.png?width=855&height=512) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044376671400235128/unknown.png) ![](https://media.discordapp.net/attachments/1044376326569730178/1044376848462790686/unknown.png?width=855&height=512) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044377174477639751/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044377215116255232/unknown.png)|
| 20 Autos | ![](https://media.discordapp.net/attachments/1044376326569730178/1044381624319606784/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044381640840978512/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044380343941210213/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044380313280860181/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044377582520508536/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044377600681853069/unknown.png) |
| 30 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044383097132691547/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044383121757454468/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044383212857737246/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044383232633880606/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044383553649131570/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044383573819523093/unknown.png) |
| 40 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384076288761957/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384097088315412/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384245755428995/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384267473522818/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384589340233769/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384607354765342/unknown.png) |
| 50 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384776871751760/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384795146330162/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384907759198319/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044384924116983930/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385036117479474/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385053234438224/unknown.png) |

### Modelo de Semáforos Intermitentes

|<br>|1 Carril|2 Carriles|3 Carriles|
|:--:|:--:|:--:|:--:|
| 10 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385236059967509/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385254586187776/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385367551377428/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385382915133480/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385490784239627/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385510958825472/unknown.png) |
| 20 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385638755086436/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385653263188008/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385763711791244/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385782678433882/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385897044529242/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044385913884643438/unknown.png) |
| 30 Autos | ![](https://media.discordapp.net/attachments/1044376326569730178/1044386052267315220/unknown.png) ![](https://media.discordapp.net/attachments/1044376326569730178/1044386068734165082/unknown.png) | ![](https://media.discordapp.net/attachments/1044376326569730178/1044386187688816681/unknown.png) ![](https://media.discordapp.net/attachments/1044376326569730178/1044386204990328832/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386313715073154/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386333075968042/unknown.png) |
| 40 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386461211963523/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386481277505546/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386593504497744/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386611670028298/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386720348643368/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386735674634311/unknown.png) |
| 50 Autos | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386870454403082/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386887252582460/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044386992290545744/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044387009373949962/unknown.png) | ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044387134435504169/unknown.png) ![](https://cdn.discordapp.com/attachments/1044376326569730178/1044387156665303050/unknown.png) |

## Análisis y Conclusiones

Por como se pueden observar en las gráficas anteriores, tanto el modelo de semáforos inteligentes como el de semáforos intermitentes tienden a estabilizarse a medida en que los pasos del modelo crecen, aparentemente llegando a un __equilibrio de Nash__ ya que en algunas configuraciónes del modelo, como la __Gráfica X__, la velocidad promedio se mantenía completamente constante. No obstante, es importante recalcar que  si existe una diferencia el margen en el que se estabilizan ambos modelos.

Mientras que en el modelo de semáforos intermitentes la velocidad promedio se estabiliza entre 3 y 1.8 (2.4), el modelo de semáforos inteligentes logra estabilizarse entre 3 y 2.4 (2.7). Esto muestra un aumento de aproximadamente 15% en la velocidad promedio de los coches en tránsito, lo cual traducido a resultados en el mundo real, podría implicar una mejor percepción del tráfico y reducción en el estrés de los conductores.

Por lo tanto, se podría concluir que los semáforos inteligentes tienen una mejora concreta en el flujo del tráfico en comparación con el modelo de semáforos intermitentes.Esto basandonos en el aumento de velocidad promedio de los coches durante la simulación y la reducción del porcentaje de autos detenidos.
