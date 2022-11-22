import mesa
import numpy as np
from random import randint, choice

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

class TrafficLineAgent (mesa.Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)
        self.type = 'placeholder'
        self.status = 'gray'

