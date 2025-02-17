import numpy as np
import boid_library as bd # type: ignore
from matplotlib.animation import FuncAnimation
import math as mat


alignment_factor = 1
cohesion_factor = 1
separation_factor = 10
alignment_visual_radius = 10
cohesion_visual_radius = 10
separation_visual_radius = 10
min_speed = 0.5
max_speed = 3
max_acceleration = 0.05
graph_size = 20


birds = np.array([])
for i in range(-5, 6):
    for j in range(-5, 6):
       
        position = np.array([2*j, 2*i])
        velocity =  (max_speed)*np.random.random(2) - max_speed/2
        acceleration = np.array([0, 0])

        bird = bd.boid(position, velocity, acceleration)
        

        birds = np.append(birds, bird)




a = bd.flock(birds, alignment_factor, cohesion_factor, separation_factor, alignment_visual_radius, cohesion_visual_radius, separation_visual_radius, max_speed, min_speed, max_acceleration, graph_size)

for j  in range(0, 1000):
    a.draw()
    for i in range(0, 1):
        a.update()
        
        print(a.time)
        
    
a.draw()
print("done")