import numpy as np
import matplotlib.pyplot as plt
import math as mat



class flock:

    def __init__(self, boids: np.array, alignment_factor, cohesion_factor, separation_factor, alignment_visual_radius, cohesion_visual_radius,  separation_visual_radius, max_speed, min_speed, max_acceleration, graph_range) -> None:

        self.alignment_factor = alignment_factor
        self.cohesion_factor = cohesion_factor
        self.separation_factor = separation_factor
        self.alignment_visual_radius = alignment_visual_radius
        self.cohesion_visual_radius = cohesion_visual_radius
        self.separation_visual_radius = separation_visual_radius
        self.max_speed = max_speed 
        self.min_speed = min_speed 
        self.max_acceleration = max_acceleration
        self.graph_range = graph_range
        
        self.boids = boids #boids is an array of boid class objects
        self.size = boids.size
        self.time = 0
    
        average_position = np.array([0, 0])
        average_velocity = np.array([0, 0])
        for i in self.boids:
            average_position = np.add(average_position, i.position)
            average_velocity = np.add(average_velocity, i.velocity)
        average_position = average_position/self.size
        average_velocity = average_velocity/self.size

        self.position = average_position
        self.velocity = average_velocity
        self.trajectory = np.array([self.position])
        
    def alignment(self, boid) -> np.array:
        visible_boids = 0
        desired_velocity = np.array([0, 0])
        for i in self.boids:
            if i != boid and np.sqrt(np.sum(np.subtract(i.position, boid.position)**2)) <= self.alignment_visual_radius:
                visible_boids += 1
                desired_velocity = np.add(desired_velocity, i.velocity)
        if visible_boids > 0:
            desired_velocity = desired_velocity/visible_boids
            desired_acceleration = np.subtract(desired_velocity, boid.velocity)
            return desired_acceleration
        else:
            return np.array([0, 0])   

    def cohesion(self, boid) -> np.array:
        visible_boids = 0
        desired_position = np.array([0, 0])
        for i in self.boids:
            if i != boid and  self.cohesion_visual_radius/2 <= np.sqrt(np.sum(np.subtract(i.position, boid.position)**2)) <= self.cohesion_visual_radius:
                visible_boids += 1
                desired_position = np.add(desired_position, i.position)
        if visible_boids > 0:
            desired_position = desired_position/visible_boids
            desired_acceleration = np.subtract(np.subtract(desired_position, boid.position), boid.velocity)
            return desired_acceleration
        else:
            return np.array([0, 0])  

    def separation(self, boid) -> None:
        visible_boids = 0
        desired_velocity = np.array([0, 0])
        for i in self.boids:
            if i != boid and np.sqrt(np.sum(np.subtract(i.position, boid.position)**2)) <= self.separation_visual_radius:
                visible_boids += 1
                sample_velocity = np.subtract(boid.position, i.position)
                sample_velocity = sample_velocity/(np.sum(sample_velocity**2)) #- optional normalisation factor
                desired_velocity = np.add(desired_velocity, sample_velocity)
        if visible_boids > 0:
            desired_velocity = desired_velocity/visible_boids # - again, normalisation factor (optional)
            desired_acceleration = np.subtract(desired_velocity, boid.velocity)
            return desired_acceleration
        else:
            return np.array([0, 0])   

    def boundary(self, boid):
        desired_velocity = np.array([0, 0])
        if boid.position[0] > self.graph_range - 12.50 or boid.position[0] < -self.graph_range + 12.50:
            desired_velocity = -boid.position 
        elif boid.position[1] > self.graph_range -12.50 or boid.position[1] < -self.graph_range + 12.50:
            desired_velocity = -boid.position 
        desired_acceleration = desired_velocity - boid.velocity
        return desired_acceleration


    def update(self) -> None:
        offset = [self.graph_range, self.graph_range]
        for i in self.boids:
            
            #compute net acceleration
            i.acceleration = np.add(self.alignment_factor*self.alignment(i), self.cohesion_factor*self.cohesion(i))
            i.acceleration = np.add(i.acceleration, self.separation_factor*self.separation(i))
            #i.acceleration = np.add(i.acceleration, self.boundary(i))

             #acceleration normalisation
            if np.sqrt(np.sum(i.acceleration**2)) > self.max_acceleration:
                i.acceleration = self.max_acceleration*i.acceleration/np.sqrt(np.sum(i.acceleration**2)) 



            #updat eboids
            i.update()
           
            #i.position = np.subtract(np.mod(np.add(i.position, offset), np.multiply(offset, [2, 2])), offset) #for spherical topology
            
            #velocity normalisation
            if np.sqrt(np.sum(i.velocity**2)) > self.max_speed:
                i.velocity = self.max_speed*i.velocity/np.sqrt(np.sum(i.velocity**2)) 
            elif np.sqrt(np.sum(i.velocity**2)) < self.min_speed:
                i.velocity = self.min_speed*i.velocity/np.sqrt(np.sum(i.velocity**2)) 


        self.time += 1

        
        average_position = np.array([0, 0])
        average_velocity = np.array([0, 0])
        for i in self.boids:
            average_position = np.add(average_position, i.position)
            average_velocity = np.add(average_velocity, i.velocity)
        average_position = average_position/self.size
        average_velocity = average_velocity/self.size

        self.position = average_position
        self.velocity = average_velocity

        self.trajectory = np.concatenate((self.trajectory, [self.position]), axis=0)
        
        
        


    def draw(self) -> None:
        
                
        x_position = np.array([])
        y_position = np.array([])
        x_direction = np.array([])
        y_direction = np.array([])
        x_acceleration = np.array([])
        y_acceleration = np.array([])

        for i in self.boids:
            x_position = np.append(x_position, i.position[0])
            y_position = np.append(y_position, i.position[1])
            x_direction = np.append(x_direction, i.velocity[0]/(100*mat.sqrt(i.velocity[0]**2+i.velocity[1]**2)))
            y_direction = np.append(y_direction, i.velocity[1]/(100*mat.sqrt(i.velocity[0]**2+i.velocity[1]**2)))
            x_acceleration = np.append(x_acceleration, i.acceleration[0])
            y_acceleration = np.append(y_acceleration, i.acceleration[1])

        x_average = self.trajectory[:, 0]
        y_average = self.trajectory[:, 1]

        
        plt.scatter(x_position, y_position)
        plt.quiver(x_position, y_position, x_direction, y_direction, color='b', width=0.005)
        #plt.quiver(x_position, y_position, x_acceleration, y_acceleration, color='g', width=0.005)
        #plt.scatter(self.position[0], self.position[1], color = "r")
        """
        plt.quiver(self.position[0], self.position[1], self.velocity[0], self.velocity[1], color = "r", width=0.005)"""
        #plt.plot(x_average, y_average, marker='', linestyle='--', color='r')  # 'o' for markers at points, '-' for connecting lines
        """plt.xlim(-self.graph_range, self.graph_range)
        plt.ylim(-self.graph_range, self.graph_range)"""
        plt.axis("off")
        """plt.figtext(0.5, 0.01, "max |v| = {}, min |v| = {}, max |a| = {} \n t = {}".format(self.max_speed, self.min_speed, self.max_acceleration, self.time), ha="center", fontsize=10)
        plt.title("Flocking sim: α = {}, β = {}, γ = {}\n α range = {}, β range = {}, γ range = {}".format(self.separation_factor, self.cohesion_factor, self.alignment_factor, self.separation_visual_radius, self.cohesion_visual_radius, self.alignment_visual_radius))
        """
        filename = r"c:\Users\andre\Desktop\School\Math\DiscretePlots\flocking_animation\plot{}.png".format(self.time)
        plt.savefig(filename.format(self.time))
        plt.clf()

        plt.plot(x_average, y_average, marker='')
        plt.figtext(0.5, 0.01, "max |v| = {}, min |v| = {}, max |a| = {} \n t = {}".format(self.max_speed, self.min_speed, self.max_acceleration, self.time), ha="center", fontsize=10)
        plt.title("Flocking sim: α = {}, β = {}, γ = {}\n α range = {}, β range = {}, γ range = {}".format(self.separation_factor, self.cohesion_factor, self.alignment_factor, self.separation_visual_radius, self.cohesion_visual_radius, self.alignment_visual_radius))
        filename = r"c:\Users\andre\Desktop\School\Math\DiscretePlots\flocking_trajectory\plot{}.png".format(self.time)
        plt.savefig(filename.format(self.time))
        plt.clf()

    def __str__(self) -> None:
        
        flock = np.array([])
        for i in self.boids:
            boid = np.array([i.position, i.velocity, i.acceleration])
            flock = np.append(flock, boid)

        return f"{flock}"

    
class boid:

    def __init__(self, position: np.array, velocity: np.array, acceleration: np.array) -> None:
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
   

    def update(self) -> None:
        self.position = np.add(self.position, self.velocity)
        self.velocity = np.add(self.velocity, self.acceleration)
        
        
    def __str__(self) -> None:
        return f"{self.position}, {self.velocity}, {self.acceleration}"

