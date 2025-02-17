import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Callable

def my_lambdify(function):
    # Convert symbolic expression to a numerical function
    return sp.lambdify((sp.symbols('x'), sp.symbols('y')), function, 'numpy')

def symbolise(function):
    x, y = sp.symbols('x y')
    # Convert Python functions to symbolic expressions
    return function(x, y)







class VectorField:
    def __init__(self, function_x: Callable[[float, float], float], function_y: Callable[[float, float], float]):

        
    
        self.x = symbolise(function_x)
        self.y = symbolise(function_y)

    def simplify(self):
        self.x = sp.simplify(self.x)
        self.y = sp.simplify(self.y)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):

            x_component = my_lambdify(self.x * scalar)
            y_component = my_lambdify(self.y * scalar)

            return VectorField(x_component, y_component)
        return NotImplemented


    def negate(self):
        
        self.x = -(self.x)
        self.y = -(self.y)

    def __repr__(self):
        return f"VectorField(x={self.x}, y={self.y})"

    def __str__(self):
        return f"VectorField: \n x-component: {self.x}\n  y-component: {self.y}"



def add(field1: VectorField, field2: VectorField) -> VectorField:
    # Add the components of two vector fields
    sumX = field1.x + field2.x
    sumY = field1.y + field2.y
    """
    sumX = sp.simplify(field1.x + field2.x)
    sumY = sp.simplify(field1.y + field2.y)
    """
    return VectorField(lambda x, y: sumX, lambda x, y: sumY)
    
def gradient(function):

    x, y = sp.symbols('x y')

    gradX = -sp.diff(function, x, 1)
    gradY = -sp.diff(function, y, 1)

    return VectorField(lambda x, y: gradX, lambda x, y: gradY)

def cohesion(function):
    
        # Step 1: Define symbolic variables
        x, y = sp.symbols('x y')

        cohesion = function
        # Step 3: Compute the symbolic divergence
        cohesionX = (sp.diff(function, x, 1))**2/((sp.diff(function, x, 1))+1)
        
        cohesionY = (sp.diff(function, y, 1))**2/((sp.diff(function, y, 1))+1)
        # Step 4: Simplify the divergence (optional)

        cohesion = VectorField(lambda x, y: cohesionX, lambda x, y: cohesionY)
        #cohesion.simplify()

        return cohesion

def sheer(vectorFunction: VectorField) -> VectorField:
    
    # Step 1: Define symbolic variables
    x, y = sp.symbols('x y')

    # Step 3: Compute the symbolic divergence
    sheer_x = sp.diff(vectorFunction.x, x, 2) + sp.diff(vectorFunction.x, y, 2)
    sheer_y = sp.diff(vectorFunction.y, x, 2) + sp.diff(vectorFunction.y, y, 2)
    # Step 4: Simplify the divergence (optional)

    

    """
    sheer_x = sp.simplify(sheer_x)
    sheer_y = sp.simplify(sheer_y)
    """


    sheer = VectorField(my_lambdify(sheer_x), my_lambdify(sheer_y))
    
    return sheer

def divergence(vectorFunction: VectorField):
    # Define symbolic variables
    x, y = sp.symbols('x y')

    

    # Compute the symbolic divergence
    divergence = sp.diff(vectorFunction.x, x) + sp.diff(vectorFunction.y, y)

    # Simplify the divergence (optional)
    #divergence_simplified = sp.simplify(divergence)
    
    return divergence


class flock:
    
    def __init__(self, populationDensity: Callable[[float, float], float], 
                 velocityFieldX: Callable[[float, float], float], 
                 velocityFieldY: Callable[[float, float], float],
                 accelerationFieldX: Callable[[float, float], float],
                 accelerationFieldY: Callable[[float, float], float], alpha = 1, beta = 1, gamma = 1):
        

        self.populationDensity = symbolise(populationDensity)
        self.velocityField = VectorField(velocityFieldX, velocityFieldY)
        self.accelerationField = VectorField(accelerationFieldX, accelerationFieldY)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.time = 0
    
    def update(self):

        self.time += 1

        print("Computing Divergence")
        self.populationDensity = self.populationDensity-self.populationDensity*divergence(self.velocityField)
        print("Divergence Computed")
        print("Simplifying Population Density")
        #self.populationDensity = sp.simplify(self.populationDensity)
        print("Population Density Simplified")
        print("Adding Acceleration Field to Velocity Field")
        #print("Velocity Field:", self.velocityField)
        #print("Acceleration Field: ", self.accelerationField)
        print("Adding Acceleration Field to Velocity Field")
        self.velocityField = add(self.velocityField, self.accelerationField)
        #print("Velocity Field:", self.velocityField)
        print("Acceleration Field Added to Velocity Field")
        print("Computing Acceleration Field")


        if -(self.alpha-self.beta) == 0:
            self.accelerationField = sheer(self.velocityField)*self.gamma
        elif self.gamma == 0:
            self.accelerationField = gradient(self.populationDensity)*(-self.alpha+self.beta)
        else:    
            self.accelerationField = add(gradient(self.populationDensity)*(-self.alpha+self.beta), sheer(self.velocityField)*self.gamma)
        
        #print("Velocity Field:", self.velocityField)
        print("Gradient  and Sheer (Alignment) Added")
        print("Acceleration Field Computed")


    def plot(self):

        x_vals = np.linspace(-5, 5, 400)  # range for x values
        y_vals = np.linspace(-5, 5, 400)  # range for y values
        x, y = np.meshgrid(x_vals, y_vals)  # create a grid of x and y


        z = my_lambdify(self.populationDensity)
        z = z(x, y)

        plt.figure(figsize=(8, 6))
        plt.contourf(x, y, z, 100, cmap='gray')
        plt.colorbar(label="Function Value")
        plt.title("Heatmap of the Given Function")
        plt.xlabel('x')
        plt.ylabel('y')

        x_vals = np.linspace(-5, 5, 20)  # range for x values
        y_vals = np.linspace(-5, 5, 20)  # range for y values
        x, y = np.meshgrid(x_vals, y_vals)  # create a grid of x and y

        X = my_lambdify(self.velocityField.x)
        Y = my_lambdify(self.velocityField.y)

        X = X(x, y)
        Y = Y(x, y)


       
        magnitude = np.sqrt(X**2 + Y**2)
        vx_norm = X / (magnitude - 1e-2)
        vy_norm = Y / (magnitude - 1e-2)

        plt.quiver(x, y, 
                vx_norm, vy_norm, 
                color='blue', scale=2.5, width=0.005, scale_units="xy", angles="xy")


        

        

        filename = r"C:\Users\andre\Desktop\School\Math\continuous_flocking_animation\plot{}.png".format(self.time)
        plt.savefig(filename.format(self.time))
        plt.clf()

        
                     
    def __repr__(self):
        return f"Flock(P = {self.populationDensity}, V = {self.velocityField}, A = {self.accelerationField})"

    def __str__(self):
        return f"Flock(Population Density = {self.populationDensity},\n Velocity Field = {self.velocityField},\n Acceleration Field = {self.accelerationField})"
     

