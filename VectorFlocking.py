import continousflocking as fl
import sympy as sp
import numpy as np

max_velocity = 10


if __name__ == "__main__":

    def P(x, y):
        return -x**2-y**2
    

    # Define the vector field functions
    def Vx(x, y):
        return -y+x+x**2+y**2 # Example x-component

    def Vy(x, y):

        return x+y # Example y-component
    
    def Ax(x, y):
        return 0 # Example x-component

    def Ay(x, y):
        return 0 # Example y-component
    

    flock = fl.flock(P, Vx, Vy, Ax, Ay, alpha = 0, beta = 1, gamma = 1)


    for i in range(0, 1000):
        print()
        print(flock)
        print("-----------------------------------------------------")
        print("    t = ", i)
        print("-----------------------------------------------------")
        flock.plot()
        flock.update()




    """
    
    print(flock)

    flock.update()

    print("-----------------------------------------------------")
    print("")
    print("-----------------------------------------------------")


    print(flock)

    flock.update()

    print("-----------------------------------------------------")
    print("")
    print("-----------------------------------------------------")

    print(flock)

    flock.update()

    print("-----------------------------------------------------")
    print("")
    print("-----------------------------------------------------")

    print(flock)

    flock.update()

    print("-----------------------------------------------------")
    print("")
    print("-----------------------------------------------------")

    print(flock)
"""

    """# sheer calculation
    vf = fl.VectorField(Vx, Vy)
    
    # Calculate and print the divergence
    sheer = vf.sheer()
    print("sheer expression:", sheer.symbolic_x, ",", sheer.symbolic_y)

    # Create a numerical function for divergence
    sheerX = sp.lambdify(('x', 'y'), sheer.symbolic_x)
    
    sheerY = sp.lambdify(('x', 'y'), sheer.symbolic_x)

    # Example: Evaluate sheer at (1, 2)
    sheer_value = [sheerX(1,2), sheerY(1, 2)]
    print("sheer value at (1, 2):", sheer_value)

    
    # div calculation
    
    vf = fl.VectorField(Vx, Vy)
    
    # Calculate and print the divergence
    vf.divergence()
    print("Divergence expression:", vf.symbolic_x)

    # Create a numerical function for divergence
    divergence_func = sp.lambdify(('x', 'y'), vf.symbolic_x)

    # Example: Evaluate divergence at (1, 2)
    divergence_value = divergence_func(0, 0)
    print("Divergence value at (1, 2):", divergence_value)"""