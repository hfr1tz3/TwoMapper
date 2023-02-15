import numpy as np
import plotly.graph_objects as go
from scipy import optimize

''' The following function sample and plot conventional surfaces '''

## Sample from sphere
def sample_spherical(num_points, dim=3, radius=1):
    vec_list = [] 
    for _ in range(num_points):
        # Generate a random vector
        vec = np.random.normal(0, 1, dim)
        # Normalize and scale to desired radius
        vec /= np.linalg.norm(vec, axis=0)
        vec *= radius
        vec_list.append(list(vec))
    return vec_list

## Sample from the torus
def sample_torus(num_points, R=3, r=1):
    assert R>r, "This torus will self intersect. Choose R>r."
    point_list = []
    for _ in range(num_points):
        #Try uniform?
        theta = 2*np.pi*np.random.normal(0,1)
        phi = 2*np.pi*np.random.normal(0,1)
        x = (R+r*np.cos(phi))*np.cos(theta)
        y = (R+r*np.cos(phi))*np.sin(theta)
        z = r*np.sin(phi)
        point_list.append([x,y,z])
    return point_list

## Sample from the genus 2 torus
'''this could be extended to genus n torus, perhaps we could work on that later
In general, the polynomial
f(x) = \prod_{i=1}^n (x-(i-1))(x-i) = x(x-1)^2(x-2)^2 ... (x-(n-1))^2(x-n)
has roots at x = 0,1,...,n.
Then we let g(x,y) = f(x)+y^2 so that the set of points g(x,y)=0 forms n connected loops. Define
F(x,y,z) = g(x,y)^2 + z^2 - r^2
so for small enough r, the level set F(x,y,z)=0 is an genus n torus.
'''
def sample_g2torus(num_points, thickness=0.1):
    # Implicit function which defines the genus 2 torus
    def F(x,y,z,e):
        return (x*(x-1)**2*(x-2)+y**2)**2+z**2-e**2
    
    point_list = []
    while len(point_list)<num_points:
        x = np.random.uniform(low=-1,high=3, size =(num_points))
        y = np.random.uniform(low=-1,high=1, size =(num_points))
        for (xi,yi) in zip(x,y):
            g= lambda z: F(xi,yi,z,thickness)
            # Attempt to find solution to F(xi,yi,z). 
            # This will be our sample point
            res = optimize.fsolve(g, 1, full_output=1)
            if res[2] == 1:
                zii = res[0]
                zi = zii[0]
                point_list.append([xi,yi,zi])
                if len(point_list) == num_points:
                    break
    return point_list

## Quick graph of points
'I should edit this to include more arguments for plotly'
def graph_sample(point_list):
    xi = [point_list[i][0] for i in range(len(point_list))]
    yi = [point_list[i][1] for i in range(len(point_list))]
    zi = [point_list[i][2] for i in range(len(point_list))]
    
    graph = go.Figure(data=[go.Scatter3d(x=xi,y=yi,z=zi,mode='markers')])
    return graph.show()