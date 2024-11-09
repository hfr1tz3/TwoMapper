import numpy as np
import plotly.graph_objects as go
from scipy import optimize

''' The following function sample and plot conventional surfaces '''

""" Sample Data from a sphere.
    Input
    --------
    num_points : integer, default 1000. Number of points to sample.
    dim : integer, default 3. Dimension of sphere to sample.
    radius : float, default 1. Radius of sphere to sample.

    Returns
    ---------
    ndarray of shape (num_points, dim)

"""
def sample_sphere(num_points=1000, dim=3, radius=1):
    assert dim > 0 and radius > 0, f'Dimension and radius must be positive.'
    points = np.random.normal(size=(num_points, dim))
    pts = radius * (points.T / np.linalg.norm(points, axis=1)).T
    return pts

""" Sample data from the 2-dimensional torus embedded in R^3 with usual
parametrized embedding.
    Input
    -------
    num_points : integer, default 1000. Number of points to sample.
    R : float, default 3. Outer radius of torus.
    r : float, default 1. Inner radius of torus. 
            Note that r must be less than R.
    Returns
    -------
    ndarray of shape (num_points, 3)
"""
def sample_torus(num_points=1000, R=3, r=1):
    assert R>r, f"This torus will self intersect. Choose R > r. Currently R={R} and r={r}."
    thetas = 2 * np.pi * np.random.normal(size=(num_points,))
    phis = 2 * np.pi * np.random.normal(size=(num_points,))
    xs = (R+r*np.cos(phis))*np.cos(thetas)
    ys = (R+r*np.cos(phis))*np.sin(thetas)
    zs = r*np.sin(phis)
    points = np.vstack((xs,ys,zs)).T
    return points

""" Sample data from the genus 2 torus embedded in R^3.
Note: This can be extended to n-torus. 
In general, the polynomial
f(x) = \prod_{i=1}^n (x-(i-1))(x-i) = x(x-1)^2(x-2)^2 ... (x-(n-1))^2(x-n)
has roots at x = 0,1,...,n.
Then we let g(x,y) = f(x)+y^2 so that the set of points g(x,y)=0 forms n connected loops. Define
F(x,y,z) = g(x,y)^2 + z^2 - r^2
so for small enough r, the level set F(x,y,z)=0 is a torus of genus n.

    Input
    -------
    num_points : integer, default 1000. Number of points to sample.
    thickness : float, default 0.1. Value to determine thickenss of the shape in R^3.
        Should be kept small.

    Return
    -------
    ndarray of shape (num_points, 3)
"""
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
    return np.asarray(point_list)

''' Simple graph function to graph samples in R^2 or R^3.
    Input
    -------
    data : ndarray. Should have shape (n, 2) or (n, 3) where n is the number of data points.

    Returns
    -------
    plotly.graph_objects.figure 

'''
def graph_sample(data):
    if data.shape[1] == 2:
        graph = go.Figure(data=[go.Scatter(x=data[:,0], y=data[:,1], mode='markers')])
    if data.shape[1] == 3:
        graph = go.Figure(data=[go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2], mode='markers')])
    graph.update_traces(marker=dict(size=2))
    return graph.show()