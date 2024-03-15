import numpy as np
from itertools import combinations, compress
from functools import reduce
from operator import iconcat
import plotly
from math import factorial
from gtda.mapper.utils._visualization import _get_colors_for_vals

""" Compute two-dimensional simplices from refined Mapper cover to produce a 2Mapper graph.
    Returns 
    --------
    List of tuples (n,m,k) representing the vertices to a 2-simplex.
    Each entry in the tuple the index of a ::class:: `i.graph.Vertex` object 
    generated from the `fit_transform` method.

    Array of length (number of 2-simplices,) which contains the
    number of data points in the 2-simplex at the corresponding index.
"""

def list_2simplices(graph, intersection_data = False):
    node_triples = graph.list_triangles()
    num_triples = len(node_triples)
    simplex_list = [None] * (num_triples+1)
    simplex_mask = [False] * (num_triples+1)
    intersections = np.zeros(num_triples)
    for i, triple in enumerate(node_triples):
        intersection = reduce(np.intersect1d, 
                              graph.vs[triple]['node_elements']
                             )
        if len(intersection) > 0:
            simplex_mask[i] = True
            simplex_list[i] = triple
            if intersection_data:
                intersections[i] = len(intersection)
    simplex_list = list(compress(simplex_list, simplex_mask))
    if intersection_data:
        nonzero_intersections = intersections[intersections > 0]
        return simplex_list, nonzero_intersections
    if intersection_data is False:
        return simplex_list

""" Display the 2Mapper graph for a 3-dimensional Mapper graph produced in giotto-tda.
    Input : graph : `i.graph`
    Returns
    --------
    graph : :class: `plotly.graph_objs._figurewidget.FigureWidget` object
        Undirected 2Mapper graph according to the two-dimensional nerve computed with
        the `two_dim_nerve` method. Each triangle in the 3d mesh is a 2-simplex where 
        its bounding nodes have nonzero intersection.

"""
def two_mapper(fancy_edges = False, fancy_simplices = False, **mapper_figure_dict):
    if mapper_figure_dict['layout_dim'] != 3:
        raise ValueError("layout_dim must equal 3 to produce 2Mapper graph")
    graph = mapper_figure_dict['pipeline'].fit_transform(data)
    figure = plot_static_mapper_graph(**mapper_figure_dict)
    # Fancy edges makes the graph look clunky
    if fancy_edges is True:
        warnings.warn('fancy_edges can cause visually clunky graphs. Proceed with caution.', RuntimeWarning)
        edge_weights = dict(list())
        for i,weight in enumerate(graph.es['weight']):
            if weight not in edge_weights.keys():
                edge_weights[weight] = [i]
            else:
                edge_weights[weight].append(i)             
    if fancy_simplices is True:
        simplex_list, simplex_intersections = list_2simplices(graph, intersection_data = True)
        opacities = dict(list())
        for intersection_value in set(simplex_intersections):
            if intersection_value not in opacities.keys():
                simplices = np.argwhere(
                    np.asarray(simplex_intersections) == intersection_value
                )
                opacities[intersection_value] = list(simplices.reshape(len(simplices),))
            else:
                continue
    if fancy_simplices is False:         
        simplex_list = list_2simplices(graph)
        num_simplices = len(simplex_list)
        i = np.full(num_simplices, -1)
        j = np.full(num_simplices, -1)
        k = np.full(num_simplices, -1)
        for x in range(num_simplices):
            i[x] = simplex_list[x][0]
            j[x] = simplex_list[x][1]
            k[x] = simplex_list[x][2]
    node_pos = np.asarray(graph.layout(mapper_figure_dict['layout'], dim = 3).coords)   
    node_colors = figure.data[1].marker.color
    node_colorscale = figure.data[1].marker.colorscale
    face_color_vals = _get_simplex_colors(node_colors, simplex_list)
    face_colors = _get_colors_for_vals(face_color_vals,
                                   vmin = np.min(node_colors),
                                   vmax = np.max(node_colors),
                                   colorscale = node_colorscale,
                                   return_hex = True)
    if ((fancy_simplices is True) and (fancy_edges is True)):
        fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
        
        # We first want to add the 2-simplicies with varying opacities
        for opacity in opacities:
            f = _opacity_trace(opacity, opacities, simplex_list, node_pos, face_colors, node_colorscale)
            fancy_figure.add_trace(f)
        # We then add the edges with varying line weights
        # Note we cannot change the color of each edge due to limitations 
        # in Plotly.
        # If we wanted to do this we would need a trace for each edge color.
        for weight in edge_weights:
            f = _lineweight_trace(weight, edge_weights, figure)
            fancy_figure.add_trace(f)

        # Add the node trace from the original figure
        fancy_figure.add_trace(figure.data[1])
        return fancy_figure
    
    elif ((fancy_simplices is True) and (fancy_edges is False)):
        fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
        for opacity in opacities:
            f = _opacity_trace(opacity, opacities, simplex_list, node_pos, face_colors, node_colorscale)
            fancy_figure.add_trace(f)
        fancy_figure.add_traces([figure.data[0], figure.data[1]])
        return fancy_figure
    
    elif ((fancy_simplices is False) and (fancy_edges is True)):
        fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
        for weight in edge_weights:
            f = _lineweight_trace(weight, edge_weights, figure)
            fancy_figure.add_trace(f)
        simplex_trace = plotly.graph_objects.Mesh3d(
            x=node_pos[:,0], y=node_pos[:,1], z=node_pos[:,2],
            i=i, j=j, k=k,
            facecolor = face_colors, 
            colorscale = node_colorscale,
            name = 'simplex_trace',
            legendrank = 2000
        )
        fancy_figure.add_traces([simplex_trace, figure.data[1]])
        return fancy_figure
        
    else:
        figure.add_mesh3d(x=node_pos[:,0], y=node_pos[:,1], z=node_pos[:,2],
                          i=i, j=j, k=k,
                          facecolor = face_colors, 
                          colorscale = node_colorscale,
                          name = 'simplex_trace',
                          legendrank = 2000,
                          opacity = 0.7
                         )
        return figure
        
def _lineweight_trace(weight, edge_weights, figure):
    f = dict()
    f['hoverinfo'] = 'none'
    f['mode'] = 'lines'
    f['line'] = {'color': '#888',
                 'width': weight} 
    f['name'] = f'edge_trace_weight_{weight}'
    f['legendrank'] = 1500
    f['x'] = list()
    f['y'] = list()
    f['z'] = list()
    for edge in edge_weights[weight]:
        f['x'].extend(figure.data[0]['x'][3*edge:3*edge+3])
        f['y'].extend(figure.data[0]['y'][3*edge:3*edge+3])
        f['z'].extend(figure.data[0]['z'][3*edge:3*edge+3])
    return plotly.graph_objects.Scatter3d(f)

def _opacity_trace(opacity, opacities, simplex_list, node_pos, face_colors,
                   node_colorscale):  
    f = dict()
    f['i'] = np.full(len(opacities[opacity]), -1)
    f['j'] = np.full(len(opacities[opacity]), -1)
    f['k'] = np.full(len(opacities[opacity]), -1)
    f['facecolor'] = ['a'] * int(len(opacities[opacity]))
    for i, x in enumerate(opacities[opacity]):
        f['i'][i] = simplex_list[x][0]
        f['j'][i] = simplex_list[x][1]
        f['k'][i] = simplex_list[x][2]
        f['facecolor'][i] = face_colors[x]
    f['x'] = node_pos[:,0]
    f['y'] = node_pos[:,1]
    f['z'] = node_pos[:,2]
    f['colorscale'] = node_colorscale
    f['name'] = f'simplex_trace_opacity_{opacity}'
    if len(opacities) > 1:
        f['opacity'] = float((opacity-min(opacities))/(max(opacities)-min(opacities)))
    if len(opacities) == 1:
        f['opacity'] = 1
    f['legendrank'] = 2000                     
    return plotly.graph_objects.Mesh3d(f)

def _get_simplex_colors(node_colors, simplex_list):
    face_color_vals = np.full(len(simplex_list), -1)
    for i, x in enumerate(simplex_list):
        face_color_vals[i] = np.mean(
            [node_colors[x[0]],
            node_colors[x[1]],
            node_colors[x[2]]]
        )
    return face_color_vals