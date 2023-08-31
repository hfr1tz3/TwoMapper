import numpy as np
from itertools import combinations
from functools import reduce
from operator import iconcat
import plotly
from gtda.mapper.utils._visualization import _get_colors_for_vals

""" Compute higher order simplices from refined Mapper cover to produce a 2Mapper graph.
    Returns 
    --------
    List of 2 simplices for 2Mapper graph. 
    Each 2 simplex is a list, where each entry is a node from 
    the ::class:: `i.graph.Vertex` object generated from the `fit_transform` method.
"""
## We should change this so that we have data series of 
# simplex id, nodes, len(intersection), etc
# Can we make this a dictionary?

def two_dim_nerve(graph, intersection_data = False):
    high_degree_vertices = list(filter(lambda x: x.degree() > 1, 
                                       graph.vs()
                                      )
                               )
    node_triples = combinations(high_degree_vertices,3)
    simplex_list = list()
    intersections = list()
    for triple in node_triples:
        intersection = reduce(np.intersect1d,(triple[0]['node_elements'],
                                              triple[1]['node_elements'],
                                              triple[2]['node_elements']
                                             )
                             )
        if len(intersection) > 0:
            simplex_list.append([triple[0], triple[1], triple[2]])
            if intersection_data == True:
                intersections.append(len(intersection))
    
    if intersection_data == True:
        return simplex_list, intersections
    else:
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
def two_mapper(graph, figure, fancy_edges = False, fancy_simplices = False):
    if type(figure.data[0]) != plotly.graph_objs._scatter3d.Scatter3d:
        raise ValueError("layout_dim must equal 3 to produce 2Mapper graph")
        
    if fancy_edges == True:
        edge_weights = dict(list())
        for i,weight in enumerate(graph.es['weight']):
            if weight not in edge_weights.keys():
                edge_weights[weight] = [i]
            else:
                edge_weights[weight].append(i)
                
    if fancy_simplices == True:
        simplex_list, simplex_intersections = two_dim_nerve(graph,
                                                            intersection_data = True
                                                           )
        opacities = dict(list())
        for i, intersection_value in enumerate(simplex_intersections):
            if intersection_value not in opacities.keys():
                opacities[intersection_value] = [i]
            else:
                opacities[intersection_value].append(i)
    else:
        simplex_list = two_dim_nerve(graph) 
        i = [simplex_list[x][0].index for x in range(len(simplex_list))]
        j = [simplex_list[x][1].index for x in range(len(simplex_list))]
        k = [simplex_list[x][2].index for x in range(len(simplex_list))]
        
    node_pos = np.asarray(graph.layout('kk3d', dim = 3).coords)   
    node_colors = figure.data[1].marker.color
    node_colorscale = figure.data[1].marker.colorscale
    face_color_vals = [np.mean([node_colors[x[0].index],
                                node_colors[x[1].index],
                                node_colors[x[2].index]]) 
                       for x in simplex_list]
    face_colors = _get_colors_for_vals(face_color_vals,
                                   vmin = np.min(node_colors),
                                   vmax = np.max(node_colors),
                                   colorscale = node_colorscale,
                                   return_hex = True)
    if ((fancy_simplices == True) and (fancy_edges == True)):
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
    
    elif ((fancy_simplices == True) and (fancy_edges == False)):
        fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
        for opacity in opacities:
            f = _opacity_trace(opacity, opacities, simplex_list, node_pos, face_colors, node_colorscale)
            fancy_figure.add_trace(f)
        fancy_figure.add_traces([figure.data[0], figure.data[1]])
        return fancy_figure
    
    elif ((fancy_simplices == False) and (fancy_edges == True)):
        fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
        for weight in edge_weights:
            f = _lineweight_trace(weight, edge_weights, figure)
            fancy_figure.add_trace(f)
        simplex_trace = plotly.graph_objects.Mesh3d(x=node_pos[:,0], y=node_pos[:,1], z=node_pos[:,2],
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
                          legendrank = 2000
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
    f['i'] = [simplex_list[x][0].index for x in opacities[opacity]]
    f['j'] = [simplex_list[x][1].index for x in opacities[opacity]]
    f['k'] = [simplex_list[x][2].index for x in opacities[opacity]]
    f['x'] = node_pos[:,0]
    f['y'] = node_pos[:,1]
    f['z'] = node_pos[:,2]
    f['facecolor'] = [face_colors[x] for x in opacities[opacity]]
    f['colorscale'] = node_colorscale
    f['name'] = f'simplex_trace_opacity_{opacity}'
    f['opacity'] = float((opacity-min(opacities))/(max(opacities)-min(opacities)))
    f['legendrank'] = 2000
                         
    return plotly.graph_objects.Mesh3d(f)