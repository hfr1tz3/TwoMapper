import numpy as np
from itertools import combinations
from functools import reduce
from operator import iconcat
import plotly
from gtda.mapper.utils._visualization import _get_colors_for_vals
import gtda.mapper.cover
#(_find_interval_limits,_cover_limits, _limits_from_ranks)
from gtda.mapper.utils._cover import _remove_empty_and_duplicate_intervals, _check_has_one_column


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
    node_triples = combinations(graph.vs(),3)
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
def two_mapper(graph, figure, fancy_output = False):
    if type(figure.data[0]) != plotly.graph_objs._scatter3d.Scatter3d:
        raise ValueError("layout_dim must equal 3 to produce 2Mapper graph")
    if fancy_output == True:
        edge_weights = dict(list())
        for i,weight in enumerate(graph.es['weight']):
            if weight not in edge_weights.keys():
                edge_weights[weight] = [i]
            else:
                edge_weights[weight].append(i)
        simplex_list, simplex_intersections = two_dim_nerve(graph, intersection_data = True)
        opacities = dict(list())
        for i, intersection_value in enumerate(simplex_intersections):
            if intersection_value not in opacities.keys():
                opacities[intersection_value] = [i]
            else:
                opacities[intersection_value].append(i)
    else:
        simplex_list = two_dim_nerve(graph)
    
    node_pos = np.asarray(graph.layout('kk3d', dim = 3).coords)
    i = [simplex_list[x][0].index for x in range(len(simplex_list))]
    j = [simplex_list[x][1].index for x in range(len(simplex_list))]
    k = [simplex_list[x][2].index for x in range(len(simplex_list))]
    
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
    if fancy_output == True:
        fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
        #for weight in edge_weights:
            
        

        
        return fancy_figure
        
    else:
        figure.add_mesh3d(x=node_pos[:,0], y=node_pos[:,1], z=node_pos[:,2],
                          i=i, j=j, k=k,
                          facecolor = face_colors, 
                          colorscale = node_colorscale,
                          name = 'simplex_trace'
                          #intensity = intersections,
                          #intensitymode = "cell"
                         )
        return figure

    ''' Intensity could be a scale for face colors, however
    it doesn't really work.
    Opacity scales the entire trace so unusable.
    THINK AbOUT ADDING FACTORS WHERE WE ADD TRACES BASED
    ON OPACITY AND INTERSECTION VALUES
    This will drastically increase run time, but things will
    look nicer.
    '''
        
#def _lineweight_trace():

#def _opacity_trace():
        
        
def circle_cover(data):
    
    return cover