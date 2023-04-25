import numpy as np
from collections import defaultdict
from kmapper import nerve
from kmapper.plotlyviz import (plotlyviz, get_mapper_graph, plotly_graph)
from itertools import combinations
import plotly.graph_objects as go

'''
The following file will be used to compute the 2-dimensional simplex of a Mapper graph, and plot the mapper graph. 

This can be, in theory, added to the kepler-mapper repository (whenever I find the time to do so).
'''
# NERVE --------------------------------------------
def two_simplex(simplicial_complex):
    'For a mapper simplicial complex constructed using km.KeplerMapper.map we want to find higher dimensional simplices in the Cech-complex.'

    two_simplices = defaultdict(list)
    candidates = combinations(simplicial_complex["nodes"].keys(),3)
    for candidate in candidates:
        if len(set(simplicial_complex["nodes"][candidate[0]]).intersection(
            set(simplicial_complex["nodes"][candidate[1]]),
            set(simplicial_complex["nodes"][candidate[2]]))) > 0:
            two_simplices[candidate[0]] += [[candidate[1],candidate[2]]]
    'Output is a list of 2-dimensional simplices in the Cech-complex'       
    return two_simplices

# GRAPHING-----------------------------------------------
def graph_2Mapper(mapper_graph, simplicial_complex):
    'For an already constructed mapper_graph from simplicial_complex, we extend this to visualize 2-dimensional simplices, ie. a 2mapper_graph.'

    # Need to construct a dictionary of all 2-dimensional simplices
    kmgraph, _, __ = get_mapper_graph(simplicial_complex)
    Tri = find_triangles(kmgraph)
    traces = find_traces(Tri, kmgraph)
    
    for i in range(len(list(traces))):
        mapper_graph.add_trace(traces[f'trace{i}'])
    return mapper_graph


'Creates a dictionary of all 2-simplices in the 2-dimensional nerve of simplicial_complex.'
'This can most likely be simplified if we auto specify a 2-dimensional mapper'
def find_triangles(kmgraph):
    tri = {'triangles': []}
    candidates = combinations(kmgraph['nodes'],3)
    for i, candidate in enumerate(candidates):
        if len(set(candidate[0]['member_ids']).intersection(
            set(candidate[1]['member_ids']),
            set(candidate[2]['member_ids']))) > 0:
            t = {'id' : i,
                 'v0' : candidate[0]['id'],
                 'v1' : candidate[1]['id'],
                 'v2' : candidate[2]['id'],
                 'color' : np.average([candidate[i]['color'] for i in range(3)])}
            tri["triangles"].append(t)
    return tri

'Create traces which will fill in each 2-simplex in the mapper graph'
'The set of traces $\{D_{\alpha}\}$ we will append to the 1-mapper graph will be a dictionary of simplices $(a_{\alpha},b_{\alpha},c_{\alpha})\in D_{\alpha}$, where for all $(a,b,c),(d,e,f)\in D_{\alpha}$ we have that $a=d$ and $b=e$. Choosing our dictionaries this way is strictly due to the fact that the `fill="toself"` argument in a Scatter trace in `Plotly` is dependent on edge orientation when filling a plot. '
def find_traces(Tri, kmgraph):
    # First we should sort Tri(2-dimensional simplex dictionary) into bins 
    # where each bin contains simplices with identical v0, v1. The order does matter!
    
    simplex_groups = {}
    
    for i,simplex in enumerate(Tri['triangles']):
        # Generate the first trace of the trace dictionary
        if i == 0:
            simplex_groups[f'trace{i}'] = [Tri['triangles'][i]]
        # See if the current simplex has 1 edge matching the simplex before it. 
        # If it matches append to most recent trace. If not add new trace to dict
        if i > 0:
            if simplex_groups[list(simplex_groups)[-1]][0]['v0'] == Tri['triangles'][i]['v0'] and \
            simplex_groups[list(simplex_groups)[-1]][0]['v1'] == Tri['triangles'][i]['v1']:
                simplex_groups[list(simplex_groups)[-1]].append(Tri['triangles'][i])
                
            else:
                k = len(list(simplex_groups))
                simplex_groups[f'trace{k}'] = [Tri['triangles'][i]]
    #print(simplex_groups)   
    # Now with our dictionarys sorted correctly (hopefully) 
    # we can find our coordinates for graphing
    nodetrace = plotly_graph(kmgraph)[1]
    simplex_traces = {}
    k = len(list(simplex_groups))
    for i in range(k):
        xcoord_list = []
        ycoord_list = []
        for j in range(len([list(simplex_groups)[i]])):
            xcoords = [nodetrace['x'][simplex_groups[list(simplex_groups)[i]][j]['v0']], 
                       nodetrace['x'][simplex_groups[list(simplex_groups)[i]][j]['v1']],
                       nodetrace['x'][simplex_groups[list(simplex_groups)[i]][j]['v2']],
                       nodetrace['x'][simplex_groups[list(simplex_groups)[i]][j]['v0']],
                       None]
            ycoords = [nodetrace['y'][simplex_groups[list(simplex_groups)[i]][j]['v0']],
                       nodetrace['y'][simplex_groups[list(simplex_groups)[i]][j]['v1']],
                       nodetrace['y'][simplex_groups[list(simplex_groups)[i]][j]['v2']],
                       nodetrace['y'][simplex_groups[list(simplex_groups)[i]][j]['v0']],
                       None]
            
            xcoord_list.extend(xcoords)
            ycoord_list.extend(ycoords)
        # Once we have are coordinates for each trace we construct our graph trace
        # We will include 'color' arguments so that the coloring makes sense once
        # we can get this thing to run properly
        simplex_traces[list(simplex_groups)[i]] = dict(
            type = 'scatter',
            x = xcoord_list,
            y = ycoord_list,
            fill = 'toself')
           
    return simplex_traces    