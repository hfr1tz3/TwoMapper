import igraph
import scipy
import plotly
import numpy as np
import networkx as nx
from gtda.mapper.utils._visualization import _get_colors_for_vals

""" From a standard mapper graph computed using gtda.mapper, comute a directed graph as follows:

NOTE: For temporal mapper to work, data must be ordered with respect to time.
For higher efficiency, compute initial mapper graph object with `store_edge_elements=True` in the mapper pipeline.


Given a mapper graph $\mathcal{M} = (V, E)$ constructed on data set $(X,T)$, we create the temporal mapper graph $\tilde{\mathcal{M}} = (V, \tilde{E})$ as follows:

For each edge $e\in E$ write $e = (v_1, v_2)$ where $v_1, v_2\in V$. Each vertex $v_i\in V$ contains data points which are spatially connected, call these sets $\bar{v_1}$ resp $\bar{v_2}$. First, consider data $N_e = \bar{v_1}\cap\bar{v_2}$. If for some $1\leq j\leq N-1$ there exist points $x_j, x_{j+1}\in N_e$ we will draw $e = (v_1, v_2)$ as a directed edge.

We additionally will draw $e$ a directed edge if there exists a $1\leq j\leq N-1$ such that $x_j \in \bar{v_1}$ and $x_{j+1}\in\bar{v_2}$.

For edges in $E$ which do not meet this condition we will include them in $\tilde{E}$ as double (both directions) edges.

    Input
    -------
    ::class:: `igraph.Graph` object.

    Returns
    --------
    ::class:: `igraph.Graph` which is directed.
"""

def find_temporal_edges(graph):
    num_edges = len(graph.es)
    # boolean vector describe if an edge is directed or not
    directed_edges = np.full(num_edges, False, dtype=bool)
    # direction of the edge: 
    # 0: double edge (source <--> target)
    # > 0: source --> target
    # < 0: source <-- target
    edge_direction = np.full(num_edges, 0, dtype=int)
    for edge in graph.es():
        source_data = graph.vs[edge.source]['node_elements']
        target_data = graph.vs[edge.target]['node_elements']
        source_data = source_data.reshape((len(source_data),1))
        target_data = target_data.reshape((len(target_data),1))
        
        difference_matrix = scipy.spatial.distance.cdist(source_data, target_data, lambda u,v: v-u).flatten()
        forward_count = np.argwhere(difference_matrix == 1).flatten().size
        backward_count = -1 * np.argwhere(difference_matrix == -1).flatten().size
        if forward_count == 0 and backward_count == 0:
            continue
        else:
            directed_edges[edge.index] = True
            edge_direction[edge.index] = forward_count + backward_count

    return directed_edges, edge_direction

def temporal_subgraph(graph, include_spacial=False):
    digraph = nx.DiGraph()
    for edge in graph.es():
        source_data = graph.vs[edge.source]['node_elements']
        target_data = graph.vs[edge.target]['node_elements']
        source_data = source_data.reshape((len(source_data),1))
        target_data = target_data.reshape((len(target_data),1))
        
        difference_matrix = scipy.spatial.distance.cdist(source_data, target_data, lambda u,v: v-u).flatten()
        forward_count = np.argwhere(difference_matrix == 1).flatten().size
        backward_count = -1 * np.argwhere(difference_matrix == -1).flatten().size
        edge_direction = forward_count + backward_count
        if forward_count == 0 and backward_count == 0 and not include_spacial:
            continue
    
        # forward oriented edge
        if edge_direction > 0:
            digraph.add_edge(edge.source, edge.target, travel_weight=np.abs(edge_direction)+1)
        # backward oriented edge
        if edge_direction < 0:
            digraph.add_edge(edge.target, edge.source, travel_weight=np.abs(edge_direction)+1)
        # double edge
        if edge_direction == 0:
            digraph.add_edges_from([(edge.source, edge.target), (edge.target, edge.source)], travel_weight=1)

    return digraph

"""
Find trajectories between nodes using the temporally witnessed graph.
We do so by constructing directed graph where directed edges are formed from 
edges that 'witness time' via the method ::meth::`find_temporal_edges`.
We find trajectories by traversing the highest weight path of a given starting
point. We default choose the node in the mapper graph of the highest density
(contains the most data points).

    Input
    -------
    graph: `igraph.Graph` object.
    starting_point: int.
        Data point to start the trajectory. Integer should represent the 
        argument of the dataset which contains the starting point.
        This point should be contained somewhere in the mapper graph.

    Reutrns
    --------
    List of nodes in graph in the trajectory.

"""
def find_trajectories(graph, starting_point, end_point=None, include_spacial=False, weight=None):
    if type(graph) is igraph.Graph:
        digraph = temporal_subgraph(graph, include_spacial)
    if type(graph) is nx.classes.digraph.DiGraph:
        digraph = graph
    length, path = nx.single_source_dijkstra(digraph, starting_point, target=end_point, weight=weight)    
    return length, path

"""
Find cycles of a temporal mapper graph. Do so by finding paths between all pairs of points.
Combine this paths, ie. shortest path x->y + shortest path y->x. If this cycle is simple then 
it is considered a 'cycle'.

"""
def find_cycles(digraph, max_number=None):
    node_in_cycle = np.full(len(digraph.nodes()), False, dtype=bool)
    cycles = sorted(list(nx.chordless_cycles(digraph)), key = lambda x: len(x), reverse=True)
    cycle_list = []
    for cycle in cycles:
        if np.count_nonzero(node_in_cycle[cycle]) < (len(cycle) // 2):
            node_in_cycle[cycle] = True
            cycle_list.append(cycle)

        if len(cycle_list) == max_number:
            break
    return cycle_list

def _edgecolor_trace(edge_mask, edge_direction, figure, color, name):
    directed_edges = edge_direction[edge_mask[edge_direction]]
    f = dict()
    f['hoverinfo'] = 'none'
    f['mode'] = 'lines'
    f['line'] = {'color': color} 
    f['name'] = name
    # f['legendrank'] = 1500
    f['x'] = list()
    f['y'] = list()
    f['z'] = list()
    for edge in directed_edges:
        f['x'].extend(figure.data[0]['x'][3*edge:3*edge+3])
        f['y'].extend(figure.data[0]['y'][3*edge:3*edge+3])
        f['z'].extend(figure.data[0]['z'][3*edge:3*edge+3])
    return plotly.graph_objects.Scatter3d(f)

def _cycle_trace(cycle, graph, figure, color, cycle_number):
    vertex_pairs = [(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)]
    vertex_pairs.append((cycle[-1], cycle[0]))
    graph_edges = graph.get_eids(vertex_pairs)
    f = dict()
    f['hoverinfo'] = 'name'
    f['mode'] = 'lines'
    f['line'] = {'color': color,
                 'width': 10} 
    f['name'] = f'Cycle {cycle_number}'
    f['x'] = list()
    f['y'] = list()
    f['z'] = list()
    f['showlegend'] = True
    for edge in graph_edges:
        f['x'].extend(figure.data[0]['x'][3*edge:3*edge+3])
        f['y'].extend(figure.data[0]['y'][3*edge:3*edge+3])
        f['z'].extend(figure.data[0]['z'][3*edge:3*edge+3])
    return plotly.graph_objects.Scatter3d(f)

def _trajectory_trace(trajectory, graph, figure, color, trajectory_number):
    vertex_pairs = [(trajectory[i], trajectory[i+1]) for i in range(len(trajectory)-1)]
    graph_edges = graph.get_eids(vertex_pairs)
    f = dict()
    f['hoverinfo'] = 'name'
    f['mode'] = 'lines'
    f['line'] = {'color': color,
                 'width': 10} 
    f['name'] = f'Trajectory {trajectory[0]} --> {trajectory[-1]}'
    f['x'] = list()
    f['y'] = list()
    f['z'] = list()
    f['showlegend'] = True
    for edge in graph_edges:
        f['x'].extend(figure.data[0]['x'][3*edge:3*edge+3])
        f['y'].extend(figure.data[0]['y'][3*edge:3*edge+3])
        f['z'].extend(figure.data[0]['z'][3*edge:3*edge+3])
    return plotly.graph_objects.Scatter3d(f)

def temporal_trajectories(starting_point, figure, graph, digraph=None, max_number=None, weight=None, end_point=None):
    fancy_figure = plotly.graph_objects.FigureWidget(layout=figure.layout)
    fancy_figure.add_trace(figure.data[0])
    assert type(graph) is igraph.Graph, 'graph must be original igraph.Graph object'
    assert digraph is None or type(digraph) is nx.classes.digraph.DiGraph, 'digraph must be networkx digraph or None'
    if digraph is None:
        digraph = temporal_subgraph(graph, include_spacial)

    path_lens, path = find_trajectories(digraph, starting_point, end_point=end_point, weight=weight)
    if end_point is not None:
        paths = [path]
        end_points = end_point
    if end_point is None:
        num_paths = len(path)
        if max_number is not None:
            max_number = min(max_number, num_paths)
            end_points = list(path_lens.keys())[-max_number:]
            paths = [path[point] for point in end_points]
        if max_number is None:
            end_points = list(path.keys())
            paths = list(path.values())
    # Vivd discrete color sequence
    colorlist =['rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)', 'rgb(165, 170, 153)']
    for num, trajectory in enumerate(paths):
        colornum = num % len(colorlist)
        trace = _trajectory_trace(trajectory, graph, figure, colorlist[colornum], num)
        fancy_figure.add_trace(trace)
    fancy_figure.add_trace(figure.data[1])
    fancy_figure.update_layout(showlegend=True)
    fancy_figure.update_traces(marker_showscale=False)

    return fancy_figure

def temporal_cycles(figure, graph, digraph=None, max_number=None, include_spacial=False):
    fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
    fancy_figure.add_trace(figure.data[0])
    assert type(graph) is igraph.Graph, 'graph must be original igraph.Graph object'
    assert digraph is None or type(digraph) is nx.classes.digraph.DiGraph, 'digraph must be networkx digraph or None'
    if digraph is None:
        digraph = temporal_subgraph(graph, include_spacial)

    cycles = find_cycles(digraph, max_number)
    # Vivd discrete color sequence
    colorlist =['rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)', 'rgb(165, 170, 153)']
    for num, cycle in enumerate(cycles):
        colornum = num % len(colorlist)
        trace = _cycle_trace(cycle, graph, figure, colorlist[colornum], num)
        fancy_figure.add_trace(trace)
    fancy_figure.add_trace(figure.data[1])
    fancy_figure.update_layout(showlegend=True)
    fancy_figure.update_traces(marker_showscale=False)

    return fancy_figure

def temporal_mapper(figure, graph, **kwargs):
    fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
    # Add the original edge data from figure
    fancy_figure.add_trace(figure.data[0])
    directed_edges, edge_directions = find_temporal_edges(graph)  
    forward = np.argwhere(edge_directions > 0).flatten()
    backward = np.argwhere(edge_directions < 0).flatten()
    double = np.argwhere(edge_directions == 0).flatten()
    for direction, color, name in [(forward, 'lightcoral','forward'), 
                      (backward, 'lightcoral','backward'),
                      (double, 'lightcoral', 'double')]:
        directed_trace = _edgecolor_trace(directed_edges, direction, figure, color, name)
        fancy_figure.add_trace(directed_trace)
    fancy_figure.add_trace(figure.data[1])
    return fancy_figure