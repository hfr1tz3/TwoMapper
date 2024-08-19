import numpy as np
import plotly
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
    directed_edges = np.full(num_edges, len(graph.es), dtype=bool)
    edge_elements = 'edge_elements' in graph.es.attributes()
    for edge in graph.es():
        source_data = graph.vs[edge.source]['node_elements']
        target_data = graph.vs[edge.target]['node_elements']
        if edge_elements:
            edge['edge_elements'].sort()
            intersection_data = edge['edge_elements']
            
        if not edge_elements:
            intersection_data = np.intersect1d(source_data, target_data)
        # if two consecutive points are in an edge
        # their difference should be equal to 1
        if len(intersection_data) > 1:
            e = intersection_data
            e.sort()
            if np.any(np.diff(e) == 1):
                directed_edges[edge.index] = True
                continue
        else:
        # Data point in the intersection of 2 nodes.
        # check for difference of 1 in source or target node
        # Then there will be a time witness
            for point in intersection_data:
                before = point - 1
                after = point + 1
                if before in source_data or after in source_data:
                    directed_edges[edge.index] = True
                    continue
                if before in target_data or after in target_data:
                    directed_edges[edge.index] = True
                    continue
    return directed_edges

def _edgecolor_trace(edge_mask, figure):
    directed_edges = np.flatnonzero(edge_mask)
    f = dict()
    f['hoverinfo'] = 'none'
    f['mode'] = 'lines'
    f['line'] = {'color': 'crimson'} 
    f['name'] = 'Time Witness'
    # f['legendrank'] = 1500
    f['x'] = list()
    f['y'] = list()
    f['z'] = list()
    for edge in directed_edges:
        f['x'].extend(figure.data[0]['x'][3*edge:3*edge+3])
        f['y'].extend(figure.data[0]['y'][3*edge:3*edge+3])
        f['z'].extend(figure.data[0]['z'][3*edge:3*edge+3])
    return plotly.graph_objects.Scatter3d(f)
    

def temporal_mapper(figure, graph, **kwargs):
    directed_edges = find_temporal_edges(graph)
    directed_trace = _edgecolor_trace(directed_edges, figure)
    fancy_figure = plotly.graph_objects.FigureWidget(layout = figure.layout)
    # Add the original edge data from figure
    fancy_figure.add_trace(figure.data[0])
    fancy_figure.add_trace(directed_trace)
    fancy_figure.add_trace(figure.data[1])
    return fancy_figure