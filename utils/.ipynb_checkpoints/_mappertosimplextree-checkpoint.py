import numpy as np
import igraph
import scipy
from gudhi import SimplexTree
from itertools import combinations

""" We need to align data between Mapper graphs in the filtration.
We do so by constructing a similarity matrix between nodes 
and then determines transition vectors between sets of nodes in consecutive mapper graphs.

Currently un-optimized for comupation time.
"""
def jaccard_index(n1, n2):
    # Compute the Jaccard Index between two nodes
    jaccard = 0
    intersection = np.intersect1d(n1['node_elements'], n2['node_elements'])
    if len(intersection) != 0:
        union = np.union1d(n1['node_elements'], n2['node_elements'])
        jaccard = len(intersection)/len(union)
    return jaccard

def _node_comparison_matrix(g1, g2):
    """ Construct the node comparison matrix between two
    2mappers in the multiscale 2mapper filtration.
    The matrix has size g1.num_nodes x g2.num_nodes
    and each entry is the jaccard index for each pair of nodes,
    so long as that pair comes from the same cover set.
    """
    g1_nodes = []
    g2_nodes = []
    data = []
    for node in g1.vs():
        node_coverset = node['pullback_set_label']
        n2_matching_coversets = g2.vs.select(pullback_set_label_eq = node_coverset)
        for node2 in n2_matching_coversets:
            entry = jaccard_index(node, node2)
            if entry > 0:
                g1_nodes.append(node.index)
                g2_nodes.append(node2.index)
                data.append(entry)
    comparison_matrix = scipy.sparse.coo_matrix(
        (data, (g1_nodes, g2_nodes)),
        (len(g1.vs), len(g2.vs))
    ).tocsr()
    return comparison_matrix

def _align_graphs(tm1, tm2):
    """ Align the two 2mapper graphs tm1 and tm2
    in the multiscale 2mapper filtration.
    To align, we construct the similarity matrix `_node_comparison_matrix`
    and for each node in tm1 we find its "best match" in tm2 by taking the
    maximum of entries in the matrix across the row.
    If a node in tm2 has multiple matches this means there is a collapse
    of nodes from tm1 to tm2. To align the simplex tree, we insert the 
    collapsed node from tm1 into tm2 and add edges and simplices to 
    retain the homotopy type of tm2.

    We return the node matching map from tm1 --> tm2,
    and new 2mapper in the form of the added simplices and graph.
    """
    TM1 = tm1[0].vs.select(partial_cluster_label_ne = -1)
    TM2 = tm2[0].vs.select(partial_cluster_label_ne = -1)
    G1 = tm1[0].induced_subgraph(TM1, implementation='copy_and_delete')
    G2 = tm2[0].induced_subgraph(TM2, implementation='copy_and_delete')
    node_match_matrix = _node_comparison_matrix(G1, G2)
    # Assert that there is at least one possible node matching
    # from tm1 --> tm2
    assert np.all(node_match_matrix.max(axis=1, explicit=True).toarray().flatten() != np.zeros((len(TM1),))), node_match_matrix.max(axis=1, explicit=True).toarray().flatten()
    # Assert that each node in tm1 only has one possible match in tm2
    assert node_match_matrix.max(axis=1, explicit=True).toarray().flatten().nonzero()[0].shape[0] == len(TM1)
    matches = np.asarray(node_match_matrix.argmax(axis=1, explicit=True)).flatten().reshape(len(TM1),)
    # boolean of all nodes in G1 which collapse in G2
    collapsed_node = np.full((len(TM1),), False, dtype='bool')
    # Label of where the node in G1 will collapse to
    # if entry is -1 then node (index) does not collapse
    dominant_nodes = np.full((len(TM1),), -1, dtype='int')
    new_edges = []
    g1_new_simplices = []
    unique_matches, indices, repeats = np.unique(matches, return_inverse=True, return_counts=True)
    # If there is a collapse of clusters
    if unique_matches.shape != matches.shape:
        tm1_adj = G1.get_adjacency_sparse()
        for i, count in enumerate(repeats):
            if count > 1:
                n2 = unique_matches[i]
                n2_pullback = G2.vs[n2]['pullback_set_label']
                n2_matches = np.argwhere(indices == i).flatten()
                node_size = np.array([len(TM1[n1]['node_elements']) for n1 in n2_matches])
                dominant_node = n2_matches[node_size.argmax()]
                # if there is a collapse of two clusters
                # the clusters in G1 which will collapse
                # require that G2 preserve that collapsed vertex
                # "other node"
                # and add edges and simplices to keep original 
                # homotopy type of G2
                for other_node in n2_matches:
                    pcl = 100          
                    if other_node != dominant_node:
                        # bookkeep for collapsed node
                        collapsed_node[other_node] = True
                        dominant_nodes[other_node] = dominant_node
                        # Add new vertex to G2
                        igraph.Graph.add_vertex(G2, name=None, 
                                                **{'pullback_set_label' : n2_pullback,
                                                 'partial_cluster_label' : pcl,
                                                 'node_elements' : G1.vs[other_node]['node_elements']
                                                }
                                               )
                        pcl += 1
                        # Change matches(node_map) so that other_node --> new_vertex
                        matches[other_node] = G2.vs[-1].index
                        # Add edge to G2 from collapsed vertex to its paired vertex
                        edge_data = np.intersect1d(G1.vs[dominant_node]['node_elements'], 
                                                   G1.vs[other_node]['node_elements']
                                                  )
                        edge_weight = edge_data.size
                        igraph.Graph.add_edge(G2, n2, G2.vs[-1].index,
                                              **{'weight' : edge_weight,
                                                 'edge_elements' : edge_data
                                                }
                                             )
                        other_node_adjacent_vertices = tm1_adj[other_node].indices
                        # For nodes that only connect with other_node, 
                        # there will be edges from 
                        # node ---> dominant node (created by collapse)
                        # node ---> other node (old edge in G1)
                        # other node ---> dominant node (just added)
                        # we want to then add simplex between
                        # node, other node and dominant node to retain homotopy
                        common_nodes = other_node_adjacent_vertices
                        for n in common_nodes:
                            # add simplices to G2 to retain homotopy type
                            G1simplex = [dominant_node, other_node, n]
                            G1simplex.sort()
                            g1_new_simplices.append(
                                G1simplex
                            )
                        if common_nodes.shape[0] > 1:
                            # For pairs of nodes adjecent to other_node
                            # We fill in the simplex to retain homotopy 
                            common_node_pairs = combinations(common_nodes, 2)
                            for pair in common_node_pairs:
                                if tm1_adj[pair[0], pair[1]] != 0:
                                    G1simplex = [other_node, pair[0], pair[1]]
                                    G1simplex.sort()
                                    g1_new_simplices.append(
                                        G1simplex
                                    )
        # Now we have to check for double collapses
        # If two clusters from different pullbacks
        # collapse at the same time then we have to 
        # add simplices into G2
        adjacent_nodes = np.vstack([tm1_adj.nonzero()[0], tm1_adj.nonzero()[1]]).T
        double_collapse = np.all(collapsed_node[adjacent_nodes], axis=1)
        for node_pair in adjacent_nodes[double_collapse]:
            s1 = [dominant_nodes[node_pair[0]],
                                     dominant_nodes[node_pair[1]],
                                     node_pair[0]
                                    ]
            s1.sort()
            s2 = [dominant_nodes[node_pair[0]],
                                     dominant_nodes[node_pair[1]],
                                     node_pair[1]
                                    ]
            s2.sort()
            g1_new_simplices.extend([s1, s2])
    # map added g1 simplices into g2
    g2_new_simplices = matches[g1_new_simplices]
    new_simplices = list(map(tuple, g2_new_simplices))
    return matches, new_simplices, G2

def _get_node_maps(two_mappers):
    # Create the node maps between all 
    # 2mappers in the filtration
    node_maps = []
    new_two_mappers = [two_mappers[0]]
    new_two_mapper = two_mappers[0]
    i = 1
    while i < len(two_mappers):
        node_map, new_simplices, G2 = _align_graphs(new_two_mapper, two_mappers[i])
        new_two_mapper = (G2, two_mappers[i][1] + new_simplices)
        new_two_mappers.append(new_two_mapper)
        node_maps.append(node_map)
        i += 1
    return node_maps, new_two_mappers

def _node_map_recursion(node_maps, k, tower_size):
    ''' Compose all node maps between 2mappers in the filtration.
    '''
    node_map = node_maps[k]
    step = k + 1
    while step < tower_size - 1:
        node_map = node_maps[step][node_map]
        step += 1
    return node_map

def _create_tree(graph, filter_time, node_maps):
    """ Initialize the 2mapper filtration by constructing a simplex tree.

        Parameters
        -----------
        graph : igraph.Graph
            The mapper graph we want to initialize as a simplex tree.

        filter_time : float
            The filter time for the initialized simplex tree.

        node_maps : list
            list of 'node maps'. Each node map (from 2-mapper g1 -> 2-mapper g2) 
            is an ndarray of shape (n1,), where n1 is the number of vertices in g1.

        Returns
        -----------
        gudhi.SimplexTree. 
            At each point in the filtration, the simplex tree object 
            is a 2-mapper graph. 
    
    """
    tree = SimplexTree()
    node_map = _node_map_recursion(node_maps, 0, len(node_maps)+1)
    vertex_array = node_map[np.arange(len(graph.vs))]
    for v in vertex_array:
        tree.insert(simplex=[v], filtration=filter_time)
    return tree

def _remove_duplicate_simplices(two_mappers, filter_times, node_maps):
    ''' Remove any duplicate simplices to be added to the simplex tree to 
    reduce computation time. Returns dictionaries whose (key, value) pairs
    are (filter_time, simplex) representing simplices which have yet to be
    added to the simplex tree.

    Parameters
    ----------
    two_mappers: list of tuples (mapper_graphs, two_simplices). Each tuple
        represents a single 2mapper graph; 
        mapper_graphs is a iGraph.graph object
        two_simplices is a list of tuples.
    
    filter_times: list of floats. 
        Filter values for each two_mapper graph in the cover tower.

    Returns
    --------
    edge_dict: dict
        The edge insertion dictionary.

    simplex_dict: dict
        The simplex insertion dictionary.
    '''
    tower_size = len(filter_times)
    edge_dict = {}
    simplex_dict = {}
    added_edges = []
    added_simplices = []
    for i,time in enumerate(filter_times):
        if i < tower_size - 1:
            # map node indices to match the last 2mapper in the filtration
            # using _node_map_recursion to compose the node_maps
            node_map = _node_map_recursion(node_maps, i, tower_size)
            simplices = list(
                map(tuple, 
                    node_map[two_mappers[i][1]]
                   )
            )
            edges = list(
                map(tuple, 
                    node_map[two_mappers[i][0].get_edgelist()]
                   )
            )
        if i == tower_size - 1:
            simplices = two_mappers[i][1]
            edges = two_mappers[i][0].get_edgelist()
        # remove edges and simplices which already exist
        new_edges = list(set(edges) - set(added_edges))
        new_simplices = list(set(simplices) - set(added_simplices))
        # add new edges and simplices into dicts
        if len(new_edges) != 0:
            edge_dict[time] = new_edges
            added_edges.extend(new_edges)
        if len(new_simplices) != 0:
            simplex_dict[time] = new_simplices
            added_simplices.extend(new_simplices)
    return edge_dict, simplex_dict

def _add_edges_simplices(tree, edges_dict, simplex_dict):
    ''' Add edges and simplices which have yet to be added into the simplex tree.

        Parameters
        ----------
        tree: SimplexTree. 
            The simplex tree we want to append the new simplices too.

        edge_dict: dict. 
            Keys are filtration times. Values are tuples of edges.
        
        simplex_dict: dict. 
            Keys are filtration times. Values are triples of two-simplices.
    
        Returns
        --------
        tree : gudhi.SimplexTree
    '''
    for filter_time in iter(edges_dict):
        edges = np.asarray(edges_dict[filter_time])
        tree.insert_batch(vertex_array = edges.T,
                          filtrations = np.full(edges.shape[0], filter_time, dtype = 'float')
                         )
        if filter_time in simplex_dict:
            simplices = np.asarray(simplex_dict[filter_time])
            tree.insert_batch(vertex_array = simplices.T,
                              filtrations = np.full(simplices.shape[0], filter_time, dtype = 'float')
                             )
            del simplex_dict[filter_time]
    for filter_time in iter(simplex_dict):
        simplices = np.asarray(simplex_dict[filter_time])
        tree.insert_batch(vertex_array = simplices.T,
                          filtrations = np.full(simplices.shape[0], filter_time, dtype = 'float')
                         )
    return tree