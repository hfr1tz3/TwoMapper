import igraph
import numpy as np
import pytest
import mappertosimplextree as ms
from gudhi import SimplexTree

''' 
Test file for mappertosimplextree.py
Tests are to assure the retension of homotopy
when computing multiscale mapper across a filtration.
'''

def graph_to_simplex_tree(two_mapper):
    tree = SimplexTree()
    graph = two_mapper[0][0]
    simplices = two_mapper[0][1]
    vertex_array = np.arange(len(graph.vs))
    edge_array = np.asarray(graph.get_edgelist())
    simplex_array = np.asarray(simplices)
    for v in vertex_array:
        tree.insert(simplex=[v])
    tree.insert_batch(vertex_array = edge_array.T, filtrations = np.zeros(edge_array.shape[0]))
    if len(simplices) != 0:
        tree.insert_batch(vertex_array = simplex_array.T,
                          filtrations = np.zeros(simplex_array.shape[0])
                          )
    return tree

class TestMapperToSimplexTree:
    def verify_bettis(self, multimapper):
        m = ms.MapperToSimplexTree()
        m.fit_transform(multimapper)
        for two_mapper in multimapper:
            fil = two_mapper[1]
            tree = graph_to_simplex_tree(two_mapper)
            tree.compute_persistence()
            true_bettis = tree.persistent_betti_numbers(0,0)
            test_bettis = m.simplex_tree.persistent_betti_numbers(fil, fil)
            if len(true_bettis) == len(test_bettis):
                assert np.all(np.isclose(test_bettis, true_bettis))
            else:
                top_term_dim = len(true_bettis) - 1
                for dim,betti in enumerate(test_bettis):
                    if dim > top_term_dim:
                        assert betti == 0, print('Dim', dim, 'Test =/= True', betti, 0)
                    else:
                        assert betti == true_bettis[dim], print('Dim', dim, 'Test =/= True', betti, true_bettis[dim])
            
            
    def verify_last_bettis(self, multimapper):
        terminal_mapper = multimapper[-1]
        terminal_tree = graph_to_simplex_tree(terminal_mapper)
        terminal_tree.compute_persistence()
        terminal_bettis = terminal_tree.persistent_betti_numbers(0,0)
        m = ms.MapperToSimplexTree()
        multi_bettis = m.fit_transform(multimapper)
        if len(terminal_bettis) == len(multi_bettis):
            assert np.all(np.isclose(multi_bettis, terminal_bettis))
        else:
            top_term_dim = len(terminal_bettis) - 1
            for dim,betti in enumerate(multi_bettis):
                if dim > top_term_dim:
                    assert betti == 0
                else:
                    assert betti == terminal_bettis[dim]
        
    def get_ex(self, initial=True):
    #   0   2          0
    #   |       --->   |
    #   1              1 
        if initial:
            n_vertices = 3
            edges = [(0,1)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [0, 1, 1]
            ex.vs['partial_cluster_label'] = [0, 0, 1]
            ex.vs['node_elements'] = [np.array([0,1]), np.array([1,2]), np.array([3,4])]
        if initial is False:
            n_vertices = 2
            edges = [(0,1)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [1, 0]
            ex.vs['partial_cluster_label'] = [0, 0]
            ex.vs['node_elements'] = [np.array([1, 2, 3, 4]), np.array([0,1])]
        return ex

    def test_basic_ex(self):
        initial_graph = self.get_ex()
        terminal_graph = self.get_ex(initial=False)
        X = [((initial_graph, []), 0), ((terminal_graph, []), 1)]
        m = ms.MapperToSimplexTree()
        m.fit_transform(X)
        assert m.simplex_tree.filtration([2]) == 0, print([v for v in m.simplex_tree.get_skeleton(1)])
        assert m.simplex_tree.find([0,2]), print([v for v in m.simplex_tree.get_skeleton(1)])
        assert m.simplex_tree.filtration([0,2]) == 1
        self.verify_last_bettis(X)
        self.verify_bettis(X)

    def get_double_ex(self, initial=True):
        # 2   3        0
        # |   |        |
        # |   | -----> |
        # |   |        |
        # 0   1        1
        if initial:
            n_vertices = 4
            edges = [(0,2),(1,3)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [0,0,1,1]
            ex.vs['partial_cluster_label'] = [0,1,0,1]
            ex.vs['node_elements'] = [np.array([0,1]), np.array([2,3]), 
                                      np.array([0,4]), np.array([3,5])]
        if initial is False:
            n_vertices = 2
            edges = [(0,1)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [1,0]
            ex.vs['partial_cluster_label'] = [0,0]
            ex.vs['node_elements'] = [np.array([0,4,3,5]), np.array([0,1,2,3])]
        return ex

    def test_double_collapse(self):
        initial_graph = self.get_double_ex()
        terminal_graph = self.get_double_ex(initial=False)
        X = [((initial_graph, []),0), ((terminal_graph, []),1)]
        m = ms.MapperToSimplexTree()
        m.fit_transform(X)
        two_simplices = [[0,1,2], [0,1,3], [1,2,3], [0,2,3]]
        assert m.simplex_tree.find([0,2]), print([v for v in m.simplex_tree.get_skeleton(1)])
        assert m.simplex_tree.find([1,3]), print([v for v in m.simplex_tree.get_skeleton(1)])
        for simplex in two_simplices:
            assert m.simplex_tree.find(simplex), print([v for v in m.simplex_tree.get_skeleton(2)])
        self.verify_last_bettis(X)
        self.verify_bettis(X)

    def get_simplex_ex(self, initial=True):
        #    0              0
        #   /\             /
        #  /  \   ----->  /
        # 1    2         1
        if initial:
            n_vertices = 3
            edges = [(0,1), (0,2)]
            simplex = []
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [2,3,3]
            ex.vs['partial_cluster_label'] = [0,0,1]
            ex.vs['node_elements'] = [np.array([1,2,3,4]),
                                      np.array([1,2,5]), np.array([3,8,9])
                                     ]
        if initial is False:
            n_vertices = 2
            edges = [(0,1)]
            simplex = []
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [2,3]
            ex.vs['partial_cluster_label'] = [0,0]
            ex.vs['node_elements'] = [np.array([1,2,3,4,7,8,0]),
                                      np.array([1,2,5,3,8,9,10]),
                                     ]
        return ex, simplex

    def test_single_simplex(self):
        g1, s1 = self.get_simplex_ex()
        g2, s2 = self.get_simplex_ex(initial=False)
        X = [((g1, s1), 0), ((g2, s2), 1)]
        m = ms.MapperToSimplexTree()
        m.fit_transform(X)
        two_simplices = [0,1,2]
        assert m.simplex_tree.find(two_simplices), print([v for v in m.simplex_tree.get_skeleton(2)])
        self.verify_bettis(X)

    def get_long_ex(self, index=0):
        #  0  1           0           0
        #  |\ |          /\           |
        #  | \| ----->  /  \   -----> |
        #  3  2        1    2         1
        if index == 0:
            n_vertices=4
            edges = [(0,3), (0,2), (1,2)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [0,0,1,1]
            ex.vs['partial_cluster_label'] = [0,1,0,1]
            ex.vs['node_elements'] = [np.array([0,1,2]), np.array([3,4]),
                                      np.array([1,2,3]), np.array([0,5,6])]
        if index == 1:
            n_vertices = 3
            edges = [(0,1), (0,2)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [0,1,1]
            ex.vs['partial_cluster_label'] = [0,1,0]
            ex.vs['node_elements'] = [np.array([0,1,2,3,4]),
                                      np.array([0,5,6]), np.array([1,2,3,7])
                                     ]
        if index == 2:
            n_vertices = 2
            edges = [(0,1)]
            ex = igraph.Graph(n_vertices, edges)
            ex.vs['pullback_set_label'] = [0,1]
            ex.vs['partial_cluster_label'] = [0,0]
            ex.vs['node_elements'] = [np.array([0,1,2,3,4]), np.array([0,1,2,3,5,6,7,10])]
        return ((ex, []), index)

    def test_long_ex(self):
        X = [self.get_long_ex(index=i) for i in range(3)]
        m = ms.MapperToSimplexTree()
        m.fit_transform(X)
        two_simplices = [[0,1,3], [0,1,2]]
        for simplex in two_simplices:
            assert m.simplex_tree.find(simplex), print([v for v in m.simplex_tree.get_skeleton(2)])
        self.verify_bettis(X)