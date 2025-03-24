import numpy as np
from gudhi import SimplexTree
from gtda.utils.validation import validate_params
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from utils._mappertosimplextree import _create_tree, \
    _remove_duplicate_simplices, _add_edges_simplices, _get_node_maps
from twomapper import list_2simplices


class MapperToSimplexTree(BaseEstimator, TransformerMixin):
    """ Construct Multiscale Mapper Barcode from a SimplexTree.
    In :meth:`fit`, given set `X` representing a collection of pairs
    (2Mapper graphs, filtration value), a filtration of Simplex Trees is constructed.
    In :meth:`transform` peristence is computed on the filtration 
    and outputs the peristence diagram/barcode of the filtration using gudhi.  

    Parameters
    ----------
    

    Attributes
    ----------
    simplex_tree: `gudhi.SimplexTree` object computed from 
        a sequence of 2mapper graphs.

    two_mappers: list
        list of 2mappers in the simplex tree.
        Each 2mapper is a tuple (`igraph.Graph`, list)
        where the first entry is the mapper graph and the second
        is the list of 2simplices found via meth::twomapper.list_2simplices.

    filtration_values: list
        list of filtration values of the simplex tree.

    persistence: list
        list of pairs (dimension, (birth, death))
    """

    _hyperparameters = {} # maybe add njobs?

    def __init__(self):
        self.simplex_tree = SimplexTree()

    def _fit(self, X):
        self.two_mappers = [X[k][0] for k in range(len(X))]
        self.filtration_values = [X[k][1] for k in range(len(X))]
        self.simplex_tree = self._to_simplextree(self.simplex_tree,
                                            self.two_mappers,
                                            self.filtration_values,
                                           )
        return self
        
    def fit(self, X, y=None):
        """Construct a filtration of Simplex Trees from given list of 2mapper
        graphs with filtration values. 
        Store the simplex tree in :attr:`simplex_tree`.

        Parameters
        ----------
        X : list of tuples (2Mapper graphs, filtration value)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        X = check_array(X)
        return self._fit(X)

    def _transform(self, X):
        return self.simplex_tree.persistence()
        
    def fit_transform(self, X):
        """ Construct a filtration of Simplex Trees from given list of 2mappers
        with filtration values. Store this simplex tree in :attr:`simplex_tree`
        and its filtration values in :`filtration_values`.
        Then, compute the persistence of fitted the simplex tree and 
        compute its betti numbers across the filtration and store them 
        in :attr:`persistence`.
        Return the betti numbers of the last 2mapper in the filtration.

        Parameters
        ----------
        X : list of tuples (2Mapper graphs, filtration value)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Bettis : list
            The betti numbers of the the last 2mapper in the filtration.
        """
        validate_params(self.get_params(), self._hyperparameters)
        check_is_fitted(self, 'simplex_tree')
        self.persistence = self._fit(X)._transform(X)
        bettis = self.simplex_tree.persistent_betti_numbers(int(len(X)),int(len(X)))
        
        return bettis

    @staticmethod
    def _to_simplextree(tree, two_mappers, filter_times):
        ''' Compute the simplex tree. '''
        # Start with an empty tree
        if not tree.is_empty():
            tree = SimplexTree()
        assert tree.is_empty()
        def _remove_noise(two_mappers):
            new_mappers = []
            for k in range(len(two_mappers)):
                graph = two_mappers[k][0]
                # simplices = two_mappers[k][1]
                # simplex_array = np.asarray(simplices)
                noiseless_graph = graph.induced_subgraph(graph.vs.select(partial_cluster_label_ne = -1))
                new_simplices = list_2simplices(noiseless_graph)
                new_mappers.append((noiseless_graph, new_simplices))
            return new_mappers
        two_mappers = _remove_noise(two_mappers)
        node_maps, new_two_mappers = _get_node_maps(two_mappers)
        initial_mapper = two_mappers[0]
        tree = _create_tree(initial_mapper[0], filter_times[0], node_maps)
        edge_dict, simplex_dict = _remove_duplicate_simplices(new_two_mappers, filter_times, node_maps)
        tree = _add_edges_simplices(tree, edge_dict, simplex_dict)
        return tree