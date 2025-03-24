import numpy as np
import gudhi
from twomapper import list_2simplices, plot_2mapper
from mappertosimplextree import MapperToSimplexTree
from covertower import CubicalCoverTower, LatticeCoverTower
from gtda.mapper.pipeline import make_mapper_pipeline
from gtda.mapper.visualization import plot_static_mapper_graph
from gtda.utils.validation import validate_params
from gtda.utils.intervals import Interval
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.cluster import DBSCAN
from tqdm import tqdm

class Multiscale2Mapper(BaseEstimator, TransformerMixin):
    """ Compute the Multiscale2Mapper from a data set X.
    Choose mapper parameters `filter_func`, `cover_tower`,
    and DBSCAN parameters `DBSCAN_eps` and `DBSCAN_minpts`
    for the `gtda.mapper.pipeline.mapper_pipeline`.
    You can choose additional params using **kwargs, however
    `store_edge_elements` is automatically set to True in the 
    pipeline. 

    The pipeline fits then transforms a tower of covers with
    respect to data set X and creates a filtration of 
    2mapper graphs. We fit this filtration to a `gudhi.SimplexTree`
    using `mappertosimplextree` and compute the persistent betti
    numbers of the filtration.

    Currently the only clustering algorithm choice is fixed to 
    DBSCAN.

    Parameters
    -----------
    filter_func : object, callable or None.
        The filter function for the mapper pipeline.
        If `filter_func` is None the pipeline uses
        `sklearn.decomposition.PCA(n_componenets=2)`.

    cover_tower : object or None.
        The tower of covers for the mapper pipeline.
        If `cover_tower` is None the pipeline uses
        `covertower.CubicalCoverTower`.
        Choices for cover tower include:
        `covertower.CubicalCoverTower` and
        `covertower.LatticeCoverTower`.

    DBSCAN_eps : float, default `0.5`.
        The epsilon parameters for the clustering algorithm
        `sklearn.cluster.DBSCAN`.

    DBSCAN_minpts : int, default `2`.
        The minimum points for classifying a cluster in 
        `sklearn.cluster.DBSCAN`. 
        Note that if `DBSCAN_minpts` is greater than 2 then 
        *free border points* can occur causing possible 
        miscalculation for the multiscale 2mapper.
        We suggest fixing `DBSCAN_minpts` to 2 or 1.

    filtration : str, default ``'overlap_frac'``. 
        The filtration value for the multiscale 2mapper.
        Options are ``'overlap_frac'`` and ``'cover_set_volume'``.

    **kwargs : Additional parameters for the
        `gtda.mapper.pipeline.make_mapper_pipeline` object.
        This should not include `store_edge_elements` as a 
        key, as our pipeline fixes this argument to True.
    
    Attributes
    -----------
    mapper_pipeline : `gtda.mapper.pipeline.MapperPipeline`
        The mapper pipeline used to compute the filtration of 2mappers.
        The cover parameter is set to one of the covers in the cover tower.
        To change which cover is in the current pipeline use
        `self.mapper_pipeline.set_params(cover=self.cover_tower.cover_sets[tower_index])`
        for the tower_index of the requested cover.

    mappertosimplextree : `mappertosimplextree.MapperToSimplexTree`.
        The MapperToSimplexTree object computed when applying :meth:`fit_transform`.
        To access the simplex tree from this computation use
        `self.mappertosimplextree.simplex_tree`.
        From this object you can view all persistent betti numbers for the 
        multiscale 2mapper.
    """
    _hyperparameters = {'DBSCAN_eps' : {'type': (int, float), 'in': Interval(0, np.inf, closed='neither')},
                        'DBSCAN_minpts' : {'type': int, 'in': Interval(1, np.inf, closed='left')},
                        'cover_tower' : {'type': (CubicalCoverTower, LatticeCoverTower)},
                        'filtration' : {'type': str, 'in': ['overlap_frac', 'cover_set_volume']},
                       }

    def __init__(self, filter_func=None, cover_tower=None, DBSCAN_eps=0.5, DBSCAN_minpts=2, filtration='overlap_frac', **kwargs):
        if filter_func is None:
            from sklearn.decomposition import PCA
            self.filter_func = PCA(n_components=2)
        else:
            self.filter_func = filter_func
        if cover_tower is None:
            self.cover_tower = CubicalCoverTower()
        else:
            self.cover_tower = cover_tower
        self.DBSCAN_eps = DBSCAN_eps
        self.DBSCAN_minpts = DBSCAN_minpts
        DBSCAN_params = dict(eps=DBSCAN_eps, min_samples=DBSCAN_minpts)
        self.filtration = filtration
        self.mapper_pipeline = make_mapper_pipeline(filter_func=self.filter_func,
                                                    cover=self.cover_tower,
                                                    clusterer=DBSCAN(**DBSCAN_params),
                                                    store_edge_elements=True,
                                                    **kwargs)                                              

    def _fit_transform(self, X, plot, **plot_kwargs):
        is_overlap_frac = self.filtration == 'overlap_frac'
        self.cover_tower.fit_transform(self.filter_func.fit_transform(X))
        fitted_cover_sets = self.cover_tower.cover_sets
        if is_overlap_frac:
            filtration_values = self.cover_tower.cover_overlap_fracs
        else:
            filtration_values = self.cover_tower.cover_set_volumes
        two_mappers = []
        for cover in tqdm(fitted_cover_sets, desc='Cover Tower', total = self.cover_tower.tower_size):
            # fit_transform each cover in the tower of covers
            self.mapper_pipeline.set_params(cover = cover)
            graph = self.mapper_pipeline.fit_transform(X)
            simplices = list_2simplices(graph)
            two_mappers.append((graph,simplices))
            if plot:
                # plot each mapper graph in the filtration
                fig = plot_static_mapper_graph(pipeline=self.mapper_pipeline, 
                                         data=X,
                                         **plot_kwargs)
                fig.show()
        data = list(zip(two_mappers, filtration_values))
        print('----- Computing Betti Numbers -----')
        # compute the simplex tree and betti numbers
        # from the filtration of 2mappers
        self.mappertosimplextree = MapperToSimplexTree()
        bettis = self.mappertosimplextree.fit_transform(data)
        return bettis

    def fit_transform(self, X, y=None, plot=False, **plot_kwargs):
        """ Create the multiscale 2-mapper from a data set. 
        We fit the mapper pipeline for each cover in the cover tower,
        and then we align the filtration, construct a simplex tree,
        and then compute the persistent betti numbers.
        This method returns the betti numbers of the last 2mapper in
        the filtration. To access the persistent betti numbers use
        `self.mappertosimplextree.simplex_tree.persistent_betti_numbers()`.

        Additionally you can plot each mapper graph in the filtration using
        the `plot` parameter.

        Parameters
        -----------
        X : ndarray of shape(n_samples, n_features)

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        plot : bool, default ``False``.
            Plot the mapper graph for each filter value in the filtration.
            For viewing the 2mapper graph at each step in the filtration, 
            we suggest the use of :func:``plot_2mapper`` individually.

        plot_kwargs : dict, 
            Any additional parameters for plotting as in the method
            :func:``gtda.mapper.visualization.plot_static_mapper_graph``
            given in the form of a dictionary.

        Returns
        ----------
        bettis: list.
            The betti numbers of the last 2mapper in the filtration as in
            :meth:``mappertosimplextree.MapperToSimplexTree.fit_transform``.
        """
        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(deep=False), self._hyperparameters, exclude=['filter_func'])
        bettis = self._fit_transform(Xt, plot=plot, **plot_kwargs)
        return bettis

    def barcode(self):
        """ Return the persistence barcode of the 
        multiscale 2mapper.
        """
        is_overlap_frac = self.filtration == 'overlap_frac'
        if is_overlap_frac:
            x_title = 'Overlap Fraction g'
        else:
            x_title = 'Cover Set Volume'
        plot = gudhi.plot_persistence_barcode(self.mappertosimplextree.persistence)
        plot.set_xlabel(x_title)
        return plot

    def diagram(self):
        """ Return the persistence diagram of the
        multiscale 2mapper.
        """
        is_overlap_frac = self.filtration == 'overlap_frac'
        if is_overlap_frac:
            x_title = 'Overlap Fraction g'
        else:
            x_title = 'Cover Set Volume'
        plot = gudhi.plot_persistence_diagram(self.mappertosimplextree.persistence)
        plot.set_xlabel(x_title)
        return plot

    def plot_mapper(self, X, filtration_value=None, **graph_kwargs):
        """ Plot a mapper graph in the filtration.

        Parameters
        -----------

        filtration_value : float or None, default None.
            The filtration value of the mapper graph to be plotted.
            If None, the last mapper graph in the filtration is plotted.

        **graph_kwargs : dict
            Dictionary for additional plotting parameters. 
            Keys should be arguments for 
            :meth:`gtda.mapper.visualization.plot_static_mapper_graph`.

        Returns
        -----------
        figure : `plotly.graph_objects.Figure`
        """
        if not hasattr(self, 'mappertosimplextree'):
            raise AttributeError('Must apply fit_transform to Multiscale2Mapper object first.')
        is_overlap_frac = self.filtration == 'overlap_frac'
        if is_overlap_frac:
            if filtration_value is None:
                filtration_value = self.cover_tower.cover_overlap_fracs[-1]
            assert np.any(np.isclose(filtration_value, self.cover_tower.cover_overlap_fracs)),\
            f"filtration_value must be in overlap fracs {self.cover_tower.cover_overlap_fracs}. {filtration_value} was input."
            index = np.argwhere(
                np.isclose(filtration_value, self.cover_tower.cover_overlap_fracs)==np.full(self.cover_tower.tower_size, True)).flatten()[0] 
        else:
            if filtration_value is None:
                filtration_value = self.cover_tower.cover_set_volumes[-1]
            assert np.any(np.isclose(filtration_value, self.cover_tower.cover_set_volumes)),\
            f"filtration_value must be in cover set volumes {self.cover_tower.cover_set_volumes}. {filtration_value} was input."
            index = np.argwhere(
                np.isclose(filtration_value, self.cover_tower.cover_set_volumes)==np.full(self.cover_tower.tower_size, True)).flatten()[0]
        self.mapper_pipeline.set_params(cover=self.cover_tower.cover_sets[index])
        figure = plot_static_mapper_graph(pipeline=self.mapper_pipeline, data=X, **graph_kwargs)
        return figure

    def plot_2mapper(self, X, filtration_value=None, **graph_kwargs):
        """ Plot a 2mapper in the filtration.

        Parameters
        -----------

        filtration_value : float or None, default None.
            The filtration value of the 2mapper to be plotted.
            If None, the last mapper graph in the filtration is plotted.

        **graph_kwargs : dict
            Dictionary for additional plotting parameters. 
            Keys should be arguments for 
            :meth:`gtda.mapper.visualization.plot_static_mapper_graph`.

        Returns
        -----------
        figure : `plotly.graph_objects.Figure`
        """
        if not hasattr(self, 'mappertosimplextree'):
            raise AttributeError('Must apply fit_transform to Multiscale2Mapper object first.')
        is_overlap_frac = self.filtration == 'overlap_frac'
        if is_overlap_frac:
            if filtration_value is None:
                filtration_value = self.cover_tower.cover_overlap_fracs[-1]
            assert np.any(np.isclose(filtration_value, self.cover_tower.cover_overlap_fracs)),\
            f"filtration_value must be in overlap fracs {self.cover_tower.cover_overlap_fracs}. {filtration_value} was input."
            index = np.argwhere(
                np.isclose(filtration_value, self.cover_tower.cover_overlap_fracs)==np.full(self.cover_tower.tower_size, True)).flatten()[0]
        else:
            if filtration_value is None:
                filtration_value = self.cover_tower.cover_set_volumes[-1]
            assert np.any(np.isclose(filtration_value, self.cover_tower.cover_set_volumes)),\
            f"filtration_value must be in cover set volumes {self.cover_tower.cover_set_volumes}. {filtration_value} was input."
            index = np.argwhere(
                np.isclose(filtration_value, self.cover_tower.cover_set_volumes)==np.full(self.cover_tower.tower_size, True)).flatten()[0]
        self.mapper_pipeline.set_params(cover=self.cover_tower.cover_sets[index])
        if 'layout_dim' not in graph_kwargs:
            graph_kwargs['layout_dim'] = 3
        graph_kwargs['data'] = X
        graph_kwargs['pipeline'] = self.mapper_pipeline
        figure = plot_2mapper(graph_kwargs)
        return figure
