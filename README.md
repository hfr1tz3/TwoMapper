# 2-Mapper

The mapper algorithm is an unsupervised clustering algorithm used for used for data visualization in topological data analysis (TDA) \[1\]. 
It uses a range of parameters for constructing a graph from data chosen by the user. For a data set $X$,
to construct a mapper representation of $X$ one must choose the following parameters.

1. A filter (lens) function $f:X\to Z$ where $Z$ is a lower dimensional space.
2. A cover for $Z$. This traditionally is a uniform cover of overlaping cubes.
3. A clustering procedure, for example, DBSCAN \[2\].

(insert image)

The mapper graph is then the 1-dimensional nerve of the collection of clusters given after the clustering procedure.
This gives a good description of graphical connections in our data, but cannot tell us anything about topological 
features beyond dimension zero.
For filter functions that map to metric spaces of dimension $m\geq 2$, 
we can construct a mapper graph from the 2-dimsnional nerve, called 2-Mapper.

(insert image)

2-Mapper is a simplical complex approximation of $X$. This can give visualization to density changes within a point cloud, 
and allow use to compute some approximate betti-1 values.


References
----------
\[1\] Gurjeet Singh, Facundo M´emoli, and Gunnar E. Carlsson. Topological meth-
ods for the analysis of high dimensional data sets and 3d object recognition.
In Mario Botsch, Renato Pajarola, Baoquan Chen, and Matthias Zwicker,
editors, 4th Symposium on Point Based Graphics, PBG@Eurographics 2007,
Prague, Czech Republic, September 2-3, 2007, pages 91–100. Eurographics
Association, 2007.

\[2\] Martin Ester, Hans-Peter Kriegel, J¨org Sander, and Xiaowei Xu. A density-
based algorithm for discovering clusters in large spatial databases with noise.
In Proceedings of the Second International Conference on Knowledge Dis-
covery and Data Mining, KDD’96, page 226–231. AAAI Press, 1996.
