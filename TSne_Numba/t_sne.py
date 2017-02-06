import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import BallTree
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals.six import string_types
from sklearn.base import BaseEstimator
from sklearn.manifold import t_sne as SK_TSNE


class TSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding using NUMBA for GPU calculations

    This wraps the functionality of the t_sne function of the sklearn package
     around a _fit function that calls numba code to generate the required distances
     between samples and to sort them using the GPU. It is designed to be called in
     almost an identical way to the way the TSNE of sklearn would be (i.e. create
     a model with all the parameters and then call the the fit_transform(X) function
     passing the array to be t-sned).

     The extra parameters

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.

    early_exaggeration : float, optional (default: 4.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float, optional (default: 1000)
        The learning rate can be a critical parameter. It should be
        between 100 and 1000. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high. If the cost function gets stuck in a bad local
        minimum increasing the learning rate helps sometimes.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 200.

    n_iter_without_progress : int, optional (default: 30)
        Maximum number of iterations without progress before we abort the
        optimization.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, optional (default: 1E-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.

    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Note that different initializations
        might result in different local minima of the cost function.

    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.


    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = TSNE(n_components=2, random_state=0)
    >>> np.set_printoptions(suppress=True)
    >>> model.fit_transform(X) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[ 0.00017599,  0.00003993],
           [ 0.00009891,  0.00021913],
           [ 0.00018554, -0.00009357],
           [ 0.00009528, -0.00001407]])

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        http://homepage.tudelft.nl/19j49/t-SNE.html

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
    """

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000,
                 n_iter_without_progress=30, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        if not ((isinstance(init, string_types) and
                init in ["pca", "random"]) or
                isinstance(init, np.ndarray)):
            msg = "'init' must be 'pca', 'random', or a numpy array"
            raise ValueError(msg)
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.embedding_ = None
        self.TSNE = SK_TSNE.TSNE(n_components=self.n_components, perplexity=self.perplexity,
                 early_exaggeration=self.early_exaggeration, learning_rate=self.learning_rate, n_iter=self.n_iter,
                 n_iter_without_progress=self.n_iter_without_progress, min_grad_norm=self.min_grad_norm,
                 metric=self.metric, init=self.init, verbose=self.verbose,
                 random_state=self.random_state, method=self.method, angle=self.angle)


    def _fit(self, X, skip_num_points=0):
        """Fit the model using X as training data.

        Note that sparse arrays can only be handled by method='exact'.
        It is recommended that you convert your sparse array to dense
        (e.g. `X.toarray()`) if it fits in memory, or otherwise using a
        dimensionality reduction technique (e.g. TruncatedSVD).

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. Note that this
            when method='barnes_hut', X cannot be a sparse array and if need be
            will be converted to a 32 bit float array. Method='exact' allows
            sparse arrays and 64bit floating point inputs.

        skip_num_points : int (optional, default:0)
            This does not compute the gradient for points with indices below
            `skip_num_points`. This is useful when computing transforms of new
            data where you'd like to keep the old data fixed.
        """
        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.method == 'barnes_hut' and sp.issparse(X):
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required for method="barnes_hut". Use '
                            'X.toarray() to convert to a dense numpy array if '
                            'the array is small enough for it to fit in '
                            'memory. Otherwise consider dimensionality '
                            'reduction techniques (e.g. TruncatedSVD)')
        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=np.float64)
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is "
                             "%f" % self.early_exaggeration)

        if self.n_iter < 200:
            raise ValueError("n_iter should be at least 200")

        if self.metric == "precomputed":
            if isinstance(self.init, string_types) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be used "
                                 "with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            distances = X
        else:
            if self.verbose:
                print("[t-SNE] Computing pairwise distances...")

            if self.metric == "euclidean":
                distances = pairwise_distances(X, metric=self.metric,
                                               squared=True)
            else:
                distances = pairwise_distances(X, metric=self.metric)

        if not np.all(distances >= 0):
            raise ValueError("All distances should be positive, either "
                             "the metric or precomputed distances given "
                             "as X are not correct")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1.0, 1)
        n_samples = X.shape[0]
        # the number of nearest neighbors to find
        k = min(n_samples - 1, int(3. * self.perplexity + 1))

        neighbors_nn = None
        if self.method == 'barnes_hut':
            if self.verbose:
                print("[t-SNE] Computing %i nearest neighbors..." % k)
            if self.metric == 'precomputed':
                # Use the precomputed distances to find
                # the k nearest neighbors and their distances
                neighbors_nn = np.argsort(distances, axis=1)[:, :k]
            else:
                # Find the nearest neighbors for every point
                bt = BallTree(X)
                # LvdM uses 3 * perplexity as the number of neighbors
                # And we add one to not count the data point itself
                # In the event that we have very small # of points
                # set the neighbors to n - 1
                distances_nn, neighbors_nn = bt.query(X, k=k + 1)
                neighbors_nn = neighbors_nn[:, 1:]
            P = TSNE._joint_probabilities_nn(distances, neighbors_nn,
                                        self.perplexity, self.verbose)
        else:
            P = TSNE._joint_probabilities(distances, self.perplexity, self.verbose)
        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be zero or positive"
        assert np.all(P <= 1), ("All probabilities should be less "
                                "or then equal to one")

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X)
        elif self.init == 'random':
            X_embedded = None
        else:
            raise ValueError("Unsupported initialization scheme: %s"
                             % self.init)

        return self.TSNE._tsne(P, degrees_of_freedom, n_samples, random_state,
                               X_embedded=X_embedded,
                               neighbors=neighbors_nn,
                               skip_num_points=skip_num_points)


    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """
        self.fit_transform(X)
        return self