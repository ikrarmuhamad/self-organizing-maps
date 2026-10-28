import numpy as np
import pandas as pd
from typing import Optional, Literal
import torch
import math
from sklearn.preprocessing import StandardScaler 

class SelfOrganizingMaps :
    """Self-Organizing Map (SOM) trained with online (one sample per step) learning.

    Neurons are laid out on a 2D rectangular grid of ``x_neuron * y_neuron`` units.
    At every iteration a random sample is drawn, its Best Matching Unit (BMU) is
    found, and every neuron weight is pulled toward the sample by

        w_j(t + 1) = w_j(t) + alpha(t) * h_cj(t) * (x - w_j(t))

    where ``alpha(t)`` is the decaying learning rate and ``h_cj(t)`` is a Gaussian
    neighborhood centered on the BMU whose width ``sigma(t)`` also decays.

    Parameters
    ----------
    data : array-like
        Training data. A DataFrame, array or tensor of shape ``(n_samples, n_features)``.
        1D input is treated as a single feature, and input with more than 2 dimensions
        (e.g. images) is flattened to ``(n_samples, -1)``. Must not contain missing
        or infinite values. Features are standardized when ``normalize=True``.
    x_neuron : int, default=10
        Number of grid rows.
    y_neuron : int, default=10
        Number of grid columns. ``x_neuron * y_neuron`` must be at least 2.
    n_iter : int, default=2000
        Number of training iterations (one random sample per iteration).
    learning_rate : float, default=0.5
        Initial learning rate ``alpha_0``. Must be greater than ``lr_final``.
    lr_final : float, default=0.01
        Learning rate reached at the last iteration.
    sigma : float, optional
        Initial neighborhood radius ``sigma_0``. If None, it is set to
        ``max(x_neuron, y_neuron) / 3``. Must be greater than ``sigma_f``.
    sigma_f : float, optional
        Neighborhood radius reached at the last iteration. If None, it is set to
        ``min(1.0, sigma / 2)``.
    grid_shape : {'Hexagonal', 'Rectangle'}, default='Rectangle'
        Grid topology. Stored, but only the rectangular grid is implemented.
    decay_function : {'exponential', 'linear', 'geometric'}, default='exponential'
        Decay schedule. Stored, but only the exponential decay is implemented.
    neighborhood_function : {'gaussian', 'bubble'}, default='gaussian'
        Neighborhood kernel. Stored, but only the Gaussian kernel is implemented.
    weight_init : str, default='random'
        Weight initialization. Stored, but weights are always drawn uniformly
        between the minimum and maximum of each feature.
    d_type : {'EuclideanDistance', 'ManhattanDistance', 'CosineDistance'}, default='EuclideanDistance'
        Distance metric. Stored, but only the Euclidean distance is implemented.
    random_state : int, default=42
        Non-negative seed of the torch generator used for weight initialization and
        sampling. ``fit_train`` resets the generator to this seed, so repeated calls
        give the same result.
    normalize : bool, default=True
        Standardize every feature (zero mean, unit variance) with a ``StandardScaler``
        fitted once on the training data. Use False if the data is already scaled.
        Distances, ``quantization_err`` and ``final_weight`` are expressed in the
        scaled units.

    Attributes
    ----------
    data : torch.Tensor
        Training data after processing and scaling, shape ``(n_samples, n_features)``,
        dtype float32. It is already scaled, so do not pass it to ``predict``.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on the training data. ``predict`` applies it to new data, and
        ``scaler.inverse_transform`` converts weights back to the original units.
    normalize : bool
        Whether scaling is active.
    random_state : int
        Seed used by ``fit_train``.
    D_grid : torch.Tensor
        Pairwise Euclidean distances between neuron positions on the grid,
        shape ``(total_neuron, total_neuron)``.
    r_grid : torch.Tensor
        Grid coordinates of every neuron, shape ``(total_neuron, 2)``.
    total_neuron : int
        ``x_neuron * y_neuron``.
    tau, lam : float
        Time constants of the learning rate and sigma decay.
    final_weight : torch.Tensor
        Trained weights, shape ``(total_neuron, n_features)``. Set by ``fit_train``.
    quantization_err, topographic_err : list of torch.Tensor
        Errors recorded every ``steps`` iterations. Set by ``fit_train``.
    iter_steps : list of int
        Iteration indices (0-based) at which the errors were recorded.
    heat_map : torch.Tensor
        Hit counts per neuron, shape ``(x_neuron, y_neuron)``. Set by ``heat_maps``.

    Examples
    --------
    >>> som = SelfOrganizingMaps(data=X, x_neuron=5, y_neuron=5, n_iter=1000)
    >>> som.fit_train()
    >>> bmu, index_bmu, distance = som.predict()        # training data
    >>> bmu, index_bmu, distance = som.predict(X_new)   # new data, original units
    >>> som.heat_maps()
    >>> som.u_matrix()

    ``predict`` expects data in the original units and scales it with the scaler fitted
    on the training data. Do not pass ``som.data``, which is already scaled.

    Neuron ``k`` sits at grid row ``k // y_neuron`` and column ``k % y_neuron``.
    """

    def __init__(self,
                 data,
                 x_neuron : int = 10,
                 y_neuron : int = 10,
                 n_iter : int = 2000,
                 learning_rate : float = 0.5,
                 lr_final : float = 0.01,
                 sigma : Optional[float] = None,
                 sigma_f : Optional[float] = None,
                 grid_shape : Literal['Hexagonal', 'Rectangle'] = 'Rectangle',
                 decay_function : Literal['exponential', 'linear', 'geometric'] = 'exponential',
                 neighborhood_function : Literal['gaussian', 'bubble'] = 'gaussian',
                 weight_init : str = 'random',
                 d_type : Literal['EuclideanDistance', 'ManhattanDistance', 'CosineDistance'] = 'EuclideanDistance',
                 random_state : int = 42,
                 normalize : bool = True
                 ):

        self.validate_type('x_neuron', x_neuron)
        self.validate_type('y_neuron', y_neuron)
        self.validate_number('x_neuron', x_neuron)
        self.validate_number('y_neuron', y_neuron)
        if x_neuron * y_neuron < 2 :
            raise ValueError(f'Grid size {x_neuron*y_neuron}, grid size should >= 2')
        self.validate_type('n_iter', n_iter)
        self.validate_type('random_state', random_state)
        self.checker_missing_value(data)
        if sigma is None:
            sigma = max(x_neuron, y_neuron) / 3
        self.validate_number('sigma', sigma)
        if sigma_f is None:
            sigma_f = min(1.0, sigma / 2)
        self.validate_number('sigma_f', sigma_f)
        self.validate_number('learning_rate', learning_rate)
        self.validate_number('lr_final', lr_final)
        self.validate_param('learning_rate', 'lr_final', learning_rate, lr_final)
        self.validate_param('sigma', 'sigma_f', sigma, sigma_f)
        self.validate_number('n_iter', n_iter)
        self.validate_type('random_state', random_state)
        if random_state < 0:
            raise ValueError("random_state should be a non-negative integer")
        if not isinstance(normalize, bool):
            raise ValueError("normalize should be boolean")

        
        data = self.processing_data(data)
        self.normalize = normalize
        self.scaler = StandardScaler(with_mean=normalize, with_std=normalize)
        self.data = torch.from_numpy(self.scaler.fit_transform(data.numpy())).float()
        self.x_neuron = x_neuron
        self.y_neuron = y_neuron
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.lr_final = lr_final
        self.sigma = sigma
        self.sigma_f = sigma_f
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.weight_init = weight_init
        self.d_type = d_type
        self.grid_shape = grid_shape
        self.random_state = random_state
        self.generator = torch.Generator().manual_seed(random_state)
        self.D_grid = self._grid_distance()
        self.tau, self.lam = self._find_tau_lambda()
        self.total_neuron = self.x_neuron * self.y_neuron

    @staticmethod
    def validate_param(init, final, val_init, val_final):
        """Check that an initial value is strictly greater than its final value.

        Parameters
        ----------
        init, final : str
            Names of the parameters, used in the error message.
        val_init, val_final : float
            Values to compare.

        Raises
        ------
        ValueError
            If ``val_init <= val_final``.
        """
        if val_init <= val_final:
            raise ValueError(f'Parameter {init} more bigger than {final}. Choose {init} more lower than {final}')

    @staticmethod
    def validate_type(param, arg):
        """Check that an argument is an integer (booleans are rejected).

        Parameters
        ----------
        param : str
            Name of the parameter, used in the error message.
        arg : object
            Value to check.

        Raises
        ------
        ValueError
            If ``arg`` is not an ``int`` or is a ``bool``.
        """
        if not isinstance(arg, int) or isinstance(arg, bool):
            raise ValueError(f"{param} should be integer")

    @staticmethod
    def checker_missing_value(data):
        """Raise an error if the data contains any missing value (None or NaN).

        Parameters
        ----------
        data : array-like
            Data to check.

        Raises
        ------
        ValueError
            If a missing value is found. The message gives the number of missing
            values and their ``[row, column]`` positions.
        """
        null_data = pd.isna(np.asarray(data, dtype=object))
        if null_data.any():
            index = np.argwhere(null_data).tolist()
            raise ValueError(f"There's {int(null_data.sum())} missing value for index : {index}")

    @staticmethod
    def validate_number(param, arg):
        """Check that an argument is a positive number (booleans are rejected).

        Parameters
        ----------
        param : str
            Name of the parameter, used in the error message.
        arg : object
            Value to check.

        Raises
        ------
        ValueError
            If ``arg`` is not an ``int`` or ``float``, or is not greater than 0.
        """
        if isinstance(arg, bool) or not isinstance(arg, (int, float)) or arg <= 0:
            raise ValueError(f"{param} should be a positive number")

    @staticmethod
    def processing_data(data):
        """Convert input data into a float32 tensor of shape ``(n_samples, n_features)``.

        1D data becomes a single column, and data with more than 2 dimensions
        (e.g. ``(n, height, width)`` images) is flattened per sample.

        Parameters
        ----------
        data : array-like
            DataFrame, array or tensor. The first axis is the sample axis.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_samples, n_features)``.

        Raises
        ------
        ValueError
            If ``data`` is a scalar.
        """
        data = np.asarray(data, dtype=float).copy()
        if data.ndim == 0:
            raise ValueError("data should be at least 1D array")
        if data.ndim == 1:
            data = data.reshape(-1,1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        
        return torch.from_numpy(data).float()

    def get_random_sample(self):
        """Draw one random training sample using the model's generator.

        Returns
        -------
        torch.Tensor
            Sample of shape ``(1, n_features)``.
        """
        index_data = torch.randint(0, len(self.data), (1,), generator=self.generator)
        sample_data = self.data[index_data]
        return sample_data

    def _grid_distance(self):
        """Compute the Euclidean distance between every pair of neurons on the grid.

        Neuron ``k`` is placed at ``(k // y_neuron, k % y_neuron)``. The coordinates
        are also stored in ``self.r_grid``.

        Returns
        -------
        torch.Tensor
            Distance matrix of shape ``(total_neuron, total_neuron)``.
        """
        r_grid = torch.cartesian_prod(torch.arange(self.x_neuron), torch.arange(self.y_neuron)).float() # Position Grid
        D_grid = torch.cdist(r_grid, r_grid, p=2) # Euclidean Distance
        self.r_grid = r_grid
        return D_grid

    def _initialize_weight_neuron(self):
        """Initialize neuron weights uniformly between the min and max of each feature.

        Returns
        -------
        W_init : torch.Tensor
            Initial weights, shape ``(total_neuron, n_features)``.
        n_features : int
            Number of features in the data.
        """
        high = self.data.max(dim=0).values
        low = self.data.min(dim=0).values
        n_features = self.data.shape[1]
        W_init = low + (high - low) * torch.rand(self.total_neuron, n_features, generator=self.generator) 

        return W_init, n_features

    def _find_bmu(self, weight, data, k = 1):
        """Find the ``Top k`` Best Matching Units (closest neurons) of each sample.

        Parameters
        ----------
        weight : torch.Tensor
            Neuron weights, shape ``(total_neuron, n_features)``.
        data : torch.Tensor
            One or more samples, shape ``(n_samples, n_features)``.
        k : int, default=1
            Number of closest neurons to return. Must not exceed ``total_neuron``.

        Returns
        -------
        bmu : torch.Tensor
            Weights of the BMUs, shape ``(n_samples, k, n_features)``.
        index_bmu : torch.Tensor
            Neuron indices of the BMUs, shape ``(n_samples, k)``.
        lowest_d : torch.Tensor
            Euclidean distances to the BMUs, shape ``(n_samples, k)``.
        """
        distance = torch.cdist(data, weight)  #Euclidean Distance
        top_bmu = torch.topk(distance, k=k, dim=1, largest=False) # Index Neuron BMU
        lowest_d = top_bmu.values # The lowest distance from sample data
        index_bmu = top_bmu.indices
        bmu = weight[index_bmu] # Neuron_bmu

        return bmu, index_bmu, lowest_d

    def _find_tau_lambda(self):
        """Compute the time constants of the exponential decays.

        ``tau = n_iter / ln(learning_rate / lr_final)`` makes the learning rate reach
        ``lr_final`` at the last iteration, and ``lam = n_iter / ln(sigma / sigma_f)``
        does the same for sigma.

        Returns
        -------
        tau, lam : float
            Time constants for the learning rate and for sigma.
        """
        T, alpha_0, alpha_f = self.n_iter, self.learning_rate, self.lr_final
        sigma_0, sigma_f = self.sigma, self.sigma_f

        tau = T / (math.log((alpha_0 / alpha_f)))
        lam = T / (math.log((sigma_0 / sigma_f)))

        return tau, lam

    def exponential_function(self, t):
        """Exponential decay of the learning rate and the neighborhood radius.

            alpha(t) = learning_rate * exp(-t / tau)
            sigma(t) = sigma * exp(-t / lam)

        Parameters
        ----------
        t : int
            Current iteration (0-based).

        Returns
        -------
        alpha, sigma : float
            Learning rate and neighborhood radius at iteration ``t``.
        """
        alpha_0, sigma_0 = self.learning_rate, self.sigma

        alpha = alpha_0 * (math.exp((-t / self.tau))) # Decay function learning rate
        sigma = sigma_0 * (math.exp((-t / self.lam))) # Decay function sigma
        return alpha, sigma

    def gaussian_neighborhood_function(self, weight, sigma, sample_data):
        """Gaussian neighborhood of the BMU of a single sample.

            h_cj = exp(-d_cj ** 2 / (2 * sigma ** 2))

        where ``d_cj`` is the grid distance between neuron ``j`` and the BMU ``c``.

        Parameters
        ----------
        weight : torch.Tensor
            Current neuron weights, shape ``(total_neuron, n_features)``.
        sigma : float
            Neighborhood radius at the current iteration.
        sample_data : torch.Tensor
            A single sample, shape ``(1, n_features)``.

        Returns
        -------
        torch.Tensor
            Influence of the BMU on every neuron, shape ``(total_neuron, 1)``.
        """
        _, index_bmu ,_ = self._find_bmu(weight, sample_data)
        d_cj = self.D_grid[index_bmu[0,0]]
        h_cj = torch.exp((-d_cj ** 2 / (2 * (sigma ** 2)))).unsqueeze(1)
        return h_cj

    def updated_weight(self, weight, t, sample_data):
        """Compute the weight update for one sample.

            delta_w_j = alpha(t) * h_cj(t) * (x - w_j)

        Parameters
        ----------
        weight : torch.Tensor
            Current neuron weights, shape ``(total_neuron, n_features)``.
        t : int
            Current iteration (0-based).
        sample_data : torch.Tensor
            A single sample, shape ``(1, n_features)``.

        Returns
        -------
        torch.Tensor
            The change to add to ``weight`` (not the new weights), same shape as ``weight``.
        """
        alpha, sigma = self.exponential_function(t)
        h_cj = self.gaussian_neighborhood_function(weight, sigma, sample_data)
        xw_v = sample_data - weight
        w_t = alpha * h_cj * xw_v
        return w_t

    def fit_train(self, steps = 10):
        """Train the map for ``n_iter`` iterations.

        Each iteration draws one random sample and updates all neuron weights.
        Quantization and topographic errors are recorded every ``steps`` iterations
        and at the last iteration. The generator is reset to ``random_state`` first, so
        calling ``fit_train`` again gives the same result.

        Parameters
        ----------
        steps : int, default=10
            Recording interval of the errors. Must be a positive integer.

        Returns
        -------
        SelfOrganizingMaps
            The trained model (``self``), so calls can be chained. Results are stored in
            ``final_weight``, ``quantization_err``, ``topographic_err`` and ``iter_steps``.
        """
        self.validate_type('steps', steps)
        self.validate_number('steps', steps)
        self.generator.manual_seed(self.random_state)
        w, _ = self._initialize_weight_neuron()
        iter_steps = []
        quantization_err = []
        topographic_err = []
        for t in range(self.n_iter):
            sample_data = self.get_random_sample()
            w += self.updated_weight(w, t, sample_data)
            if (t + 1) % steps == 0 or t == self.n_iter - 1:
                iter_steps.append(t)
                quantization_err.append(self.quantization_error(w))
                topographic_err.append(self.topographic_error(w))
        self.final_weight = w
        self.quantization_err = quantization_err
        self.topographic_err = topographic_err
        self.iter_steps = iter_steps
        return self

    def predict(self, data = None):
        """Find the BMU of each sample using the trained weights.

        Parameters
        ----------
        data : array-like, optional
            New samples in the **original units** (DataFrame, array, list or tensor of
            shape ``(n_samples, n_features)``). They are reshaped and scaled with the
            scaler fitted on the training data. If None, the training data is used.
            Do not pass ``self.data``, which is already scaled.

        Returns
        -------
        bmu : torch.Tensor
            BMU weights, shape ``(n_samples, 1, n_features)``.
        index_bmu : torch.Tensor
            BMU neuron indices, shape ``(n_samples, 1)``.
        lowest_d : torch.Tensor
            Distances to the BMUs, shape ``(n_samples, 1)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """

        if not hasattr(self, 'final_weight'):
            raise RuntimeError("Model has not been trained. Call fit_train() first.")
        
        if data is None:
            data = self.data
        else:
            data = torch.from_numpy(self.scaler.transform(self.processing_data(data).numpy())).float()
        
        bmu, index_bmu, lowest_d = self._find_bmu(self.final_weight, data)
        return bmu, index_bmu, lowest_d

    def heat_maps(self):
        """Count how many training samples each neuron wins (hit map).

        The result is also stored in ``self.heat_map``. Neurons with a count of 0 are
        never the BMU of any training sample.

        Returns
        -------
        torch.Tensor
            Hit counts, shape ``(x_neuron, y_neuron)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """

        if not hasattr(self, 'final_weight'):
            raise RuntimeError("Model has not been trained. Call fit_train() first.")
        
        _, index_bmu, _ = self._find_bmu(self.final_weight, self.data)
        heat = torch.bincount(index_bmu[:,0], minlength=self.total_neuron)
        heat_map = heat.reshape(self.x_neuron, self.y_neuron)
        self.heat_map = heat_map
        return heat_map

    def quantization_error(self, weight):
        """Mean distance between each training sample and its BMU (lower is better).

        The distance is measured in the scaled units of ``self.data``.

        Parameters
        ----------
        weight : torch.Tensor
            Neuron weights, shape ``(total_neuron, n_features)``.

        Returns
        -------
        torch.Tensor
            Scalar quantization error.
        """
        _, _, distance = self._find_bmu(weight, self.data)
        q_error = distance.mean()
        return q_error

    def topographic_error(self, weight):
        """Proportion of training samples whose two closest neurons are not grid neighbors.

        Two neurons count as neighbors when their grid distance is 1, so diagonal
        neighbors (distance of about 1.41) are counted as errors. Lower is better.

        Parameters
        ----------
        weight : torch.Tensor
            Neuron weights, shape ``(total_neuron, n_features)``.

        Returns
        -------
        torch.Tensor
            Scalar topographic error between 0 and 1.
        """
        _, index ,_ = self._find_bmu(weight, self.data, k=2)
        rows = index[:, 0]
        cols = index[:, 1]
        te = (self.D_grid[rows, cols] > 1).float().mean()
        return te

    def u_matrix(self):
        """Unified distance matrix (U-matrix) of the trained map.

        For each neuron, the mean weight-space distance to its directly adjacent
        neurons (up, down, left, right). High values mark cluster borders and low
        values mark the inside of clusters.

        Returns
        -------
        torch.Tensor
            U-matrix, shape ``(x_neuron, y_neuron)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if not hasattr(self, 'final_weight'):
            raise RuntimeError("Model has not been trained. Call fit_train() first.")
        filter = ((self.D_grid > 0.5) & (self.D_grid < 1.2)).float()
        w_dist = torch.cdist(self.final_weight, self.final_weight)
        u = ((filter * w_dist).sum(dim=1) / filter.sum(dim=1)).reshape(self.x_neuron, self.y_neuron)
        return u


    
        
