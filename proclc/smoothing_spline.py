"""
This module provides natural cubic spline smoothing with automatic
parameter selection using improved AIC.

References:
    [1] Reinsch CH. Smoothing by spline functions. Numer Math. 1967, 10,
        177–183.
    [2] Eubank RL. Nonparametric Regression and Spline Smoothing. 2nd
        Ed. New York; Basel: Marcel Dekker. 1999.
    [3] Green PJ, Silverman BW. Nonparametric Regression and
        Generalized Linear Models: A roughness penalty approach.
        Chapman and Hall/CRC. 1993.
    [4] Hurvich CM, Simonoff JS, Tsai CL. Smoothing Parameter Selection
        in Nonparametric Regression Using an Improved Akaike
        Information Criterion. J R Statist Soc B. 1998, 60, 271-293.
    [5] Lee TCM. Smoothing parameter selection for smoothing splines: a
        simulation study. Comput Stat Data Anal. 2003, 42, 139–148.
    [6] Clarke B, Fokoue E, Zhang HH. Principles and Theory for Data
        Mining and Machine Learning. Springer New York, NY. 2009.
    [7] Hutchinson MF, de Hoog FR. Smoothing Noisy Data with Spline
        Functions. Numer Math. 1985, 47, 99-106.

"""
import numpy as np

from typing import List, Optional, Tuple, Dict

from .core import fit_ss, ss_coefs, predict_ss


class SplineSmoothing:
    """
    This class performs spline smoothing with a series of smoothing
    parameters, with optimized one selected using improved AIC and
    generalized cross validation.

    Args:
        smooth_params (optional, list): Smoothing parameters.
            Defaulted to 1 to 10^6.
        criteria (str): Criterion for selection of parameters,
            {`aicc`, `gcv`, `cv`}
            `aicc`: Improved AIC, this is the default.
            `gcv`: Generalized cross validation.
            `cv`: Cross validation.

    """
    def __init__(self,
                 smooth_params: Optional[List[float]] = None,
                 criteria: str = "aicc"):

        self.smooth_params = [10 ** v for v in range(-3, 8)]
        self.criteria = criteria

        if smooth_params is not None:
            self.smooth_params = smooth_params

        self._Q: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None
        self._score: Dict[str, List[float]] = {"aicc": [], "cv": [], "gcv": []}
        self._best_index: Optional[int] = None
        self._x: Optional[np.ndarray] = None
        self._coefficients: Optional[np.ndarray] = None
        self._d2: Optional[np.ndarray] = None
        self._best_fit: Optional[np.ndarray] = None

        self._check_params()

    @property
    def best_smoothing_param(self) -> float:
        """
        Returns the best smoothing parameters.

        Raises:
            ValueError

        """
        if self._best_index is None:
            raise ValueError("The smoothing is not performed.")
        return self.smooth_params[self._best_index]

    @property
    def best_criterion(self) -> float:
        """
        Returns the lowest criterion value.

        Raises:
            ValueError

        """
        if self._best_index is None:
            raise ValueError("The smoothing is not performed.")
        return self._score[self.criteria][self._best_index]

    @property
    def cv_scores(self) -> List[Tuple[float, float]]:
        """ Returns cross validation values. """
        return list(zip(self.smooth_params, self._score["cv"]))

    @property
    def gcv_scores(self) -> List[Tuple[float, float]]:
        """ Returns generalized cross validation scores. """
        return list(zip(self.smooth_params, self._score["gcv"]))

    @property
    def aicc_scores(self) -> List[Tuple[float, float]]:
        """ Returns AICC scores. """
        return list(zip(self.smooth_params, self._score["aicc"]))

    @property
    def interval_edges(self) -> Optional[np.ndarray]:
        """ Returns interval edges. """
        if self._best_index is None:
            return None
        return self._x

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """ Returns coefficients of splines. """
        if self._best_index is None:
            return None
        return self._coefficients

    @property
    def q(self):
        """ Returns matrix Q. """
        return self._Q

    @property
    def r(self):
        """ Returns matrix R. """
        return self._R

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates spline smoothing.

        Args:
            x: Knots, must be in strictly increasing order.
            y: Responses corresponding to x.

        Returns:
            array: Smoothed values.

        """
        self._check_array(x, y)
        return self._fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the smoothing curve.

        Args:
            x: Array for evaluation.

        Returns:
            array: Evaluated curve.

        """
        if self._coefficients is None:
            raise RuntimeError("The coefficients are not estimated. Should "
                               "run the fit in advance.")
        x = self._check_pred_x(x)

        y = predict_ss(x, self._best_fit, self._d2,
                       self._x, self._coefficients)

        return y

    def _fit(self, x, y) -> np.ndarray:
        """ Smooths the arrays. """
        self._initialize_scores()

        x = x.astype(np.float64)
        y = y.astype(np.float64)
        params = np.fromiter(self.smooth_params, dtype=np.float64)

        criteria, fit_values, d2, q, r = fit_ss(x, y, params)
        print(criteria)

        # Q, R matrices
        self._Q = q
        self._R = r
        self._x = x
        self._score["cv"] = criteria[0]
        self._score["gcv"] = criteria[1]
        self._score["aicc"] = criteria[2]

        i = np.argmin(self._score[self.criteria])
        best_fit = fit_values[i]
        self._best_index = i
        self._best_fit = best_fit

        # coefficients
        self._coefficients = ss_coefs(x, best_fit, d2[i])

        best_d2 = np.zeros_like(x)
        best_d2[1:-1] = d2[i]
        self._d2 = best_d2

        return best_fit

    def _initialize_scores(self):
        """ Initializes scores for new fitting. """
        self._score = {"aicc": [], "cv": [], "gcv": []}
        self._best_index = None
        self._coefficients = None

    def _check_params(self):
        """ Check whether parameters are valid. """
        if min(self.smooth_params) <= 0:
            raise ValueError("The smooth parameters must be non-negative.")

        if self.criteria not in ("aicc", "gcv", "cv"):
            raise ValueError(f"Expected 'aicc' or 'gcv', got {self.criteria}.")

    @staticmethod
    def _check_array(x, y):
        """ Checks whether the knots array is valid. """
        x_shape = x.shape
        y_shape = y.shape
        if x_shape != y_shape:
            raise ValueError("Shapes of x and y must be consistent.")

        if x.ndim != 1:
            raise ValueError("Currently only 1D is accepted.")

        if x.size < 3:
            raise ValueError(f"The number of knots must be larger than 3.")

        xd = np.diff(x)
        if (xd <= 0).any():
            raise ValueError("The knots array x must be strictly increasing.")

    @staticmethod
    def _check_pred_x(x):
        """ Checks x and returns x as float64 for calculation. """
        return x.astype(np.float64)
