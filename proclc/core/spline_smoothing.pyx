cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport log

import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
DTYPE_INT = np.int32


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int fbleft(double[::1] x, double v):

    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        Py_ssize_t lo = 0
        Py_ssize_t hi = n

    if v < x[0]:
        return 0

    if v > x[n - 1]:
        return <int> n

    while lo < hi:
        i = (hi + lo) // 2
        if v > x[i]:
            lo = i + 1
        else:
            hi = i

    return <int> lo


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dot(double[::1] x, double[:, ::1] m, double c, double[::1] r):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = m.shape[1]
        Py_ssize_t i, j
        double a

    for i in range(p):
        a = 0.
        for j in range(n):
            a += x[j] * m[j, i]
        r[i] = c * a


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void q_y(double[:, ::1] q, double[::1] y, double[::1] z):

    cdef:
        Py_ssize_t n = q.shape[0]
        Py_ssize_t i, j
        double v

    for i in range(n):
        v = 0.
        for j in range(i, i + 3):
            v += q[i, j] * y[j]
        z[i] = v


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void hat_matrix_diagonal(double[:, ::1] q, double[:, ::1] b,
                              double a, double[::1] d):
    """
    Calculates diagonal elements of hat matrix
        I - a * Q' * inv(R + a * Q * Q') * Q
    Where
        R + a * Q * Q'
    is the Reinsch coefficient matrix. The inverse can be calculated
    through LDL Cholesky decomposition, by
        B^-1 = D^-1 * L^-1 + (I - L') * B^-1

    Args:
        q: Matrix Q.
        b: Inverse of Reinsch coefficient matrix.
        a: Smoothing parameter.
        d: Diagonal elements of hat matrix

    """
    cdef:
        Py_ssize_t n = q.shape[1]
        Py_ssize_t i, j, k, i0, i1
        double v, vi

    # s[0]
    d[0] = q[0, 0] * q[0, 0] * b[0, 0]
    # s[i], n-1 > i > 0
    for i in range(1, n - 1):
        i0 = max(i - 2, 0)
        i1 = min(i + 1, n - 2)
        v = 0.
        for j in range(i0, i1):
            vi = 0.
            for k in range(i0, i1):
                vi += q[k, i] * b[k, j]
            v += q[j, i] * vi
        d[i] = v
    # s[n-1]
    d[n - 1] = q[n - 3, n - 1] * q[n - 3, n - 1] * b[n - 3, n - 3]

    for i in range(n):
        d[i] = 1. - a * d[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void ldl_decompose(double[:, ::1] a, double[:, ::1] ml, double[::1] d):
    """
    Calculates LDL Cholesky decomposition.

    Args:
        a: Banded matrix
        ml: Lower angular matrix.
        d: Diagonal array of matrix D.

    """
    cdef:
        Py_ssize_t n = a.shape[0]
        Py_ssize_t i, i0, i1
        double t0, t1

    d[0] = a[0, 0]
    t0 = a[1, 0] / d[0]
    d[1] = a[1, 1] - (t0 * t0 * d[0])
    ml[1, 0] = t0
    for i in range(2, n):
        i0 = i - 2
        i1 = i - 1
        t0 = a[i, i0] / d[i0]
        t1 = (a[i, i1] - (ml[i1, i0] * t0 * d[i0])) / d[i1]
        d[i] = a[i, i] - (t1 * t1 * d[i1]) - (t0 * t0 * d[i0])
        ml[i, i0] = t0
        ml[i, i1] = t1

    for i in range(n):
        ml[i, i] = 1.


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void solve_linear(double[:, ::1] ml, double[::1] d, double[::1] z, double[::1] x):
    """
    Solves linear equation system by LDL Cholesky decomposition.
    For linear system Ax = z, and LDL decomposition of A:
        A = DLD'
    Solve the equations sequentially:
        Lu = z
        Dv = u
        L'x = v

    Args:
        ml: Lower diagonal matrix L.
        d: Diagonal of matrix D, 1D array.
        z: Dependent variable.
        x: Solution

    """
    cdef:
        Py_ssize_t n = d.shape[0]
        Py_ssize_t i, j
        double * u = <double *> malloc(n * sizeof(double))
        double * v = <double *> malloc(n * sizeof(double))

    # solve Lu = z
    u[0] = z[0]
    u[1] = z[1] - ml[1, 0] * u[0]
    for i in range(2, n):
        u[i] = z[i] - ml[i, i - 1] * u[i - 1] - ml[i, i - 2] * u[i - 2]

    # solve Dv = u
    for i in range(n):
        v[i] = u[i] / d[i]

    # solve L'x = v
    x[n - 1] = v[n - 1]
    x[n - 2] = v[n - 2] - ml[n - 1, n - 2] * x[n - 1]
    for i in range(3, n + 1):
        j = n - i
        x[j] = v[j] - ml[j + 1, j] * x[j + 1] - ml[j + 2, j] * x[j + 2]

    free(u)
    free(v)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void inverse_reinsch_coef_matrix(double[:, ::1] ml, double[::1] d, double[:, ::1] b):
    """
    Calculates the inverse of Reinsch coefficient matrix by LDL
    decomposition.

    Args:
        ml: Lower tri-angular matrix L.
        d: Diagonal of matrix D.
        b: Inverse matrix.

    Returns:
        array: Inverse matrix.

    """
    cdef:
        Py_ssize_t n = d.shape[0]
        Py_ssize_t i, j, k, h, j1
        double * d2 = <double *> malloc(n * sizeof(double))
        double v, d0

    for i in range(n):
        d2[i] = 1. / d[i]

    for i in range(1, n + 1):
        j = n - i
        j1 = j + 8
        if j1 > n:
            j1 = n

        d0 = d2[j]
        for k in range(j + 1, j1):
            v = 0.
            for h in range(j + 1, j1):
                v = v - ml[h, j] * b[h, k]
            b[j, k] = v
            b[k, j] = v
            d0 = d0 - ml[k, j] * v

        b[j, j] = d0

    free(d2)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void reinsch_coef_matrix(double[:, ::1] q, double[:, ::1] r,
                              double a, double[:, ::1] c):
    """
    Calculates coefficient matrix of Reinsch algorith.

    Args:
        q: Triangular matrix.
        r: Triangular matrix.
        a: Smooth parameter.
        c: Coefficient matrix.

    """
    cdef:
        Py_ssize_t n = q.shape[0]
        Py_ssize_t i, k, j, i0, i1, j0, j1
        double v

    for i in range(n):
        i1 = i + 3
        i0 = max(0, i - 2)
        for j in range(i0, min(i1, n)):
            j0 = max(i, j)
            j1 = min(i1, j + 3)
            v = 0.
            for k in range(j0, j1):
                v += q[i, k] * q[j, k]
            c[i, j] = v * a

        for j in range(max(0, i - 1), min(n, i + 2)):
            c[i, j] += r[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void cubic_splines(double[::1] x, double[:, ::1] q, double[:, ::1] r):
    """
    Generates cubic spline matrices.

    Args:
        x: Knots
        q: Q matrix
        r: R matrix

    Returns:
        array: Spline matrices

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i, j
        double * h = <double *> malloc((n - 1) * sizeof(double))
        double * h2 = <double *> malloc((n - 1) * sizeof(double))

    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
        h2[i] = 1. / h[i]

    for i in range(1, n - 1):
        j = i - 1
        q[j, j] = h2[j]
        q[j, i] = -(h2[j] + h2[i])
        q[j, i + 1] = h2[i]

    for i in range(1, n - 2):
        j = i - 1
        r[j, j] = (h[j] + h[i]) / 3.
        r[j, i] = h[i] / 6.
        r[i, j] = h[i] / 6.
    r[i, i] = (h[i] + h[j]) / 3.

    free(h)
    free(h2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:, ::1] spline_coef(double[::1] x, double[::1] z, double[::1] g):
    """
    Calculates spline coefficients.

    Args:
        x: Interval edges.
        z: Smoothed values.
        g: Second derivatives.

    Returns:
        array: Coefficients.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double * g2 = <double *> malloc(n * sizeof(double))
        double[:, ::1] c = np.zeros((4, n - 1), dtype=DTYPE)
        double dz, dx

    for i in range(n - 2):
        g2[i + 1] = g[i]
        c[0, i] = z[i]
        c[2, i + 1] = g[i] / 2.
    c[0, n - 2] = z[n - 2]

    # coefficients b and
    for i in range(n - 1):
        dz = z[i + 1] - z[i]
        dx = x[i + 1] - x[i]
        c[1, i] = dz / dx - dx / 6. * (2. * g2[i] + g2[i + 1])
        c[3, i] = (g2[i + 1] - g2[i]) / dx / 6.

    free(g2)

    return c


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[::1] predict_out_range(double[::1] x, double[:, ::1] coefs, double b0, double bn):
    """
    Predicts x out range of the bounds in fitting.

    Args:
        x: x
        coefs: Coefficients.
        b0: Left bound.
        bn: Right bound.

    Returns:
        array: Predicted array.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t nc = coefs.shape[1]
        Py_ssize_t i
        double[::1] p = np.zeros(n, dtype=DTYPE)
        double cn_a, cn_b, c0_a, c0_b

    cn_a = coefs[0, nc - 1]
    cn_b = coefs[1, nc - 1]
    c0_a = coefs[0, 0]
    c0_b = coefs[1, 0]
    for i in range(n):
        if x[i] > bn:
            p[i] = cn_a + (x[i] - bn) * cn_b
        else:
            p[i] = c0_a + (x[i] - b0) * c0_b
    return p


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[::1] predict(double[::1] x, double[::1] g, double[::1] d, double[::1] intervals):
    """
    Predicts x.

    Args:
        x: x for prediction.
        g: Smoothed array.
        d: 2nd derivative, with 0 padded at two terminal sides of the
            array.
        intervals: Interval edges for fitting.

    Returns:
        array: Predicted array.

    """
    cdef:
        Py_ssize_t n = intervals.shape[0]
        Py_ssize_t nx = x.shape[0]
        Py_ssize_t i, jr, jl
        double * h = <double *> malloc((n - 1) * sizeof(double))
        double[::1] p = np.zeros(nx, dtype=DTYPE)
        double t, a, d_lo, d_hi

    for i in range(n - 1):
        h[i] = intervals[i + 1] - intervals[i]

    for i in range(nx):
        jr = fbleft(intervals, x[i])
        jl = jr - 1
        if jr == 0:
            p[i] = g[0]
        elif jr == n:
            p[i] = g[n - 1]
        else:
            d_lo = x[i] - intervals[jl]
            d_hi = intervals[jr] - x[i]
            t = (d_lo * g[jr] + d_hi * g[jl]) / h[jl]
            a = - d_lo * d_hi / 6. * ((1. + d_lo / h[jl]) * d[jr]
                                     + (1. + d_hi / h[jl]) * d[jl])
            p[i] = t + a
    free(h)

    return p


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef (double, double, double) calculate_criteria(double[::1] residuals, double[::1] d):
    """ Calculates LOO_CV, GCV, AICC. """
    cdef:
        Py_ssize_t n = residuals.shape[0]
        Py_ssize_t i
        double n_d = <double> n
        double cv = 0.
        double mse = 0.
        double t = 0.
        double err, gcv, aicc

    for i in range(n):
        err = residuals[i] / (1. - d[i])
        cv += err * err
        t += d[i]
        mse += residuals[i] * residuals[i]

    cv /= n_d
    mse /= n_d
    # GCV
    gcv = mse / ((1. - t / n_d) ** 2.)
    # AICC
    aicc = log(mse) + 1. + 2. * (1. + t) / (n_d - t - 2.)

    return cv, gcv, aicc


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef fit_ss(double[::1] x, double[::1] y, double[::1] params):
    """ Fits smoothing splines. """

    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t nm = params.shape[0]
        Py_ssize_t n2 = n - 2
        Py_ssize_t i, j
        double[:, ::1] r = np.zeros((n2, n2), dtype=DTYPE)
        double[:, ::1] q = np.zeros((n2, n), dtype=DTYPE)
        double[:, ::1] re_coef = np.zeros((n2, n2), dtype=DTYPE)
        double[:, ::1] ml = np.zeros((n2, n2), dtype=DTYPE)
        double[:, ::1] b = np.zeros((n2, n2), dtype=DTYPE)
        double[:, ::1] criteria = np.zeros((3, nm), dtype=DTYPE)
        double[:, ::1] fit_values = np.zeros((nm, n), dtype=DTYPE)
        double[:, ::1] d2 = np.zeros((nm, n2), dtype=DTYPE)
        double[::1] d = np.zeros(n2, dtype=DTYPE)
        double[::1] s = np.zeros(n2, dtype=DTYPE)
        double[::1] z = np.zeros(n, dtype=DTYPE)
        double[::1] h = np.zeros(n, dtype=DTYPE)
        double[::1] errors = np.zeros(n, dtype=DTYPE)
        double p

    # Q, R matrices
    cubic_splines(x, q, r)

    # Q * y
    q_y(q, y, z)

    # selection of smoothing parameters
    for i in range(nm):
        p = params[i]
        reinsch_coef_matrix(q, r, p, re_coef)
        # DLD Cholesky decomposition
        ldl_decompose(re_coef, ml, d)
        # gamma by solving linear system
        solve_linear(ml, d, z, s)
        # diagonal elements of hat matrix
        inverse_reinsch_coef_matrix(ml, d, b)
        hat_matrix_diagonal(q, b, p, h)
        # fitted curve
        dot(s, q, p, errors)

        cv, gcv, aicc = calculate_criteria(errors, h)
        criteria[0, i] = cv
        criteria[1, i] = gcv
        criteria[2, i] = aicc

        for j in range(n):
            fit_values[i, j] = y[j] - errors[j]
        for j in range(n2):
            d2[i, j] = s[j]

    return (np.asarray(criteria), np.asarray(fit_values),
            np.asarray(d2), np.asarray(q), np.asarray(r))


cpdef ss_coefs(double[::1] x, double[::1] z, double[::1] g):
    cdef:
        double[:, ::1] coef = spline_coef(x, z, g)
    return np.asarray(coef)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef predict_ss(double[::1] x, double[::1] f, double[::1] g,
                 double[::1] b, double[:, ::1] coef):
    """
    Predicts values.
    
    Args:
        x: Array for prediction.
        f: Fitted values for predefined x.
        g: 2nd derivative.
        b: Predefined x.
        coef: Coefficients.
    
    Returns:
        Predicted array.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t n0 = b.shape[0]
        int[::1] ix_in = np.zeros(n, dtype=DTYPE_INT)
        int[::1] ix_out = np.zeros(n, dtype=DTYPE_INT)
        int ii = 0
        int io = 0
        double[::1] x_in = np.zeros(n, dtype=DTYPE)
        double[::1] x_out = np.zeros(n, dtype=DTYPE)
        double[::1] y = np.zeros(n, dtype=DTYPE)
        double[::1] yp_in, yp_out
        double b0 = b[0]
        double bn = b[n0 - 1]

    for i in range(n):
        if bn >= x[i] >= b0:
            ix_in[ii] = <int> i
            x_in[ii] = x[i]
            ii += 1
        else:
            ix_out[io] = <int> i
            x_out[io] = x[i]
            io += 1

    # do prediction
    if ii > 0:
        yp_in = predict(x_in[:ii], f, g, b)
        for i in range(ii):
            y[ix_in[i]] = yp_in[i]

    if io > 0:
        yp_out = predict_out_range(x_out[:io], coef, b0, bn)
        for i in range(io):
            y[ix_out[i]] = yp_out[i]

    return np.asarray(y)
