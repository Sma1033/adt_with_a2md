""" Fast DTW routines. """
""" https://github.com/craffel/djitw"""


import numba
import numpy as np


@numba.jit(nopython=True)
def band_mask(radius, mask):
    """Construct band-around-diagonal mask (Sakoe-Chiba band).  When
    ``mask.shape[0] != mask.shape[1]``, the radius will be expanded so that
    ``mask[-1, -1] = 1`` always.

    `mask` will be modified in place.

    Parameters
    ----------
    radius : float
        The band radius (1/2 of the width) will be
        ``int(radius*min(mask.shape))``.
    mask : np.ndarray
        Pre-allocated boolean matrix of zeros.

    Examples
    --------
    >>> mask = np.zeros((8, 8), dtype=np.bool)
    >>> band_mask(.25, mask)
    >>> mask.astype(int)
    array([[1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1]])
    >>> mask = np.zeros((8, 12), dtype=np.bool)
    >>> band_mask(.25, mask)
    >>> mask.astype(int)
    array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    """
    nx, ny = mask.shape
    # The logic will be different depending on whether there are more rows
    # or columns in the mask.  Coding it this way results in some code
    # duplication but it's the most efficient way with numba
    if nx < ny:
        # Calculate the radius in indices, rather than proportion
        radius = int(round(nx*radius))
        # Force radius to be at least one
        radius = 1 if radius == 0 else radius
        for i in xrange(nx):
            for j in xrange(ny):
                # If this i, j falls within the band
                if i - j + (nx - radius) < nx and j - i + (nx - radius) < ny:
                    # Set the mask to 1 here
                    mask[i, j] = 1
    # Same exact approach with ny/ny and i/j switched.
    else:
        radius = int(round(ny*radius))
        radius = 1 if radius == 0 else radius
        for i in range(nx):
            for j in range(ny):
                if j - i + (ny - radius) < ny and i - j + (ny - radius) < nx:
                    mask[i, j] = 1


@numba.jit(nopython=True)
def dtw_core(dist_mat, add_pen, mul_pen, traceback):
    """Core dynamic programming routine for DTW.

    `dist_mat` and `traceback` will be modified in-place.

    Parameters
    ----------
    dist_mat : np.ndarray
        Distance matrix to update with lowest-cost path to each entry.
    add_pen : int or float
        Additive penalty for non-diagonal moves.
    mul_pen : int or float
        Multiplicative penalty for non-diagonal moves.
    traceback : np.ndarray
        Matrix to populate with the lowest-cost traceback from each entry.
    """
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    # TOOD: Would probably be faster if xrange(1, dist_mat.shape[0])
    for i in range(dist_mat.shape[0] - 1):
        for j in range(dist_mat.shape[1] - 1):
            # Diagonal move (which has no penalty) is lowest
            if dist_mat[i, j] <= mul_pen*dist_mat[i, j + 1] + add_pen and \
               dist_mat[i, j] <= mul_pen*dist_mat[i + 1, j] + add_pen:
                traceback[i + 1, j + 1] = 0
                dist_mat[i + 1, j + 1] += dist_mat[i, j]
            # Horizontal move (has penalty)
            elif (dist_mat[i, j + 1] <= dist_mat[i + 1, j] and
                  mul_pen*dist_mat[i, j + 1] + add_pen <= dist_mat[i, j]):
                traceback[i + 1, j + 1] = 1
                dist_mat[i + 1, j + 1] += mul_pen*dist_mat[i, j + 1] + add_pen
            # Vertical move (has penalty)
            elif (dist_mat[i + 1, j] <= dist_mat[i, j + 1] and
                  mul_pen*dist_mat[i + 1, j] + add_pen <= dist_mat[i, j]):
                traceback[i + 1, j + 1] = 2
                dist_mat[i + 1, j + 1] += mul_pen*dist_mat[i + 1, j] + add_pen


@numba.jit(nopython=True)
def dtw_core_masked(dist_mat, add_pen, mul_pen, traceback, mask):
    """Core dynamic programming routine for DTW, with an index mask, so that
    the possible paths are constrained.

    `dist_mat` and `traceback` will be modified in-place.

    Parameters
    ----------
    dist_mat : np.ndarray
        Distance matrix to update with lowest-cost path to each entry.
    add_pen : int or float
        Additive penalty for non-diagonal moves.
    mul_pen : int or float
        Multiplicative penalty for non-diagonal moves.
    traceback : np.ndarray
        Matrix to populate with the lowest-cost traceback from each entry.
    mask : np.ndarray
        A boolean matrix, such that ``mask[i, j] == 1`` when the index ``i, j``
        should be allowed in the DTW path and ``mask[i, j] == 0`` otherwise.
    """
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    # TOOD: Would probably be faster if xrange(1, dist_mat.shape[0])
    for i in range(dist_mat.shape[0] - 1):
        for j in range(dist_mat.shape[1] - 1):
            # If this point is not reachable, set the cost to infinity
            if not mask[i, j] and not mask[i, j + 1] and not mask[i + 1, j]:
                dist_mat[i + 1, j + 1] = np.inf
            else:
                # Diagonal move (which has no penalty) is lowest, or is the
                # only valid move
                if ((dist_mat[i, j] <= mul_pen*dist_mat[i, j + 1] + add_pen
                     or not mask[i, j + 1]) and
                    (dist_mat[i, j] <= mul_pen*dist_mat[i + 1, j] + add_pen
                     or not mask[i + 1, j])):
                    traceback[i + 1, j + 1] = 0
                    dist_mat[i + 1, j + 1] += dist_mat[i, j]
                # Horizontal move (has penalty)
                elif ((dist_mat[i, j + 1] <= dist_mat[i + 1, j]
                       or not mask[i + 1, j]) and
                      (mul_pen*dist_mat[i, j + 1] + add_pen <= dist_mat[i, j]
                       or not mask[i, j])):
                    traceback[i + 1, j + 1] = 1
                    dist_mat[i + 1, j + 1] += (mul_pen*dist_mat[i, j + 1] +
                                               add_pen)
                # Vertical move (has penalty)
                elif ((dist_mat[i + 1, j] <= dist_mat[i, j + 1]
                       or not mask[i, j + 1]) and
                      (mul_pen*dist_mat[i + 1, j] + add_pen <= dist_mat[i, j]
                       or not mask[i, j])):
                    traceback[i + 1, j + 1] = 2
                    dist_mat[i + 1, j + 1] += (mul_pen*dist_mat[i + 1, j] +
                                               add_pen)


def dtw(distance_matrix, gully=1., additive_penalty=0.,
        multiplicative_penalty=1., mask=None, inplace=True):
    """ Compute the dynamic time warping distance between two sequences given a
    distance matrix.  The score is unnormalized.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distances between two sequences.
    gully : float
        Sequences must match up to this porportion of shorter sequence. Default
        1., which means the entirety of the shorter sequence must be matched
        to part of the longer sequence.
    additive_penalty : int or float
        Additive penalty for non-diagonal moves. Default 0. means no penalty.
    multiplicative_penalty : int or float
        Multiplicative penalty for non-diagonal moves. Default 1. means no
        penalty.
    mask : np.ndarray
        A boolean matrix, such that ``mask[i, j] == 1`` when the index ``i, j``
        should be allowed in the DTW path and ``mask[i, j] == 0`` otherwise.
        If None (default), don't apply a mask - this is more efficient than
        providing a mask of all 1s.
    inplace : bool
        When ``inplace == True`` (default), `distance_matrix` will be modified
        in-place when computing path costs.  When ``inplace == False``,
        `distance_matrix` will not be modified.

    Returns
    -------
    x_indices : np.ndarray
        Indices of the lowest-cost path in the first dimension of the distance
        matrix.
    y_indices : np.ndarray
        Indices of the lowest-cost path in the second dimension of the distance
        matrix.
    score : float
        DTW score of lowest cost path through the distance matrix, including
        penalties.
    """
    if np.isnan(distance_matrix).any():
        raise ValueError('NaN values found in distance matrix.')
    if not inplace:
        distance_matrix = distance_matrix.copy()
    # Pre-allocate path length matrix
    traceback = np.empty(distance_matrix.shape, np.uint8)
    # Don't use masked DTW routine if no mask was provided
    if mask is None:
        # Populate distance matrix with lowest cost path
        dtw_core(distance_matrix, additive_penalty, multiplicative_penalty,
                 traceback)
    else:
        dtw_core_masked(distance_matrix, additive_penalty,
                        multiplicative_penalty, traceback, mask)
    if gully < 1.:
        # Allow the end of the path to start within gully percentage of the
        # smaller distance matrix dimension
        gully = int(gully*min(distance_matrix.shape))
    else:
        # When gully is 1 require matching the entirety of the smaller sequence
        gully = min(distance_matrix.shape) - 1

    # Find the indices of the smallest costs on the bottom and right edges
    i = np.argmin(distance_matrix[gully:, -1]) + gully
    j = np.argmin(distance_matrix[-1, gully:]) + gully

    # Choose the smaller cost on the two edges
    if distance_matrix[-1, j] > distance_matrix[i, -1]:
        j = distance_matrix.shape[1] - 1
    else:
        i = distance_matrix.shape[0] - 1

    # Score is the final score of the best path
    score = float(distance_matrix[i, j])

    # Pre-allocate the x and y path index arrays
    x_indices = np.zeros(sum(traceback.shape), dtype=np.int)
    y_indices = np.zeros(sum(traceback.shape), dtype=np.int)
    # Start the arrays from the end of the path
    x_indices[0] = i
    y_indices[0] = j
    # Keep track of path length
    n = 1

    # Until we reach an edge
    while i > 0 and j > 0:
        # If the tracback matrix indicates a diagonal move...
        if traceback[i, j] == 0:
            i = i - 1
            j = j - 1
        # Horizontal move...
        elif traceback[i, j] == 1:
            i = i - 1
        # Vertical move...
        elif traceback[i, j] == 2:
            j = j - 1
        # Add these indices into the path arrays
        x_indices[n] = i
        y_indices[n] = j
        n += 1
    # Reverse and crop the path index arrays
    x_indices = x_indices[:n][::-1]
    y_indices = y_indices[:n][::-1]

    return x_indices, y_indices, score
