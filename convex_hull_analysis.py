import json

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from common import loadData


def jaccardIndexMC(
        hull1: ConvexHull,
        hull2: ConvexHull,
        n_samples: int = 200_000,
        seed: int = 0) -> dict:
    """
    Estimate Jaccard index = |A ∩ B|/|A ∪ B| via uniform Monte Carlo sampling.

    Works in any dimension. Samples uniformly inside the bounding box of both
    hulls, then tests membership using the half-space representation (Ax+b≤0).

    Parameters:
    -----------
    hull1   : Scipy ConvexHull object
    hull2   : Scipy ConvexHull object
    n_samples   : int, numer of samples to use for the Monter-Carlo integration
    seed:   int, RNG seed

    Returns:
    --------
    float, Jaccard index
    """
    rng = np.random.default_rng(seed)
    pts = hull1.points

    # Bounding box enclosing both hulls
    all_pts = np.vstack([hull1.points, hull2.points])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)

    # Uniform samples in bounding box
    samples = rng.uniform(lo, hi, size=(n_samples, pts.shape[1]))
    box_vol = np.prod(hi - lo)

    def in_hull(h: ConvexHull, pts: np.ndarray) -> np.ndarray:
        # Each row:
        # equations[i] = [normal | offset], point inside iff A@x+b <= 0
        return np.all(pts @ h.equations[:, :-1].T
                      + h.equations[:, -1] <= 1e-10, axis=1)

    in1 = in_hull(hull1, samples)
    in2 = in_hull(hull2, samples)

    n_intersect = np.sum(in1 & in2)
    n_union = np.sum(in1 | in2)

    vol1 = hull1.volume
    vol2 = hull2.volume
    vol_int = (n_intersect / n_samples) * box_vol
    vol_uni = (n_union / n_samples) * box_vol

    res = {
        "jaccard":           vol_int / vol_uni if vol_uni > 0 else 0.0,
        "vol_intersection":  vol_int,
        "vol_union":         vol_uni,
        "vol_hull1":         vol1,
        "vol_hull2":         vol2,
        "vol_overlap_frac1": vol_int / vol1 if vol1 > 0 else 0.0,
        "vol_overlap_frac2": vol_int / vol2 if vol2 > 0 else 0.0,
    }
    return res["jaccard"], res


def _hull_halfspaces(points: np.ndarray) -> np.ndarray:
    """
    Returns the half-space representation of the convex hull of input
    points cloud. Scipy's ConvexHull stores each facet as
                    equations[i] = [n, d], where
    n @ x + d <= 0 defines the interior side,
    i.e. the hull is { x : A @ x + b <= 0 } with A = equations[:, :-1] and
    b = equations[:, -1].

    Parameters:
    -----------
    points  : numpy array, the point cloud

    Returns:
    --------
    Numpy array, contains the half-space representation of the input
    convex hull
    """
    hull = ConvexHull(points)
    return hull.equations          # shape (n_facets, 4)  → [nx, ny, nz, d]


def _chebyshev_center(halfspaces: np.ndarray) -> np.ndarray:
    """
    Find a feasible interior point of the polyhedron defined by *halfspaces*
    via the Chebyshev-centre LP:
        maximise  r
        subject to  A @ x + ||A_i|| * r + b_i <= 0  for all i
                    r >= 0
    Parameters:
    -----------
    halfspaces  : numpy array, contains the half-space representation of a
    convex hull

    Returns:
    --------
    Numppy array, Holds the centre point (the last variable, r, is dropped)
    It returns None if no feasible point exists.
    """
    A = halfspaces[:, :-1]          # (m, 3)
    b = halfspaces[:, -1]           # (m,)
    norms = np.linalg.norm(A, axis=1, keepdims=True)   # (m, 1)

    # Variables: [x0, x1, x2, r]
    # Minimise -r  (i.e. maximise r)
    c = np.zeros(4)
    c[-1] = -1.0

    # A_ub @ v <= b_ub  →  A_i @ x + ||A_i|| * r <= -b_i
    A_ub = np.hstack([A, norms])    # (m, 4)
    b_ub = -b                       # (m,)

    res = linprog(c,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  bounds=[(None, None)]*3+[(0, None)],
                  method="highs")

    if res.status != 0 or res.x[-1] < 1e-10:
        return None                 # hulls do not intersect

    return res.x[:3]


def convexHullIntersectionPoints(
        cloud_a: np.ndarray,
        cloud_b: np.ndarray) -> np.ndarray | None:
    """
    Compute the vertices of the intersection of the convex hulls of two 3-D
    point clouds.

    Parameters:
    -----------
    cloud_a     : Numpy array of shape (N, 3) (point cloud A)
    cloud_b     : Numpy array of shape (N, 3) (point cloud B)

    Note:
    -----
    Each input point cloud must contain at least 4 non-coplanar points.

    Returns:
    -------_
    vertices: ndarray, shape (K, 3)  or  None
    Vertices of the intersection polyhedron, or None if the hulls do not
    intersect (or the intersection is degenerate / lower-dimensional).
    """
    cloud_a = np.asarray(cloud_a, dtype=float)
    cloud_b = np.asarray(cloud_b, dtype=float)

    if cloud_a.shape[1] != 3 or cloud_b.shape[1] != 3:
        raise ValueError("Point clouds must have shape (N, 3).")

    # --- Build half-space sets for each hull ---
    hs_a = _hull_halfspaces(cloud_a)
    hs_b = _hull_halfspaces(cloud_b)
    combined_hs = np.vstack([hs_a, hs_b])

    # --- Find a Chebyshev centre of the intersection ---
    interior = _chebyshev_center(combined_hs)
    if interior is None:
        return None

    # --- Enumerate vertices of the intersection polyhedron ---
    try:
        hsi = HalfspaceIntersection(combined_hs, interior)
        # hsi.intersections are the dual vertices; take convex hull to get
        # the actual vertices of the primal polyhedron.
        intersection_hull = ConvexHull(hsi.intersections)
        vertices = hsi.intersections[intersection_hull.vertices]
        intersections = hsi.intersections
    except Exception:
        # Degenerate case (e.g. intersection is a single point or edge)
        return None

    return vertices, intersections


if __name__ == "__main__":
    # Choose a task: toy (circles/moons) or mnist
    task = "toy"
    # Load the parameters of the experiment
    if task == "toy":
        with open("parameters.json") as f:
            params = json.load(f)
    else:
        with open("parameters_mnist.json") as f:
            params = json.load(f)

    n_type = params["n_type"]
    batch_size = params["batch_size"]
    n_experiments = params["n_experiments"]
    n_neurons = params["n_neurons"]
    lr = params["lrate"]
    epochs = params["epochs"]
    alpha = params["alpha"]
    mode = params["mode"]
    dir_ = params["directory"]

    output_dir = "./convex_hull_results/"

    sparsity = [0.0, 0.3]
    sparsity = [0.0]

    # Initialize all the necesary Numpy arrays
    # Intersection points and Jaccard index
    ints = np.zeros((len(sparsity), 4, n_experiments))
    jacc = np.zeros((len(sparsity), 4, n_experiments))
    for ii, sp in enumerate(sparsity):
        print("--" * 50)
        print(f" >>>>>>>>>>>>>> SPARSITY = {sp}")
        print(f" >>>>>>>>>>>>>> NETWORK  = {n_type}")

        # Run the TDA analysis for each sparsity value
        R1_seq, T1_seq, R2_seq, T2_seq = loadData(
                dir_=dir_,
                task=task,
                mode="sequential",
                n_type=n_type,
                sparsity=sp,
                n_experiments=n_experiments,
                epochs=epochs,
                n_neurons=n_neurons)

        # Load the data for the INTERLEAVED learning schedule
        R1_int, T1_int, R2_int, T2_int = loadData(
                dir_=dir_,
                task=task,
                mode="interleaved",
                n_type=n_type,
                sparsity=sp,
                n_experiments=n_experiments,
                epochs=epochs,
                n_neurons=n_neurons)

        # Pack all the responses, R, and targets, T, into lists
        R = [R1_seq, R2_seq, R1_int, R2_int]
        T = [T1_seq, T2_seq, T1_int, T2_int]
        n_samples = R[0].shape[1]

        # Standarize the preactivations, R, using the R[0] as reference
        scaler = StandardScaler()
        for i in range(len(R)):
            if i == 0:
                R[i] = scaler.fit_transform(
                        R[i].reshape(n_experiments*n_samples,
                                     n_neurons)
                        )
            else:
                R[i] = scaler.transform(R[i].reshape(n_experiments*n_samples,
                                                     n_neurons))
            R[i] = R[i].reshape(n_experiments, n_samples, n_neurons)

        # Initialize PCA
        embed = PCA(n_components=3)

        flag = True
        # Run over all four combinations
        # sequential/sequential, sequential/interleaved,
        # interleavced/sequential, interleaved/interleaved
        for i, (rx, ry) in enumerate([(0, 1), (0, 2), (1, 3), (3, 3)]):
            r1 = R[rx]
            r2 = R[ry]
            for j in range(n_experiments):
                if flag:
                    proj1 = embed.fit_transform(r1[j])
                    proj2 = embed.transform(r2[j])
                    flag = False
                else:
                    proj1 = embed.transform(r1[j])
                    proj2 = embed.transform(r2[j])

                hull1 = ConvexHull(proj1)
                hull2 = ConvexHull(proj2)

                res, _ = jaccardIndexMC(hull1, hull2, n_samples=100_000)
                jacc[ii, i, j] = res

                res = convexHullIntersectionPoints(proj1, proj2)
                if res is None:
                    ints[ii, i, j] = 0
                else:
                    ints[ii, i, j] = len(res[1])

            print(f"Jaccard = {jacc[ii, i].mean(axis=0)}")
            tmp = (ints[ii, i].sum(axis=0))
            print(f"Intersection Points = {tmp}")

    np.save(output_dir+"jaccard_"+task+"_"+n_type, jacc)
    np.save(output_dir+"intersect_points_"+task+"_"+n_type, ints)
