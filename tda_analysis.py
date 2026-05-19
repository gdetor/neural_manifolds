import json
from multiprocess import Pool

import numpy as np
import gudhi as gd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from common import loadData

eps = np.finfo(np.float32).eps

# np.random.seed(13)


def persistence_entropy(dgm: np.ndarray,
                        top_frac: float = 0.1) -> tuple[np.ndarray, ...]:
    """
    Persistence entropy weighting. It weights each point by its contribution to
    the persistence entropy:
        H = -sum_i p_i * log(p_i),   p_i = lifetime_i / total_lifetime
    Returns top fraction by entropy contribution.

    Parameters:
    -----------
    dgm         : np.ndarray, the persistence diagram pairs, (birth, death)
    top_grac    : float, the percentage of top fraction pairs to return
    by entropy contribution

    Returns:
    --------
    tuple, PD with most persistent points, dgm with least significant points,
    and the contribution of each point
    """
    lifetimes = dgm[:, 1] - dgm[:, 0]
    total = lifetimes.sum()
    p = lifetimes / (total + 1e-6)
    # entropy contribution per point (nats)
    contrib = -p * np.log(p + 1e-12)
    # keep top fraction
    cutoff = np.quantile(contrib, 1 - top_frac)
    mask = contrib >= cutoff
    return dgm[mask], dgm[~mask], contrib


def clip_diagram(dgm: np.ndarray, percentage: float = 0.1) -> np.ndarray:
    """ Clips the points (birth, death) on a persistence diagram (PD).
    Essentialy, it replaces the inf values with a value that's X% higher than
    the finite highest value of the PD.

    Parameters
    ----------
    dgm     : numpy array, contains the pairs (birth, death) of a persistence
    diagram
    percentage  : float, the percentage of the highest finite value in the PD
    that will be used to clip the PD

    Returns
    -------
    numpy array,    the clipped persistence diagram
    """
    assert (percentage >= 0) and (percentage <= 1)

    finite = dgm[np.isfinite(dgm[:, 1])]
    max_val = finite[:, 1].max() * (1. + percentage)
    dgm = dgm.copy()
    dgm[~np.isfinite(dgm[:, 1]), 1] = max_val
    return dgm


def parallel_permutation_test(sample1: np.ndarray,
                              sample2: np.ndarray,
                              n_resamples: int = 10_0000,
                              n_jobs: int = 10) -> tuple[float, float]:
    """
    Performs a permutation test on two samples in parallel and returns the
    observed distance and the p-value.

    Parameters:
    -----------
    sample1     : ndarray, the first persistence diagram
    sample2     : ndarray, the second persistence diagram
    n_resamples : int, how many resamples the test will use
    n_jobns     : int, how many threads to use to run in parallel

    Returns:
    --------
    tuple[float, float], the observed bottleneck distance, and its p-value
    """
    d_obs = gd.bottleneck_distance(sample1, sample2)

    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)

    def compute_single_bottleneck(combined, n1):
        shuffled = np.random.permutation(combined)
        resample1 = shuffled[:n1]
        resample2 = shuffled[n1:]
        return gd.bottleneck_distance(resample1, resample2)

    args = [(combined, n1) for _ in range(n_resamples)]

    with Pool(n_jobs) as pool:
        null_distances = pool.starmap(compute_single_bottleneck, args)

    p_value = np.sum(np.array(null_distances) >= d_obs) / n_resamples

    return d_obs, p_value


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

    output_dir = "./tda_results/"

    # We test for the following connectivity sparsity conditions
    # sparsity = [0.0, 0.3]
    sparsity = [0.0]

    # --------------------------------------------------------------------
    # Initialize all the necesary Numpy arrays
    # Bottleneck distances for H0 and H1 groups
    bottle_0 = np.zeros((len(sparsity), 4, n_experiments))
    bottle_1 = np.zeros((len(sparsity), 4, n_experiments))

    # p-values for H0 and H1 groups
    pvals_0 = np.zeros((len(sparsity), 4, n_experiments))
    pvals_1 = np.zeros((len(sparsity), 4, n_experiments))

    # Run the TDA analysis for each sparsity value
    for k, sp in enumerate(sparsity):
        print("--" * 60)
        print(f" >>>>>>>>>>>> SPARSITY : {sp}")
        print(f" >>>>>>>>>>>> NETWORK  : {n_type}")

        # Load the data for the SEQUENTIAL learning schedule
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
            # Run over experiments
            for j in range(n_experiments):
                # Compute the PCAs using R[0] as reference
                if flag:
                    proj1 = embed.fit_transform(r1[j])
                    proj2 = embed.transform(r2[j])
                    flag = False
                else:
                    proj1 = embed.transform(r1[j])
                    proj2 = embed.transform(r2[j])

                # Compute the Alpha complex and the Persistence Diagram
                # Projections 1
                alpha = gd.AlphaComplex(points=proj1)
                simplex_tree = alpha.create_simplex_tree()
                dgms = simplex_tree.persistence()

                # Homology group H0 - Clip/Choose most persistent points
                hA0 = simplex_tree.persistence_intervals_in_dimension(0)
                hA0 = clip_diagram(hA0)
                hA0, _, _ = persistence_entropy(hA0, top_frac=0.3)
                # hA0 = [(b, d) for b, d in hA0 if (d > b + threshold)]

                # Homology group H1 - Clip/Choose most persistent points
                hA1 = simplex_tree.persistence_intervals_in_dimension(1)
                hA1 = clip_diagram(hA1)
                hA1, _, _ = persistence_entropy(hA1, top_frac=0.3)

                # Compute the Alpha complex and the Persistence Diagram
                # Projections 1
                alpha = gd.AlphaComplex(points=proj2)
                simplex_tree = alpha.create_simplex_tree()
                dgms = simplex_tree.persistence()

                # Homology group H0 - Clip/Choose most persistent points
                hB0 = simplex_tree.persistence_intervals_in_dimension(0)
                hB0 = clip_diagram(hB0)
                hB0, _, _ = persistence_entropy(hB0, top_frac=0.3)

                # Homology group H1 - Clip/Choose most persistent points
                hB1 = simplex_tree.persistence_intervals_in_dimension(1)
                hB1 = clip_diagram(hB1)
                hB1, _, _ = persistence_entropy(hB1, top_frac=0.3)

                # Compute the bottleneck distances and the p-values
                # bdistH0 = gd.bottleneck_distance(hA0, hB0)
                pvalue, bdistH0 = parallel_permutation_test(
                        hA0,
                        hB0,
                        n_resamples=1000,
                        n_jobs=10)
                bottle_0[k, i, j] = bdistH0
                pvals_0[k, i, j] = pvalue

                # bdistH1 = gd.bottleneck_distance(hA1, hB1)
                pvalue, bdistH1 = parallel_permutation_test(
                        hA1,
                        hB1,
                        n_resamples=1000,
                        n_jobs=10)
                bottle_1[k, i, j] = bdistH1
                pvals_1[k, i, j] = pvalue
                # gd.plot_persistence_diagram(dgms)

    # Store all the data for further statistical analysis
    np.save(output_dir+"pvals0"+"_"+task+"_"+n_type, pvals_0)
    np.save(output_dir+"pvals1"+"_"+task+"_"+n_type, pvals_1)
    np.save(output_dir+"bdist0"+"_"+task+"_"+n_type, bottle_0)
    np.save(output_dir+"bdist1"+"_"+task+"_"+n_type, bottle_1)
