import json
import numpy as np

from scipy.stats import mannwhitneyu

from metrics import estimateSparseness
from metrics import estimateParticipationRatio

from common import assignExperimentName

# np.random.seed(13)


def computePR_SP(X):
    n_experiments = X.shape[0]
    pr = np.zeros((n_experiments,))
    sp = np.zeros((n_experiments, n_neurons))
    for i, x in enumerate(X):
        pr[i] = estimateParticipationRatio(x, neural_activity=True)
        sp[i] = estimateSparseness(x)
    return pr, sp


if __name__ == "__main__":
    # -------------------------------------------
    # Change the following based on the problem
    # Choose a task: toy (circles/moons) or mnist
    data_type = "mnist"
    n_type = "mlp"
    # Load the parameters of the experiment
    if data_type in ["toy", "circles", "moons"]:
        with open("parameters.json") as f:
            params = json.load(f)
    else:
        with open("parameters_mnist.json") as f:
            params = json.load(f)

    batch_size = params["batch_size"]
    n_experiments = params["n_experiments"]
    n_neurons = params["n_neurons"]
    lr = params["lrate"]
    epochs = params["epochs"]
    alpha = params["alpha"]
    mode = params["mode"]
    dir_ = params["directory"]
    # ----------------------------------------

    # sparsity = [0.0, 0.3, 0.5, 0.8]
    sparsity = [0.0, 0.5]

    # Load the neural activities from pickled files
    seq_pr, int_pr = [], []
    seq_sp, int_sp = [], []
    for mode in ["sequential", "interleaved"]:
        for sp in sparsity:
            acc1, acc2 = [], []
            R, T = [], []
            for i in range(n_experiments):
                exp_name = assignExperimentName(
                        directory=dir_+"_"+data_type,
                        mode=mode,
                        n_type=n_type,
                        n_neurons=n_neurons,
                        epochs=epochs,
                        sparsity=sp,
                        index=i)

                activities = np.load(
                        exp_name+"_test_activities_"+str(i)+".npy"
                        )

                _, p, q, s = activities.shape
                tmp_r = np.vstack([activities[0].reshape(p*q, s),
                                   activities[8].reshape(p*q, s)])
                R.append(tmp_r)

                targets = np.load(exp_name+"_test_labels_"+str(i)+".npy")
                _, p, q = targets.shape
                tmp_t = np.vstack([targets[0].reshape(p*q, 1),
                                   targets[8].reshape(p*q, 1)])
                T.append(tmp_t)

                acc1.append(np.load(exp_name+"_test_accuracy1_"+str(i)+".npy"))
                acc2.append(np.load(exp_name+"_test_accuracy2_"+str(i)+".npy"))

            R = np.array(R)
            T = np.array(T).astype('i')[:, :, 0]
            n_samples = R.shape[1]

            acc1 = np.array(acc1)
            acc2 = np.array(acc2)

            # ------------------------------------------------------------
            #
            # Compute Participation Ratio & Sparseness
            # ------------------------------------------------------------
            pratio, sparse = computePR_SP(R)
            sparse = sparse.mean(axis=1)

            print(f"Mean PR: {np.round(pratio.mean(),
                  3)}/{np.round(pratio.std(), 3)}")
            print(f"Mean SP: {np.round(sparse.mean(),
                  3)}/{np.round(sparse.std(), 3)}")

            if mode == "sequential":
                print(f"Sequential, Sparsity = {sp}")
                seq_pr.append(pratio)
                seq_sp.append(sparse)
            else:
                print(f"Interleaved, Sparsity = {sp}")
                int_pr.append(pratio)
                int_sp.append(sparse)

    seq_pr = np.array(seq_pr)
    int_pr = np.array(int_pr)

    seq_sp = np.array(seq_sp)
    int_sp = np.array(int_sp)

    def statistic(x, y, axis):
        # return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        return np.median(x, axis=axis) - np.median(y, axis=axis)

    for i in range(len(sparsity)):
        print("*" * 40)
        print(f" >>>>>>>>>>>>>>> Experiment #{i}")
        res = mannwhitneyu(seq_pr[i], int_pr[i])
        print(f"PR Mann Whitney U test (seq-int): {res.pvalue}")
        res = mannwhitneyu(seq_sp[i], int_sp[i])
        print(f"SP Mann Whitney U test (seq-int): {res.pvalue}")
        print(" ")
        print(" ")
