import os
import pickle
import torch
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import griddata

from dataclasses import dataclass
from typing import Dict, List

activations = {}


@dataclass
class ExperimentConfig:
    exp_name: str
    experiment_id: int
    n_type: str = "mlp"
    mode: str = "sequential"
    dir_: str = "./results"
    lr: float = 1e-3
    n_neurons: int = 100
    batch_size: int = 256
    epochs: int = 10
    sparsity: float = 0.0
    alpha: float = 1.0
    freeze_output: bool = False
    init_params_name: str = "noname"


def pickleObjectWrite(X: Dict or List,
                      fname: str = "./results/activities.pkl"):
    """
    Dump a Python object to a Pickle file.
    """
    with open(fname, "wb") as f:
        pickle.dump(X, f)


def pickleObjectRead(fname: str = "./results/activities.pkl") -> Dict or List:
    """
    Load data from a Pickled file.
    """
    if os.path.getsize(fname) > 0:
        with open(fname, "rb") as f:
            content = pickle.load(f)
        return content
    else:
        print(f"empty file {fname}")


def accuracy(prediction: np.ndarray,
             target: np.ndarray,
             dummy_val: int = 100) -> int:
    """
    Counts equal elements in two numpy arrays taking into account differences
    in lengths.
    If the predictions array is longer than the target array, then it is
    truncated. If it is shorter then dummy values are added (they do not affect
    the counting).
    """

    n = len(prediction)     # prediction array length
    m = len(target)         # target array length

    if n > m:
        prediction = prediction[:m]

    if n < m:
        prediction = np.append(prediction, [dummy_val] * (m - n))

    counter = sum(prediction == target)
    return counter


def threshold(X: np.ndarray) -> np.ndarray:
    """
    Threshold a numpy array. Values larger or equal than 0.5 are round up to
    one, zero otherwise.
    """
    return (X >= 0.5) * 1. + (X < 0.5) * 0.


def taskSelector(epoch: int,
                 thr: int = 5,
                 mode: str = "seuqential") -> int:
    """ Selects a task based on the current epoch and a threshold based on the
    current mode (sequential io interleaved)

    Parameters
    ----------
    epoch   : int, current training epoch
    thr     : int, threshold epoch, after which we switch tasks
    mode    : str, determines if we are on a sequential or interleaved mode

    Returns
    -------
    int : The number of epoch after which we switch tasks

    """
    if mode == "sequential":
        return epoch > thr
    else:
        return (epoch % 2) == 0


def hook_func(module,
              input: torch.Tensor,
              output: torch.Tensor):
    """ Pytorch forward hook function for the RNN network. It gets the
    preactivation of recurrent units.

    Parameters
    ----------
    module:     Pytorch module
    input:      torch tensor - layers input
    output:     torch tensor - layer's neurons preactivations
    """
    out, _ = output
    activations["z"] = out[:, -1, :].detach()


def hook_func_ff(module,
                 input: torch.Tensor,
                 output: torch.Tensor):
    """ Pytorch forward hook function for the MLP network. It gets the
    preactivation of linear units.

    Parameters
    ----------
    module:     Pytorch module
    input:      torch tensor - layers input
    output:     torch tensor - layer's neurons preactivations
    """
    activations["z"] = output.detach()


def assignExperimentName(directory: str,
                         mode: str,
                         n_type: str,
                         n_neurons: int,
                         epochs: int,
                         sparsity: float,
                         index: int):
    """
    Generates the core name for all the files related to one experiment.

    Parameters
    ----------
    mode    : str, mode of training either sequential or interleaved
    n_type  : int, neural network type (mlp or rnn)
    n_neurosn   : int, number of neurons in the layer of interest
    epochs  : int, number of epochs
    index   : int, current time step

    Returns
    -------
    str, the body of the name for storing all the files related to an
    experiment
    """
    return directory+"_"+mode+"_"+n_type+"_"+str(index)+"_neurons_" + \
        str(n_neurons) + "_epochs_"+str(epochs)+"_sparsity_"\
        + str(int(sparsity*10))


def convertXY2TorchDataset(X: np.ndarray,
                           Y: np.ndarray,
                           train_perc: float = 0.6,
                           batch_size: int = 32,
                           test_batch_size: int = 32):
    """
    Converts datasets that are stored in numpy arrays to torch tensors,
    and performs a train/test split.

    Parameters
    ----------
    X   : numpy array, contains the data
    Y   : numpy array, contains the labels
    train_perc : float, the percentage of data that will be used for training
    batch_size  : int, train batch size
    test_batch_size : int, test batch size

    Returns
    -------
    torch dataloaders. a train and a test dataloader objects
    """
    n_train_samples = int(len(X) * train_perc)

    X_ = torch.from_numpy(X.astype(np.float32))
    Y_ = torch.from_numpy(Y)
    dataset1 = torch.utils.data.TensorDataset(
            X_[:n_train_samples],
            Y_[:n_train_samples])
    dataset2 = torch.utils.data.TensorDataset(
            X_[n_train_samples:],
            Y_[n_train_samples:])

    train_dataloader = torch.utils.data.DataLoader(dataset1,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset2,
                                                  batch_size=test_batch_size,
                                                  shuffle=True,
                                                  drop_last=True)
    return train_dataloader, test_dataloader


def loadData(dir_: str = "./results/",
             task: str = "toy",
             mode: str = "sequential",
             n_type: str = "mlp",
             sparsity: float = 0.0,
             n_experiments: int = 5,
             epochs: int = 100,
             n_neurons: int = 100) -> tuple[np.ndarray, ...]:
    """
    Loads the data (neurons preactivations and networks responses) from files.
    It stores the data into numpy arrays and returns a tuple that contains the
    responses/targets per class for each task.

    Parameters:
    -----------
    dir_    : str, the directory where the data are stored
    task    : str, the type of task (toy/mnist)
    mode    : str, sequential or interleaved learning schedule
    n_type  : str, neural network type (rnn or mlp)
    sparsity: float, connectivity sparsity (is being used in the name)
    epochs  : int, total number of training epochs
    n_neurons: int, total number of neurons from which we record

    Returns:
    --------
    tuple[numpy arrays], the neurons preactivations and network outputs to
    subtask A and subtask B.
    """

    # R holds the preactivations, T holds the y_hat (network responses)
    R1, T1 = [], []
    R2, T2 = [], []
    for i in range(n_experiments):
        exp_name = assignExperimentName(
                dir_,
                mode,
                n_type,
                n_neurons,
                epochs,
                sparsity,
                i)

        activities = np.load(
                exp_name+"_test_activities_"+str(i)+".npy"
                )
        # Task A & Taks B
        _, p, q, s = activities.shape
        R1.append(activities[0].reshape(p*q, s))
        R2.append(activities[8].reshape(p*q, s))

        targets = np.load(exp_name+"_test_labels_"+str(i)+".npy")
        _, p, q = targets.shape
        T1.append(targets[0].reshape(p*q, 1))
        T2.append(targets[8].reshape(p*q, 1))

    R1 = np.array(R1)
    T1 = np.array(T1).astype('i')

    R2 = np.array(R2)
    T2 = np.array(T2).astype('i')

    return R1, T1, R2, T2


def plot3DSurface(X, Y):
    """
    Interpolates two numpy arrays of shape (N, 3) and plots 3D surfaces.
    Useful when one wants to visualize neural representations (PCAs)

    Parameters:
    -----------
    X   : numpy array of shape (N, 3)
    Y   : numpy array of shape (N, 3)

    Returns:
    --------
    Void
    """
    def interpolate(data):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        x_grid = np.linspace(x.min(), x.max(), 100)
        y_grid = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z = griddata((x, y), z, (X, Y), method="cubic")
        return X, Y, Z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X1, Y1, Z1 = interpolate(X)
    ax.plot_surface(X1, Y1, Z1, cmap="Reds", edgecolor="none")
    X2, Y2, Z2 = interpolate(Y)
    ax.plot_surface(X2, Y2, Z2, cmap="Blues", edgecolor="none")


def bringDirectoryToLife(directory_path):
    # Check if the path does not exist
    if not os.path.exists(directory_path):
        # Create the directory and any missing parent directories
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
