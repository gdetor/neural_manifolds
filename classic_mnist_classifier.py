import json
from tqdm import tqdm

import torch
from torch import nn
import torch.multiprocessing as mp
from torchvision import datasets, transforms

import numpy as np

from networks import NetRNNMNIST, NetMLPMNIST
from common import pickleObjectWrite, assignExperimentName
from common import hook_func, hook_func_ff, activations
from common import ExperimentConfig


device = torch.device("cuda:0")

TEST_BATCH_SIZE = 32


def test(model, criterion, dataloader, device, n_type="mlp"):
    if n_type == "rnn":
        handler = model.rnn.register_forward_hook(hook_func)
    elif n_type == "mlp":
        handler = model.fc2.register_forward_hook(hook_func_ff)
    else:
        raise ValueError("network type not found")
    model.eval()
    loss = 0.0
    accuracy = 0.0
    activity = []
    labels = []
    inputs = []
    true_labels = []
    for x, y in dataloader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            yhat, _ = model(x)
            activity.append(activations["z"].detach().cpu().numpy())

        loss += criterion(yhat, y).item()
        accuracy += yhat.argmax(dim=1).eq(y).sum().item()
        labels.append(yhat.argmax(dim=1).detach().cpu().numpy())
        true_labels.append(y.detach().cpu().numpy())
        inputs.append(x.detach().cpu().numpy())
    loss /= (32 * len(dataloader))
    accuracy /= (32 * len(dataloader))
    handler.remove()
    return loss, accuracy, activity, inputs, labels, true_labels


def train(cfg: ExperimentConfig, result_queue: mp.Queue):

    name = cfg.exp_name
    epochs = cfg.epochs
    n_neurons = cfg.n_neurons
    sparsity = cfg.sparsity
    index = cfg.experiment_id
    alpha = cfg.alpha
    lr = cfg.lr
    n_type = cfg.n_type
    batch_size = cfg.batch_size
    freeze_output = cfg.freeze_output

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}

    use_cuda = True
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    data_type = "mnist"
    if data_type == "mnist":
        X_train = datasets.MNIST(
                "../data/",
                train=True,
                download=True,
                transform=transform)

        X_test = datasets.MNIST(
                "../data/",
                train=False,
                transform=transform)
    else:
        data_type = "fmnist"

        X_train = datasets.FashionMNIST(
                    "../data/",
                    train=True,
                    download=True,
                    transform=transform)
        X_test = datasets.FashionMNIST(
                    "../data/",
                    train=False,
                    transform=transform)

    train_dataloader = torch.utils.data.DataLoader(X_train, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(X_test, **test_kwargs)

    mask = torch.from_numpy(
            np.random.choice([0, 1],
                             size=(n_neurons, n_neurons),
                             p=[sparsity, 1-sparsity])
            )

    print(f"Sparsity = {sparsity}")
    if n_type == "rnn":
        net = NetRNNMNIST(
                input_size=784,
                n_neurons=n_neurons,
                output_size=10,
                num_layers=1,
                sparsity=sparsity,
                alpha=alpha,
                mask=mask,
                sequence_len=1).to(device)
    elif n_type == "mlp":
        net = NetMLPMNIST(
                hidden_size=n_neurons,
                sparsity=sparsity,
                alpha=alpha,
                mask=mask).to(device)
    else:
        raise ValueError("no network type")

    if freeze_output:
        print("Freezing parameters of output layer")
        net.fc_out.weight.requires_grad = False
        net.fc_out.bias.requires_grad = False
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  )
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    activities, inputs, labels, true_labels = [], [], [], []
    test_loss, test_accuracy = [], []
    for e in tqdm(range(epochs+1)):
        if e == 0:
            _, _, activity, input, label, true_labels = test(
                    net,
                    criterion,
                    test_dataloader,
                    device,
                    n_type=n_type)
            activities.append(activity)
            labels.append(label)
            inputs.append(input)
            true_labels.append(true_labels)
        else:
            running_loss = []
            net.train()
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                yhat, _ = net(x)

                loss = criterion(yhat, y)
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())

            train_loss.append(np.average(running_loss))

            if e % 1 == 0:
                loss, accuracy, activity, input, label, tl = test(
                        net,
                        criterion,
                        test_dataloader,
                        device,
                        n_type=n_type)
                test_loss.append(loss)
                test_accuracy.append(accuracy)
                activities.append(activity)
                labels.append(label)
                inputs.append(input)
                true_labels.append(tl)

    activities = np.array(activities)
    inputs = np.array(inputs)

    np.save(name+"_raw_activities_"+str(index), activities)
    np.save(name+"_inputs_"+str(index), inputs)
    pickleObjectWrite(labels,
                      name+"_raw_labels_"+str(index)+".pkl")
    pickleObjectWrite(true_labels,
                      name+"_true_labels_"+str(index)+".pkl")

    np.save(name+"_train_loss_"+str(index), train_loss)
    np.save(name+"_test_loss_"+str(index), test_loss)
    np.save(name+"_test_accuracy_"+str(index), test_accuracy)

    result = {"experiment_id": index,
              "test accuracy": test_accuracy}
    result_queue.put(result)


if __name__ == "__main__":
    with open("parameters_mnist.json") as f:
        params = json.load(f)

    configs = []
    for experiment_id in range(params["n_experiments"]):
        exp_name = assignExperimentName(
                params["directory"]+"_"+params["data_type"],
                "sequential",
                params["n_type"],
                params["n_neurons"],
                params["epochs"],
                params["sparsity"],
                experiment_id)

        cfg = ExperimentConfig(
                exp_name=exp_name,
                experiment_id=experiment_id,
                n_type=params["n_type"],
                mode="sequential",
                dir_=params["directory"],
                lr=params["lrate"],
                n_neurons=params["n_neurons"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                sparsity=params["sparsity"],
                alpha=params["alpha"],
                freeze_output=params["freeze_output"]
                )
        configs.append(cfg)

    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue = mp.Queue()

    processes = []
    # wall_start = time.time()
    for cfg in configs:
        p = mp.Process(target=train, args=(cfg, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # wall_elapsed = time.time() - wall_start

    # ---- Collect & display results ----
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda r: r["experiment_id"])

    print(results)
