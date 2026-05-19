import json
from tqdm import tqdm

import torch
from torch import nn
from torchvision import datasets, transforms

import numpy as np

from networks import NetRNNMNIST, NetMLPMNIST
from common import pickleObjectWrite, assignExperimentName
from common import hook_func, hook_func_ff, activations


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


def train(train_dataloader,
          test_dataloader,
          name="./results/test_",
          epochs=50,
          n_neurons=10,
          sparsity=0.5,
          index=0,
          alpha=1.0,
          lr=1e-4,
          n_type="mlp",
          mask=None):

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

    pickleObjectWrite(activities,
                      name+"_raw_activities_"+str(index)+".pkl")
    pickleObjectWrite(inputs,
                      name+"_inputs_"+str(index)+".pkl")
    pickleObjectWrite(labels,
                      name+"_raw_labels_"+str(index)+".pkl")
    pickleObjectWrite(true_labels,
                      name+"_true_labels_"+str(index)+".pkl")

    np.save(name+"_train_loss_"+str(index), train_loss)
    np.save(name+"_test_loss_"+str(index), test_loss)
    np.save(name+"_test_accuracy_"+str(index), test_accuracy)


if __name__ == "__main__":
    with open("parameters_mnist.json") as f:
        params = json.load(f)

    n_type = params["n_type"]
    batch_size = params["batch_size"]
    n_experiments = params["n_experiments"]
    n_neurons = params["n_neurons"]
    lr = params["lrate"]
    epochs = params["epochs"]
    sparsity = 0.0
    alpha = params["alpha"]
    mode = "sequential"
    dir_ = params["directory"]

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

        X = datasets.FashionMNIST(
                    "../data/",
                    train=True,
                    download=True,
                    transform=transform)
        Y = datasets.FashionMNIST(
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

    for experiment_id in range(n_experiments):
        exp_name = assignExperimentName(
                dir_+"_"+data_type,
                "sequential",
                n_type,
                n_neurons,
                epochs,
                sparsity,
                experiment_id)

        res = train(
                train_dataloader,
                test_dataloader,
                name=exp_name,
                epochs=epochs,
                n_neurons=n_neurons,
                sparsity=sparsity,
                index=experiment_id,
                alpha=alpha,
                lr=lr,
                n_type=n_type,
                mask=mask)
