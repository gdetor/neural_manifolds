import json
from tqdm import tqdm

import torch
from torch import nn

import numpy as np

from sklearn.datasets import make_circles, make_moons

from networks import NetRNN, NetMLP
from common import pickleObjectWrite, assignExperimentName
from common import hook_func, hook_func_ff
from common import convertXY2TorchDataset, activations


# activations = {}

device = torch.device("cpu")


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


def train(trainData,
          testData,
          name="./results/test_",
          epochs=50,
          n_neurons=10,
          sparsity=0.5,
          index=0,
          alpha=1.0,
          lr=1e-4,
          mode="sequential",
          n_type="mlp",
          linearly=False,
          mask=None):

    print(f"Sparsity = {sparsity}")
    if n_type == "rnn":
        net = NetRNN(input_size=1,
                     n_neurons=n_neurons,
                     output_size=2,
                     num_layers=1,
                     sparsity=sparsity,
                     alpha=alpha,
                     mask=mask,
                     sequence_len=1).to(device)
    elif n_type == "mlp":
        net = NetMLP(hidden_size=n_neurons,
                     sparsity=sparsity,
                     alpha=alpha,
                     mask=mask).to(device)
    else:
        raise ValueError("no network type")

    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  )
    criterion = nn.CrossEntropyLoss()

    weight = []
    tasks = []
    train_loss = []
    activities, inputs, labels, true_labels = [], [], [], []
    test_loss, test_accuracy = [], []
    for e in tqdm(range(epochs+1)):
        train_dataloader = trainData
        test_dataloader = testData
        task = "Circles"
        tasks.append(task)
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

        if n_type == "rnn":
            weight.append(net.rnn.weight_hh_l0.detach().cpu().numpy())
        if n_type == "mlp":
            weight.append(net.fc2.weight.detach().cpu().numpy())

    activities = np.array(activities)
    inputs = np.array(inputs)
    test_accuracy = np.array(test_accuracy)

    np.save(name+"_weights_"+str(index), np.array(weight))
    np.save(name+"_raw_activities_"+str(index), activities)
    np.save(name+"_inputs_"+str(index), inputs)
    np.save(name+"_train_loss_"+str(index), train_loss)
    np.save(name+"_test_loss_"+str(index), test_loss)
    np.save(name+"_test_accuracy_"+str(index), test_accuracy)

    pickleObjectWrite(labels,
                      name+"_raw_labels_"+str(index)+".pkl")
    pickleObjectWrite(true_labels,
                      name+"_true_labels_"+str(index)+".pkl")

    print(f"Accuracy = {test_accuracy.mean()}")


if __name__ == "__main__":
    limit = 3000
    with open("parameters.json") as f:
        params = json.load(f)

    n_type = "mlp"
    batch_size = params["batch_size"]
    n_experiments = params["n_experiments"]
    n_neurons = params["n_neurons"]
    lr = params["lrate"]
    epochs = params["epochs"]
    sparsity = params["sparsity"]
    alpha = params["alpha"]
    mode = "sequential"
    dir_ = params["directory"]

    data_type = "circles"
    if data_type == "cicles":
        X, Y = make_circles(
                n_samples=5000,
                noise=0.1,
                factor=0.3,
                random_state=0
                )
    else:
        data_type = "moons"
        X, Y = make_moons(
                n_samples=5000,
                noise=0.1,
                random_state=0
                )

    trainData, testData = convertXY2TorchDataset(
            X,
            Y,
            train_perc=0.6,
            batch_size=batch_size,
            test_batch_size=32)

    mask = torch.from_numpy(
            np.random.choice([0, 1],
                             size=(n_neurons, n_neurons),
                             p=[sparsity, 1-sparsity])
            )

    for experiment_id in range(n_experiments):
        exp_name = assignExperimentName(
                dir_+"_"+data_type+"_",
                mode,
                n_type,
                n_neurons,
                epochs,
                sparsity,
                experiment_id)

        res = train(
                trainData,
                testData,
                name=exp_name,
                epochs=epochs,
                n_neurons=n_neurons,
                sparsity=sparsity,
                index=experiment_id,
                alpha=alpha,
                lr=lr,
                mode=mode,
                n_type=n_type,
                mask=mask)
