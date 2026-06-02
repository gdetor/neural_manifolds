import json
from tqdm import tqdm

import torch
from torch import nn
import torch.multiprocessing as mp
from torchvision import datasets, transforms

import numpy as np

from networks import NetRNNMNIST, NetMLPMNIST
from common import ExperimentConfig
from common import pickleObjectWrite, assignExperimentName
from common import hook_func, hook_func_ff, taskSelector, activations

device = torch.device("cuda:0")

TEST_BATCH_SIZE = 32


def initializeNStoreParameters(name="noname",
                               n_neurons=100,
                               sparsity=0.0,
                               n_type="mlp",
                               store_mask=False):

    mask = np.random.choice([0, 1],
                            size=(n_neurons, n_neurons),
                            p=[sparsity, 1-sparsity])
    np.save(name+"_"+n_type+"_mask", mask)
    mask = torch.from_numpy(mask)

    if n_type == "rnn":
        net = NetRNNMNIST(
                input_size=784,
                n_neurons=n_neurons,
                output_size=10,
                num_layers=1,
                sparsity=sparsity,
                alpha=1.0,
                mask=mask,
                sequence_len=1)
    elif n_type == "mlp":
        net = NetMLPMNIST(
                hidden_size=n_neurons,
                sparsity=sparsity,
                alpha=1.0,
                mask=mask)
    torch.save(net, name+"_initial_parameters.pt")


def test(model, criterion, dataloader, device, n_type="mlp"):
    if n_type == "rnn":
        handler = model.rnn.register_forward_hook(hook_func)
    elif n_type == "mlp":
        handler = model.fc4.register_forward_hook(hook_func_ff)
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
    mode = cfg.mode
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
    init_params_fname = cfg.init_params_name

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

    train_mnist = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data/",
                train=True,
                download=True,
                transform=transform),
            **train_kwargs
            )

    test_mnist = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data/",
                train=False,
                transform=transform),
            **test_kwargs
            )

    train_fmnist = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                "../data/",
                train=True,
                download=True,
                transform=transform),
            **train_kwargs
            )
    test_fmnist = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                "../data/",
                train=False,
                transform=transform),
            **test_kwargs
            )

    print(f"Sparsity = {sparsity}")
    print(f"Neural Network Type: {n_type}")

    mask = torch.from_numpy(np.load(init_params_fname+"_"+n_type+"_mask.npy"))
    if n_type == "rnn":
        net = NetRNNMNIST(
                input_size=784,
                n_neurons=n_neurons,
                output_size=10,
                num_layers=1,
                sparsity=sparsity,
                alpha=alpha,
                mask=mask,
                sequence_len=1)
    elif n_type == "mlp":
        net = NetMLPMNIST(
                hidden_size=n_neurons,
                sparsity=sparsity,
                alpha=alpha,
                mask=mask)
    else:
        raise ValueError("no network type")

    net = torch.load(init_params_fname+"_initial_parameters.pt",
                     weights_only=False)
    net = net.to(device)

    if freeze_output:
        print("Freezing parameters of output layer")
        net.fc_out.weight.requires_grad = False
        net.fc_out.bias.requires_grad = False

    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  )
    criterion = nn.CrossEntropyLoss()

    weight = []
    tasks = []
    train_loss = []
    activities, inputs, labels, true_labels = [], [], [], []
    test_loss, test_accuracy, test_accuracy1, test_accuracy2 = [], [], [], []
    for e in tqdm(range(epochs+1)):
        # if e >= (epochs+1) // 2:
        if taskSelector(e, thr=epochs//2, mode=mode):
            train_dataloader = train_mnist
            test_dataloader = test_mnist
            task = "MNIST"
        else:
            train_dataloader = train_fmnist
            test_dataloader = test_fmnist
            task = "F-MNIST"
        tasks.append(task)
        if e == 0:
            _, _, activity, input, label, tl = test(
                    net,
                    criterion,
                    test_dataloader,
                    device,
                    n_type=n_type)
            activities.append(activity)
            labels.append(label)
            inputs.append(input)
            true_labels.append(tl)
        else:
            running_loss = []
            net.train()
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                yhat, _ = net(x)

                # loss = (1.0 / alpha**2) * criterion(yhat, y)
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

                _, accuracy, _, _, _, _ = test(
                        net,
                        criterion,
                        test_fmnist,
                        device,
                        n_type=n_type)
                test_accuracy1.append(accuracy)

                _, accuracy, _, _, _, _ = test(
                        net,
                        criterion,
                        test_mnist,
                        device,
                        n_type=n_type)
                test_accuracy2.append(accuracy)
        if n_type == "rnn":
            weight.append(net.rnn.weight_hh_l0.detach().cpu().numpy())
        if n_type == "mlp":
            weight.append(net.fc2.weight.detach().cpu().numpy())

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
    np.save(name+"_test_accuracy1_"+str(index), test_accuracy1)
    np.save(name+"_test_accuracy2_"+str(index), test_accuracy2)

    np.save(name+"_weights_"+str(index), np.array(weight))

    test_tasks = []
    test_inputs = []
    accuracy_trace = []
    test_activities = []
    test_labels = []
    with torch.random.fork_rng():
        torch.manual_seed(13)
        for i in range(10):
            if i > 4:
                data = test_mnist
                task = "MNIST"
            else:
                data = test_fmnist
                task = "F-MNIST"
            test_tasks.append(task)
            _, accuracy, activity, inputs, labels, _ = test(
                    net,
                    criterion,
                    data,
                    device,
                    n_type=n_type)
            accuracy_trace.append(accuracy)
            test_activities.append(activity)
            test_labels.append(labels)
            test_inputs.append(inputs)

    np.save(name+"_post_train_accuracy_"+str(index),
            np.array(accuracy_trace))
    np.save(name+"_test_activities_"+str(index), np.array(test_activities))
    np.save(name+"_test_labels_"+str(index), np.array(test_labels))
    np.save(name+"_test_inputs_"+str(index), np.array(test_inputs))

    ts = [tasks, test_tasks]
    pickleObjectWrite(ts,
                      name+"_tasks_test_sequence_"+str(index)+".pkl")

    result = {"experiment_id": index,
              "n_type": n_type,
              "mode": mode,
              "sparsity": sparsity,
              "test accuracy": test_accuracy}
    result_queue.put(result)


if __name__ == "__main__":
    with open("parameters_mnist.json") as f:
        params = json.load(f)

    configs = []
    for experiment_id in range(params["n_experiments"]):
        exp_name = assignExperimentName(
                params["directory"]+"_"+params["data_type"],
                params["mode"],
                params["n_type"],
                params["n_neurons"],
                params["epochs"],
                params["sparsity"],
                experiment_id)

        if params["n_type"] == "mlp":
            name = exp_name.replace("mlp_"+str(experiment_id)+"_neurons",
                                    "mlp_neurons")
        else:
            name = exp_name.replace("rnn_"+str(experiment_id)+"_neurons",
                                    "rnn_neurons")
        if experiment_id == 0:
            initializeNStoreParameters(name=name,
                                       n_neurons=params["n_neurons"],
                                       sparsity=params["sparsity"],
                                       n_type=params["n_type"])

        cfg = ExperimentConfig(
                exp_name=exp_name,
                data_type=params["data_type"],
                experiment_id=experiment_id,
                n_type=params["n_type"],
                mode=params["mode"],
                dir_=params["directory"],
                lr=params["lrate"],
                n_neurons=params["n_neurons"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                sparsity=params["sparsity"],
                alpha=params["alpha"],
                freeze_output=params["freeze_output"],
                init_params_name=name
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
