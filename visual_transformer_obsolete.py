import json
import numpy as np
# import matplotlib.pylab as plt

import torch
from torch import nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from common import assignExperimentName

device = torch.device("cuda:0")

activations = {}


def hook_fn(module, input, output):
    """
    input  → tuple of tensors BEFORE the layer runs  (pre-activations)
    output → tensor AFTER the layer runs              (post-activations)
    """
    activations["pre"] = input[0].detach()
    activations["post"] = output.detach()


def RandomizeParameters(model):
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


num_workers = {"train": 5, "val": 0, "test": 0}
data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]
            ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]
            ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]
            )
        }

TASK_A_CLASSES = [4, 30, 55, 72, 95, 1, 32, 67, 73, 91]  # aquatic animals
TASK_B_CLASSES = [8, 13, 48, 58, 90, 41, 69, 81, 85, 89]  # vehicles

full_train_dataset = datasets.CIFAR100(root="./data/",
                                       train=True,
                                       download=True,
                                       transform=data_transforms["train"])

full_test_dataset = datasets.CIFAR100(root="./data/",
                                      train=False,
                                      download=True,
                                      transform=data_transforms["test"])


def get_subset_indices(dataset, target_classes):
    targets = torch.tensor(dataset.targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for cls in target_classes:
        mask |= (targets == cls)
    return torch.where(mask)[0].tolist()


trainA_indices = get_subset_indices(full_train_dataset, TASK_A_CLASSES)
testA_indices = get_subset_indices(full_test_dataset, TASK_A_CLASSES)
subsetA_train = torch.utils.data.Subset(full_train_dataset, trainA_indices)
subsetA_test = torch.utils.data.Subset(full_test_dataset, testA_indices)

original_labels = [full_train_dataset.targets[i] for i in
                   subsetA_train.indices]
unique_labels = sorted(set(original_labels))
label_mappingA = {old: new for new, old in enumerate(unique_labels)}

for i in subsetA_train.indices:
    full_train_dataset.targets[i] = label_mappingA[full_train_dataset.targets[i]]

original_labels = [full_test_dataset.targets[i] for i in
                   subsetA_test.indices]
unique_labels = sorted(set(original_labels))
label_mappingA = {old: new for new, old in enumerate(unique_labels)}

for i in subsetA_test.indices:
    full_test_dataset.targets[i] = label_mappingA[full_test_dataset.targets[i]]

trainloaderA = data.DataLoader(subsetA_train,
                               batch_size=100,
                               shuffle=True,
                               num_workers=num_workers["train"])

testloaderA = data.DataLoader(subsetA_test,
                              batch_size=100,
                              shuffle=True,
                              num_workers=num_workers["test"])

trainB_indices = get_subset_indices(full_train_dataset, TASK_B_CLASSES)
testB_indices = get_subset_indices(full_test_dataset, TASK_B_CLASSES)
subsetB_train = torch.utils.data.Subset(full_train_dataset, trainB_indices)
subsetB_test = torch.utils.data.Subset(full_test_dataset, testB_indices)

original_labels = [full_train_dataset.targets[i] for i in
                   subsetB_train.indices]
unique_labels = sorted(set(original_labels))
label_mappingB = {old: new for new, old in enumerate(unique_labels)}

for i in subsetB_train.indices:
    full_train_dataset.targets[i] = label_mappingB[full_train_dataset.targets[i]]

original_labels = [full_test_dataset.targets[i] for i in
                   subsetB_test.indices]
unique_labels = sorted(set(original_labels))
label_mappingB = {old: new for new, old in enumerate(unique_labels)}

for i in subsetB_test.indices:
    full_test_dataset.targets[i] = label_mappingB[full_test_dataset.targets[i]]

trainloaderB = data.DataLoader(subsetB_train,
                               batch_size=100,
                               shuffle=True,
                               num_workers=num_workers["train"])

testloaderB = data.DataLoader(subsetB_test,
                              batch_size=100,
                              shuffle=True,
                              num_workers=num_workers["test"])

with open("parameters_cifar100.json") as f:
    params = json.load(f)
epochs = params["epochs"]

weights = models.ViT_B_16_Weights.DEFAULT
# net = models.vit_b_16(weights=weights)
net = models.vit_b_16(weights=None)
in_features = net.heads.head.in_features
net.heads.head = nn.Linear(in_features, 10)
net.to(device)

for experiment_id in range(params["n_experiments"]):

    RandomizeParameters(net)

    print(f"Running epxeriment: {experiment_id+1}")
    exp_name = assignExperimentName(
            params["directory"]+"_"+params["data_type"],
            params["mode"],
            params["n_type"],
            params["n_neurons"],
            params["epochs"],
            params["sparsity"],
            experiment_id)

    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=1e-3,
                                end_factor=1.0,
                                total_iters=10)
    decay_scheduler = CosineAnnealingLR(optimizer,
                                        T_max=(epochs-10),
                                        eta_min=1e-6
                                        )

    train_loss, test_loss = [], []
    test_accuracy = []
    task, activities, predictions, targets = [], [], [], []
    for epoch in range(epochs):

        if params["mode"] == "sequential":
            print("Sequential")
            if epoch < epochs//2:
                train_dataloader = trainloaderA
                test_dataloader = testloaderA
                task.append("seq_aquatic")
            else:
                train_dataloader = trainloaderB
                test_dataloader = testloaderB
                task.append("seq_vehicles")
            print(task[-1])
        elif params["mode"] == "interleaved":
            print("Interleaved")
            if epoch % 2 == 0:
                train_dataloader = trainloaderA
                test_dataloader = testloaderA
                task.append("int_aquatic")
            else:
                train_dataloader = trainloaderB
                test_dataloader = testloaderB
                task.append("int_vehicles")
            print(task[-1])

        running_loss = []
        net.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            yhat = net(x)

            loss = criterion(yhat, y)
            loss.backward()

            optimizer.step()

            running_loss.append(loss.item())

        train_loss.append(np.average(running_loss))
        print(f"Epoch: {epoch}, Loss = {train_loss[-1]}")
        print(f"LR: {optimizer.param_groups[0]["lr"]}")

        if epoch < 10:
            warmup_scheduler.step()
        else:
            decay_scheduler.step()

        net.eval()
        hook = net.encoder.layers[-1].register_forward_hook(hook_fn)
        accuracy = 0.0
        running_test_loss = []
        for x, y in test_dataloader:
            targets.append(y.detach().numpy())
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                yhat = net(x)
            predictions.append(yhat.detach().cpu().numpy())

            loss = criterion(yhat, y)
            running_test_loss.append(loss.item())

            prediction = yhat.argmax(dim=1)
            accuracy += prediction.eq(y).sum().item()

            activities.append(activations["pre"].cpu().numpy())

        hook.remove()
        test_accuracy.append(accuracy / len(testloaderA))
        test_loss.append(np.average(running_test_loss))

    activities = np.array(activities)
    np.save(exp_name+"_test_activities_"+str(experiment_id), activities)
    predictions = np.array(predictions)
    np.save(exp_name+"_test_labels_"+str(experiment_id), predictions)
    targets = np.array(targets)
    np.save(exp_name+"_test_targets_"+str(experiment_id), targets)

    train_loss = np.array(train_loss)
    np.save(exp_name+"_train_loss_"+str(experiment_id), train_loss)
    test_loss = np.array(test_loss)
    np.save(exp_name+"_test_loss_"+str(experiment_id), test_loss)
    test_accuracy = np.array(test_accuracy)
    np.save(exp_name+"_test_accuracy_"+str(experiment_id), test_accuracy)
    print("*****************************************************************")

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.plot(train_loss, label="Training loss")
    # ax.plot(test_loss, label="Test loss")
    # ax.legend()

    # ax = fig.add_subplot(122)
    # ax.plot(test_accuracy)
    # plt.show()
