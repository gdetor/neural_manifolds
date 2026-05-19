import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import prune


class NetRNNHebb(nn.Module):
    def __init__(self, input_size=1,
                 n_neurons=10,
                 output_size=1,
                 num_layers=1,
                 sparsity=0.6,
                 alpha=1.0,
                 mask=None,
                 sequence_len=1):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = n_neurons
        self.sparsity = sparsity
        self.decay = 1.0
        self.lr = 0.01

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=n_neurons,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          nonlinearity="relu",
                          dtype=torch.float32)

        self.num_layers = num_layers
        self.hidden_size = n_neurons

        self.fc_out = nn.Linear(n_neurons * sequence_len,
                                output_size,
                                bias=True)

        self.reset()
        self._apply_pruning(sparsity)

    def reset(self):
        nn.init.normal_(self.rnn.weight_hh_l0, 0, 0.5)
        nn.init.constant_(self.rnn.bias_hh_l0, 0.0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0.0)
        nn.init.normal_(self.fc_out.weight,
                        0.0, 1.0/math.sqrt(self.hidden_size))

        self._apply_pruning(self.sparsity)

    def _apply_pruning(self, sparsity):
        prune.random_unstructured(self.rnn,
                                  name="weight_hh_l0",
                                  amount=sparsity)

    def forward(self, x, hn=None):
        x = x.unsqueeze(2)
        bs, sl, _ = x.shape
        h_out, hn = self.rnn(x, hn)
        out = self.fc_out(hn.reshape(bs, self.num_layers*self.hidden_size))
        return (1.0 / self.alpha) * out, h_out

    def hebbian_update(self, h_history, logits):
        delta_W = torch.zeros_like(self.rnn.weight_hh_l0)
        n_transitions = len(h_history) - 1

        for t in range(n_transitions):
            h_post = h_history[t + 1]   # (batch, n_neurons)
            h_pre = h_history[t]        # (batch, n_neurons)
            # Outer product averaged over batch
            delta_W += h_post.T @ h_pre / h_post.size(0)

        delta_W /= n_transitions

        # Apply decay + Hebbian increment
        self.rnn.weight_hh_l0.data = (
                self.decay * self.rnn.weight_hh_l0.data + self.lr * delta_W
                )

        # Normalise rows to prevent runaway weights
        norms = self.rnn.weight_hh_l0.data.norm(
                dim=1, keepdim=True).clamp(min=1e-8)
        self.rnn.weight_hh_l0.data /= norms

        tmp = logits.unsqueeze(2) @ h_history[:, -1, :].unsqueeze(1)
        self.fc_out.weight.data += torch.mean(tmp, dim=0)
        norms = self.fc_out.weight.data.norm(
                dim=1, keepdim=True).clamp(min=1e-8)
        self.fc_out.weight.data /= norms


class NetRNN(nn.Module):
    def __init__(self, input_size=1,
                 n_neurons=10,
                 output_size=1,
                 num_layers=1,
                 sparsity=0.6,
                 alpha=1.0,
                 mask=None,
                 sequence_len=1):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = n_neurons
        self.sparsity = sparsity

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=n_neurons,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          nonlinearity="relu",
                          dtype=torch.float32)

        self.num_layers = num_layers
        self.hidden_size = n_neurons

        self.fc_out = nn.Linear(n_neurons * sequence_len,
                                output_size,
                                bias=True)

        self.reset()
        self._apply_pruning(sparsity)

    def reset(self):
        nn.init.xavier_normal_(self.rnn.weight_hh_l0,
                               gain=nn.init.calculate_gain("tanh"))
        nn.init.constant_(self.rnn.bias_hh_l0, 0.0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0.0)
        nn.init.xavier_normal_(self.fc_out.weight,
                               gain=nn.init.calculate_gain("tanh"))

        self._apply_pruning(self.sparsity)

    def _apply_pruning(self, sparsity):
        prune.random_unstructured(self.rnn,
                                  name="weight_hh_l0",
                                  amount=sparsity)

    def forward(self, x, hn=None):
        x = x.unsqueeze(2)
        bs, sl, _ = x.shape
        out, hn = self.rnn(x, hn)
        out = self.fc_out(hn.reshape(bs, self.num_layers * self.hidden_size))
        return (1.0 / self.alpha) * out, hn


class NetRNNMNIST(nn.Module):
    def __init__(self,
                 input_size=1,
                 n_neurons=10,
                 output_size=1,
                 num_layers=1,
                 sparsity=0.6,
                 alpha=1.0,
                 mask=None,
                 sequence_len=1):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = n_neurons
        self.input_size = input_size
        self.sparsity = sparsity

        self.rnn = nn.RNN(input_size=n_neurons,
                          hidden_size=n_neurons,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          nonlinearity="relu",
                          dtype=torch.float32)

        self.num_layers = num_layers
        self.hidden_size = n_neurons

        self.fc_in = nn.Linear(input_size, n_neurons)
        self.fc1 = nn.Linear(n_neurons, n_neurons)
        self.fc_out = nn.Linear(n_neurons * sequence_len,
                                output_size,
                                bias=True)

        self.act = nn.Tanh()
        self.reset()
        self._apply_pruning(sparsity)

    def reset(self):
        nn.init.xavier_normal_(self.rnn.weight_hh_l0,
                               gain=nn.init.calculate_gain("tanh"))
        nn.init.constant_(self.rnn.bias_hh_l0, 0.0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0.0)
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.xavier_normal_(self.fc1.weight,
                               gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_normal_(self.fc_in.weight,
                               gain=nn.init.calculate_gain("tanh"))

        self._apply_pruning(self.sparsity)

    def _apply_pruning(self, sparsity):
        prune.random_unstructured(self.rnn,
                                  name="weight_hh_l0",
                                  amount=sparsity)

    def forward(self, x, hn=None):
        x = x.squeeze(1)
        bs, sl, _ = x.shape
        x = x.reshape(-1, 784)
        out = self.act(self.fc_in(x))
        out, hn = self.rnn(out.unsqueeze(1), hn)
        out = self.act(self.fc1(
            hn.reshape(bs, self.num_layers * self.hidden_size))
                       )
        out = self.fc_out(out)
        return (1.0 / self.alpha) * out, hn


class NetMLP(nn.Module):
    def __init__(self, hidden_size, alpha=1.0, sparsity=0.0, mask=None):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.mask = mask

        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

        self.act = nn.Tanh()

        self.reset()
        self._apply_pruning(sparsity)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.fc3(out)

        return (1.0 / self.alpha) * out, None

    def _apply_pruning(self, sparsity):
        if self.mask is None:
            prune.random_unstructured(self.fc2,
                                      name="weight",
                                      amount=sparsity)
        else:
            prune.custom_from_mask(self.fc2,
                                   name="weight",
                                   mask=self.mask)

    def reset(self):
        for p in self.parameters():
            # nn.init.normal_(p, 0.0, 0.5)
            if p.ndim >= 2:
                # nn.init.normal_(p, 0.0, 0.5)
                nn.init.xavier_normal_(p, gain=nn.init.calculate_gain('tanh'))
            if p.ndim < 2:
                nn.init.constant_(p, 0.0)
        nn.init.normal_(self.fc3.weight,
                        0.0, 1.0/math.sqrt(self.hidden_size))

        self._apply_pruning(self.sparsity)


class NetMLPMNIST(nn.Module):
    def __init__(self, hidden_size, alpha=1.0, sparsity=0.0, mask=None):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.mask = mask

        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 10)

        self.act = nn.Tanh()

        self.reset()
        self._apply_pruning(sparsity)

    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.act(self.fc4(out))
        out = self.fc5(out)

        return (1.0 / self.alpha) * out, None

    def _apply_pruning(self, sparsity):
        if self.mask is None:
            prune.random_unstructured(self.fc4,
                                      name="weight",
                                      amount=sparsity)
        else:
            prune.custom_from_mask(self.fc4,
                                   name="weight",
                                   mask=self.mask)

    def reset(self):
        for p in self.parameters():
            # nn.init.normal_(p, 0.0, 0.5)
            if p.ndim >= 2:
                nn.init.xavier_normal_(p, gain=nn.init.calculate_gain("tanh"))
            if p.ndim < 2:
                nn.init.constant_(p, 0.0)
        nn.init.xavier_normal_(self.fc3.weight,
                               gain=nn.init.calculate_gain("tanh"))

        self._apply_pruning(self.sparsity)


class VisualNetCNN(nn.Module):
    def __init__(self, input_size=28, latent_size=100, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, latent_size)
        self.fc3 = nn.Linear(latent_size, latent_size)

        self.fc_out = nn.Linear(latent_size, 10)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc_out(out)
        return out, None
