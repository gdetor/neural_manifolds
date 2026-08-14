import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from common import assignExperimentName

# matplotlib.use("Agg")


# sparsity = [0.0, 0.3, 0.5, 0.8]
sparsity = [0.0, 0.3]
mode = ["sequential", "interleaved"]

fig = plt.figure()
fig.suptitle("FashionMNIST & MNIST - RNN")
xx = np.arange(4)
data = np.zeros((2, 4, 5, 10))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
for k, m in enumerate(mode):
    for i, sp in enumerate(sparsity):
        for p in range(5):
            name = assignExperimentName("./results/frozen_mnist",
                                        mode=m,
                                        n_type="rnn",
                                        n_neurons=100,
                                        epochs=100,
                                        sparsity=sp,
                                        index=p)
            print(name)
            data[k, i] = np.load(name+"_post_train_accuracy_"+str(p)+".npy")
        if m == "sequential":
            ax1.bar(xx[i], data[k, i].mean())
            ax1.errorbar(xx[i], data[k, i].mean(), data[k, i].std())
        else:
            ax2.bar(xx[i], data[k, i].mean())
            ax2.errorbar(xx[i], data[k, i].mean(), data[k, i].std())

ax1.set_title("Sequential")
ax2.set_title("Interleaved")

# plt.savefig("rnn_mnist.pdf")
plt.show()
