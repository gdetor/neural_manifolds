import numpy as np
import matplotlib
import matplotlib.pylab as plt

matplotlib.use("Agg")


data = np.zeros((5, 2))
for i in range(5):
    # data[i] = np.genfromtxt("./results_vit/cifar_cifar_interleaved_vit_"+str(i+1)+"_neurons_768_epochs_24_sparsity_0_post_accuracy_"+str(i+1))
    data[i] = np.genfromtxt("./results_vit/cifar_cifar_sequential_vit_"+str(i+1)+"_neurons_768_epochs_24_sparsity_0_post_accuracy_"+str(i+1))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar([1, 2], data.mean(axis=0), width=1, color="blue", alpha=0.82, zorder=3)
ax.errorbar([1, 2], data.mean(axis=0), data.std(axis=0), fmt="none",
            ecolor="black", elinewidth=1.0, capsize=3.0, capthick=1.0, zorder=4)
# plt.savefig("vit_accuracy_barplot.pdf")
