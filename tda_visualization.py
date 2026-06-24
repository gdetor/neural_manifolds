import json
import numpy as np
import matplotlib.pylab as plt

# np.random.seed(13)


# Set up custom colors for plotting
ccolor = ["#cecece",
          "#a559aa",
          "#59a89c",
          "#f0c571",
          "#e02b35",
          "#082a54"
          ]

# **********************************************************************
#       THE USER SHOULD SET THE FOLLOWING VALUES ACCORDINGLY
exp_type = "cifar"
if exp_type == "toy":
    with open("parameters.json") as f:
        params = json.load(f)
elif exp_type == "cifar":
    with open("parameters_cifar100.json") as f:
        params = json.load(f)
else:
    with open("parameters_mnist.json") as f:
        params = json.load(f)

n_type = params["n_type"]
# n_experiments = params["n_experiments"]
n_experiments = 5
dir_ = "./tda_results/"


# **********************************************************************

labels = ["Seq A - Seq B",
          "Seq B - Inter B",
          "Seq A - Inter A",
          "Inter A - Inter B"]

pvals0 = np.load(dir_+"pvals0"+"_"+exp_type+"_"+n_type+".npy")
pvals1 = np.load(dir_+"pvals1"+"_"+exp_type+"_"+n_type+".npy")

bottle0 = np.load(dir_+"bdist0"+"_"+exp_type+"_"+n_type+".npy")
bottle1 = np.load(dir_+"bdist1"+"_"+exp_type+"_"+n_type+".npy")

# Values of connectivity sparsity for which we test for
# sparsity = [0.0, 0.3]
sparsity = [0.0]

# Plot nuances
letters1 = ["A", "B"]
letters2 = ["C", "D"]


def barPlot(dist, ax):
    """
    Wraps the bar plot of matplotlib to make life easier
    """
    if len(dist) > 0:
        y = dist.mean()
        yerr = dist.std()

        ax.bar(x[ii], y, color=ccolor[ii])
        if yerr < y:
            ax.errorbar(x[ii],
                        y,
                        yerr=yerr,
                        xerr=0,
                        capsize=5,
                        ls="dotted",
                        c="k",
                        lw=2,
                        markersize=10)
    else:
        y = 0.5
        ax.plot(x[ii], y, "X", color="r", ms=10)


fig = plt.figure(figsize=(13, 8))
fig.suptitle(exp_type.upper()+" "+n_type.upper(), fontsize=14)

case = ["Seq-Seq", "Seq-Int", "Int-Seq", "Int-Int"]
for k, _ in enumerate(sparsity):
    ax1 = fig.add_subplot(2, 2, k+1)
    ax2 = fig.add_subplot(2, 2, k+3)
    x = [-1, 1, 3, 5]
    ii = 0

    for i in range(4):
        dist0, dist1 = [], []
        for j in range(n_experiments):
            print(f"Case: {case[i]}, Experiment: {j}, dist = {bottle0[k, i, j]}")
            print(f"Case: {case[i]}, Experiment: {j}, p-value = {pvals0[k, i, j]}")
            if pvals0[k, i, j] < 0.05:
                if np.isfinite(bottle0[k, i, j]):
                    dist0.append(bottle0[k, i, j])
            if pvals1[k, i, j] < 0.05:
                if np.isfinite(bottle1[k, i, j]):
                    dist1.append(bottle1[k, i, j])
        dist0 = np.array(dist0)
        dist1 = np.array(dist1)
        barPlot(dist0, ax1)
        barPlot(dist1, ax2)

        ax1.text(4, 0.9, "$H_0$", fontsize=14)
        ax1.set_ylim([0, 1.1])
        ax1.set_title("Sparsity $p = $"+str(sparsity[k]), fontsize=14)
        ax1.set_xticks([])
        ax1.text(-1,
                 1.15,
                 letters1[k],
                 fontsize=18,
                 weight="bold")
        if k == 0:
            ax2.text(-3., 0.8,
                     "Avg. Bottleneck Distance",
                     fontsize=14,
                     rotation="vertical")

        ax2.text(4, 0.9, "$H_1$", fontsize=14)
        ax2.set_ylim([0, 1.1])
        ax2.set_xticks([-1, 1, 3, 5])
        ax2.set_xticklabels(labels, weight="bold", rotation=35)
        ax2.text(-1.,
                 1.15,
                 letters2[k],
                 fontsize=18,
                 weight="bold")
        ii += 1

# plt.savefig(exp_type+"_"+n_type+".svg")
# plt.savefig(exp_type+"_"+n_type+".pdf")
plt.show()
