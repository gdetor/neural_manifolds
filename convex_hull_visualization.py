import json
import numpy as np
import matplotlib.pylab as plt


def normalize(x):
    # return x
    return (x - x.min()) / (x.max() - x.min())


# **********************************************************************
#       PLOTTING ATTRIBUTES
# Set up custom colors for plotting
ccolor = ["#cecece",
          "#a559aa",
          "#59a89c",
          "#f0c571",
          "#e02b35",
          "#082a54"
          ]

cases = ["Seq A - Seq B",
         "Seq B - Inter B",
         "Seq A - Inter A",
         "Inter A - Inter B"]

letters1 = ["A", "B"]
letters2 = ["C", "D"]

# **********************************************************************
#       THE USER SHOULD SET THE FOLLOWING VALUES ACCORDINGLY

metric = "jaccard"
task = "mnist"
dir_ = "./convex_hull_results/"

with open("parameters.json") as f:
    params = json.load(f)

n_experiments = params["n_experiments"]

# **********************************************************************
# Load the data for both the RNN & MLP
n_type = "mlp"
mlp_jacc = normalize(np.load(dir_+"jaccard_"+task+"_"+n_type+".npy"))
mlp_inter = normalize(np.load(dir_+"intersect_points_"+task+"_"+n_type+".npy"))

n_type = "rnn"
rnn_jacc = normalize(np.load(dir_+"jaccard_"+task+"_"+n_type+".npy"))
rnn_inter = normalize(np.load(dir_+"intersect_points_"+task+"_"+n_type+".npy"))

n_labels = mlp_jacc.shape[-1]
n_cases = n_labels * 6

# Values of connectivity sparsity for which we test for
sparsity = [0.0, 0.3]


fig = plt.figure(figsize=(13, 8))
if metric == "jaccard":
    fig.suptitle("Averaged Jaccard Similarity MLP/RNN "+task.upper())
    max_y = max([mlp_jacc.max(), rnn_jacc.max()])
else:
    fig.suptitle("Averaged Intersection Points MLP/RNN "+task.upper())
    max_y = max([mlp_inter.max(), rnn_inter.max()])

perc = int(max_y * 0.1)

for k, sp in enumerate(sparsity):
    ax1 = fig.add_subplot(2, 2, k+1)
    ax2 = fig.add_subplot(2, 2, k+3)
    ax1.set_ylim([0, max_y])
    ax2.set_ylim([0, max_y])

    ax1.text(-0.5, max_y+0.03, letters1[k], fontsize=18, weight="bold")
    ax2.text(-0.5, max_y+0.03, letters2[k], fontsize=18, weight="bold")

    ii = 0
    x = np.arange(4)
    for i in range(4):
        if metric == "jaccard":
            mu_mlp = mlp_jacc[k, i].mean(axis=0)
            mu_rnn = rnn_jacc[k, i].mean(axis=0)
        else:
            mu_mlp = mlp_inter[k, i].mean(axis=0)
            mu_rnn = rnn_inter[k, i].mean(axis=0)

        ax1.bar(x[ii],
                mu_mlp,
                width=0.8,
                color=ccolor[ii])
        ax1.set_xticks([])
        ax1.set_title("Sparsity $p = $"+str(sparsity[k]), fontsize=14)

        ax2.bar(x[ii],
                mu_rnn,
                width=0.8,
                color=ccolor[ii])
        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_xticklabels(cases, weight="bold", rotation=35)
        if metric == "intersections":
            tag = "Avg. Intersection Points"
        else:
            tag = "Avg. Jaccard Index"
        ax2.text(-6.5, 0.7, tag, fontsize=14, rotation="vertical")
        ii += 1

# plt.savefig(metric+"_"+task+"_"+n_type+".svg")
# plt.savefig(metric+"_"+task+"_"+n_type+".pdf")
plt.show()
