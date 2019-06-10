import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn.tree.export import export_text
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Parameters
n_classes = 20
plot_colors = "ry"
plot_step = 0.02

# Load data
data_file = open('../breast-cancer-wisconsin.data', 'r')

raw_data = []
for line in data_file.readlines():
    line = line.replace('\n', '')
    raw_data.append((line.split(',')))
raw_data = np.array(raw_data).astype(int)

# breast-cancer-wisconsin
bcw = {'raw_data': raw_data}
bcw['data'] = bcw['raw_data'][:, 1:-1]
bcw['target'] = bcw['raw_data'][:, -1]
bcw['feature_names'] = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
bcw['target_names'] = ['benign', 'malignant']

for pairdx, pair in enumerate([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]):

    # We only take the two corresponding features
    X = bcw['data'][:, pair]
    Y = bcw['target']

    # Train
    clf = DecisionTreeClassifier().fit(X, Y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(bcw['feature_names'][pair[0]])
    plt.ylabel(bcw['feature_names'][pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        if i == 0:
            g = 2
        elif i == 1:
            g = 4
        idx = np.where(Y == g)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=bcw['target_names'][i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

# Plotagem textual -----------------------------------------------------------------------------------------------------
r = export_text(clf, feature_names=bcw['target_names'])
print(r)

# Plotagem grafica Tipo 1 ----------------------------------------------------------------------------------------------
# plt.suptitle("Decision surface of a decision tree using paired features")
# plt.legend(loc='lower right', borderpad=0, handletextpad=0)
# plt.axis("tight")
#
# plt.figure()
# clf = DecisionTreeClassifier().fit(bcw['data'], bcw['target'])
# plot_tree(clf, filled=True)
# plt.show()

# Plotagem grafica Tipo 2 (Gera um arquivo contendo um 'digraph tree')--------------------------------------------------
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
