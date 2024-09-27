import numpy as np
import matplotlib.pyplot as plt

def hinton(matrix, max_weight=None, ax=None):
    """
    绘制Hinton图来可视化矩阵的权重。
    matrix: 要绘制的矩阵
    max_weight: 方块大小的最大值（None时自动调整）
    ax: 可选，提供已有的matplotlib轴对象
    """
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))


    ax = ax if ax is not None else plt.gca()
    ax.cla()
    ax.patch.set_facecolor("white")
    ax.set_aspect("equal", "box")

    for (x, y), w in np.ndenumerate(matrix):
        color = "lawngreen" if w > 0 else "royalblue"
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    nr, nc = matrix.shape
    ax.set_xticks(np.arange(0, nr))
    ax.set_yticks(np.arange(0, nc))
    ax.grid(linestyle="--", linewidth=1, color="gray")
    ax.autoscale_view()
    ax.invert_yaxis()