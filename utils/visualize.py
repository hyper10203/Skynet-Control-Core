from __future__ import annotations

import matplotlib.pyplot as plt


def show(grid, title: str = "") -> None:
    plt.imshow(grid)
    plt.title(title)
    plt.show()
