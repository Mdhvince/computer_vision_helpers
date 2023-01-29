import matplotlib.pyplot as plt
import numpy as np


def basis_plotting_style(title, x_label, y_label, rotation_x_label, rotation_y_label):
    plt.style.use('seaborn-paper')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation_x_label)
    plt.yticks(rotation=rotation_y_label)


def draw_arrow_on_plot(text, x_pos_arrow, y_pos_arrow):
    config_arrow = dict(headwidth=7, facecolor='black', shrink=0.05, width=2)
    plt.annotate(text, xy=(x_pos_arrow, y_pos_arrow), xycoords='data',
                 xytext=(0.75, 0.95), textcoords='axes fraction',
                 arrowprops=config_arrow,
                 horizontalalignment='right', verticalalignment='top')


if __name__ == "__main__":
    plt.figure(figsize=(3, 3))

    def relu(x):
        return np.maximum(0, x)


    x = np.linspace(-10, 10, 100)
    y = relu(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ReLU activation function')
    plt.grid()
    # plt.savefig("docs/ReLu.png")
    plt.show()
