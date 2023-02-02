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


# helper function for visualizing the output of a given layer
# default number of filters is 3
def viz_layer(layer, n_filters=3):
    fig = plt.figure(figsize=(20, 20))

    for i in range(n_filters):
        ax = fig
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))


# plot given filters as small images in a row
def plot_filters(filters):
    n_filters = len(filters)
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
    for i in range(n_filters):
        ax = fig
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i + 1))


if __name__ == "__main__":
    plt.figure(figsize=(3, 3))

    def relu(x_values):
        return np.maximum(0, x_values)

    x = np.linspace(-10, 10, 100)
    y = relu(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ReLU activation function')
    plt.grid()
    # plt.savefig("docs/ReLu.png")
    plt.show()
