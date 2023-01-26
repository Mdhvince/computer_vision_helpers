import matplotlib.pyplot as plt



def basis_plotting_style(title, xlabel, ylabel, rotation_xlabel, rotation_ylabel):
    plt.style.use('seaborn-paper')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation_xlabel)
    plt.yticks(rotation=rotation_ylabel)


def draw_arrow_on_plot(text, x_pos_arrow, y_pos_arrow):
    config_arrow = dict(headwidth=7, facecolor='black', shrink=0.05, width=2)
    plt.annotate(text, xy=(x_pos_arrow, y_pos_arrow), xycoords='data',
                 xytext=(0.75, 0.95), textcoords='axes fraction',
                 arrowprops=config_arrow,
                 horizontalalignment='right', verticalalignment='top')


if __name__ == "__main__":
	
	pass