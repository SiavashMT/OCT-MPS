import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from src.python.octmps_output import parse_OCTMPS_output_file

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18}


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def visualize(file_name, label=None):

    reflectance_grid, reflectances, x_positions, z_positions = parse_OCTMPS_output_file(file_name=file_name)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    if label:
        fig.suptitle(label, **font)

    im = axes[1].imshow(reflectance_grid.transpose(),
                        extent=(x_positions.min(), x_positions.max(), z_positions.max(), z_positions.min()), cmap='jet',
                        interpolation='none')
    axes[1].set_xlabel('Distance X [cm]', **font)
    axes[1].set_ylabel('Depth Z [cm]', **font)
    axes[1].set_xticks([x_positions.min(), x_positions.max()])

    axes[0].imshow(reflectance_grid[:, :150].transpose(),
                   extent=(x_positions.min(), x_positions.max(), z_positions.max(), z_positions.min()), cmap='jet',
                   interpolation='none')
    axes[0].set_xlabel('Distance X [cm]', **font)
    axes[0].set_ylabel('Depth Z [cm]', **font)
    axes[0].set_aspect('auto')

    colorbar = fig.colorbar(im)
    colorbar.set_label(label='Reflectance', **font)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--OCTMPS_output_file", type=str, required=True, help="OCTMPS output file")
    parser.add_argument("--Label", type=str, required=False, help="Label for the figures")
    args = parser.parse_args()

    visualize(args.OCTMPS_output_file, args.Label)

