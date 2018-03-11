import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import csv
from itertools import cycle
cycol = cycle('bgrcmk')

font = {'family': 'serif',
        'weight': 'normal',
        'size': 24}

plt.rc('font', **font)


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def parse_OCTMPS_output_files(*files_labels, show_variance=False):

    def parse_row(file_name):
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                parsed_row = list(map(lambda x: float(x), row))
                yield parsed_row

    for label, file_name in files_labels:

        classI_z_positions = np.array(list(sorted(set([x[1] for x in parse_row(file_name)]))))
        classI_reflectances = np.array([x[2] for x in parse_row(file_name)])
        classI_variances = np.array([x[3] for x in parse_row(file_name)])
        plt.semilogy(classI_z_positions, classI_reflectances, next(cycol), label=label)
        if show_variance:
            plt.errorbar(classI_z_positions, classI_reflectances, yerr=classI_variances)

    plt.xlabel('Depth [cm]', **font)
    plt.xlim(0, 0.12)
    plt.ylabel('Reflectance', **font)
    plt.legend(loc='upper right', shadow=True, fontsize=16)
    plt.axes().set_aspect('auto')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':

    # Fig 6. A-scans representing Class I reflectance-based OCT signals from the  above simulation obtained by OCT-MPS
    # and by the serial implementation [9].
    # parse_OCTMPS_output_files(('Class I signal from serial implementation [9]',
    #                           '../../output/journal_paper_validation/serial_code/ClassI_ScattFilt.out'),
    #                          ('Class I signal from OCT-MPS',
    #                           '../../output/journal_paper_validation/parallel_code/ClassI_ScattFilt.out'))

    # Fig 7. A-scans representing Class II reflectance-based OCT signals from the above simulation obtained by OCT-MPS
    # and from the serial implementation [9].
    # parse_OCTMPS_output_files(('Class II signal from serial implementation [9]',
    #                           '../../output/journal_paper_validation/serial_code/ClassII_ScattFilt.out'),
    #                           ('Class II signal from OCT-MPS',
    #                           '../../output/journal_paper_validation/parallel_code/ClassII_ScattFilt.out'))

    # Fig 8. Class I OCT signals and their confidence intervals using 10^7, 10^6, and 10^5 photon packets.
    parse_OCTMPS_output_files((r'Class I OCT signal and its variance at different depths using $10^7$ photons',
                               '../../output/journal_paper_variance_comparison/using_10^7_photons/ClassI_ScattFilt.out'),
                              (r'Class I OCT signal and its variance at different depths using $10^6$ photons',
                               '../../output/journal_paper_variance_comparison/using_10^6_photons/ClassI_ScattFilt.out'),
                              (r'Class I OCT signal and its variance at different depths using $10^5$ photons',
                               '../../output/journal_paper_variance_comparison/using_10^5_photons/ClassI_ScattFilt.out')
                              , show_variance=True)

