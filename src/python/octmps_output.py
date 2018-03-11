from __future__ import absolute_import, division, print_function, unicode_literals
import csv
import numpy as np


def parse_OCTMPS_output_file(file_name):

    def parse_row():
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                parsed_row = list(map(lambda x: float(x), row))
                yield parsed_row

    x_positions = np.array(list(sorted(set([x[0] for x in parse_row()]))))
    z_positions = np.array(list(sorted(set([x[1] for x in parse_row()]))))

    print('number of A-Scans:{}'.format(len(x_positions)))

    reflectances = np.array([x[2] for x in parse_row()])

    reflectance_grid = reflectances.reshape((x_positions.shape[0], z_positions.shape[0]))

    return reflectance_grid, reflectances, x_positions, z_positions
