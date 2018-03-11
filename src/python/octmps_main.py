from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
from math import fmod, ceil
from lib import octmps
from src.python.octmps_io import show_version
from src.python.mesh import load_mesh


def read_opt_files(opt_file_name):
    regions = list()
    with open(opt_file_name, 'r') as opt_file:
        opt_params = json.load(opt_file)
        number_of_runs = opt_params['number_of_runs']
        probe_data = octmps.simulation.ProbeData(**opt_params['probe_data'])
        # Ambient medium should be the first medium
        regions.append(
            octmps.mesh.Region(n=opt_params['n_ambient_medium']))  # Default value of parameters is zeros
        for region_params in opt_params['regions']:
            regions.append(octmps.mesh.Region(**region_params))

    return number_of_runs, {'probe': probe_data,
                            'n_regions': len(regions),
                            'regions': regions,
                            'number_of_photons': opt_params['number_of_photons']}


def read_bias_files(bias_file_name):
    with open(bias_file_name, 'r') as bias_file:
        bias_params = json.load(bias_file)
        # Calculating the Optical Depth Shift
        bias_params['optical_depth_shift'] = fmod(bias_params['target_depth_min'],
                                                  bias_params['coherence_length_source'])
        # Calculating the number of A-scan simulation points (steps) based on the coherent length of the probe
        # and the depth of interest
        bias_params['num_optical_depth_length_steps'] = int(ceil(octmps.NUM_SUBSTEPS_RESOLUTION * (
            bias_params['target_depth_max'] - bias_params['target_depth_min']) /
                                                                 bias_params['coherence_length_source']))

        return bias_params


def check_number_of_gpu_cards(value):
    number_of_available_gpu_card = octmps.cuda_utils.number_of_available_gpu_card()
    value = int(value)
    if not (1 <= value <= number_of_available_gpu_card):
        raise argparse.ArgumentTypeError(
            "Number of gpu card must be an integer between 1 to {}".format(number_of_available_gpu_card))
    return value


def parse_input():
    # Usage:
    # OCT-MPS [-S<seed>] [-G<num GPUs>] <input opt file> <input mesh file> <input bias file>

    parser = argparse.ArgumentParser(description='Massively Parallel Simulator of Optical Coherence '
                                                 'Tomography (OCTMPS)')
    parser.add_argument('-S', '--seed', help='random number generator seed',
                        dest='seed',
                        type=int, required=False)
    parser.add_argument('-G', '--number-of-gpu-cards',
                        help='Number of GPU cards, should be between 1 to {}'.format(
                            octmps.cuda_utils.get_number_of_available_gpu_cards()),
                        dest='num_gpu_cards',
                        type=check_number_of_gpu_cards,
                        required=False)

    parser.add_argument('--input-opt-json-file', help='Input opt json file',
                        dest='input_opt_file',
                        type=str, required=True)

    parser.add_argument('--input-mesh-file', help='Input .mesh file',
                        dest='input_mesh_file',
                        type=str, required=True)

    parser.add_argument('--input-bias-json-file', help='Input bias json file',
                        dest='input_bias_file',
                        type=str, required=True)

    parser.add_argument('--visualize', help='Visualize the mesh and B-Scan cross-section',
                        action="store_true", required=False)


    args = parser.parse_args()

    params = dict()
    number_of_runs, opt_params = read_opt_files(args.input_opt_file)
    params.update(opt_params)
    bias_params = read_bias_files(args.input_bias_file)
    params.update(bias_params)
    tetrahedrons, num_tetrahedrons, num_vertices, num_faces = load_mesh(args.input_mesh_file)
    params.update({'num_tetrahedrons': num_tetrahedrons})
    simulation = octmps.simulation.Simulation(**params)

    _ = octmps.octmps_run(tetrahedrons, simulation, 1, num_vertices, num_faces, num_tetrahedrons, 1)

    # Mayavi has code in their global name space that will execute command line parsing.
    # hence I cannot include it on top of the file and had to include it here and also remove
    # input command arguments!
    if args.visualize:
        import sys
        sys.argv = sys.argv[:1]
        from src.python.visualize import visualize
        visualize(tetrahedrons)


def run():
    show_version()
    parse_input()


if __name__ == '__main__':
    run()
