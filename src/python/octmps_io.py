from __future__ import absolute_import, division, print_function, unicode_literals


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def show_version():
    print("\n\n")
    print(bcolors.HEADER + "\tMassively Parallel Simulator of Optical Coherence Tomography of Inhomogeneous "
                           "Turbid Media ")
    print("\n")

    print("\tSiavash Malektaji, Mauricio R. Escobar I., Ivan T. Lima Jr., and Sherif S. Sherif")
    print("\n")

    print("\tDepartment of Electrical and Computer Engineering, University of Manitoba")
    print("\tWinnipeg, Manitoba, Canada")
    print("\tEmail: Sherif.Sherif@umanitoba.ca")
    print("\n")

    print(bcolors.OKGREEN + "\tThe program is written based on the MCML, GPUMCML and TIM-OS codes, and the following "
                            "paper:\n\n")

    print("\t(1) S. Malektaji, Ivan .T Lima Jr. Sherif S. Sherif \"Monte Carlo Simulation of Optical\n")
    print("\tCoherence Tomography for Turbid Media with Arbitrary Spatial Distribution\", Journal of\n")
    print("\tBiomedical Optics 19.4 (2014): 046001-046001.\n\n")

    print("\t(2) L.-H. Wang, S. L. Jacques, and L.-Q. Zheng, \"MCML - Monte Carlo modeling of photon\n")
    print("\ttransport in multi-layered tissues\", Computer Methods and Programs in Biomedicine, 47,\n")
    print("\t131-146, 1995.\n\n")

    print("\t(3) H. Shen and G. Wang. \"Tetrahedron-based inhomogeneous Monte-Carlo optical simulator.\"\n")
    print("\tPhys. Med. Biol. 55:947-962, 2010.\n\n")

    print("\t(4) Alerstam, Erik, Tomas Svensson, and Stefan Andersson-Engels. \"Parallel computing with\n")
    print("\tgraphics processing units for high-speed Monte Carlo simulation of photon migration.\"\n")
    print("\tJournal of biomedical optics 13.6 (2008): 060504-060504..\n")
    print("\n")


if __name__ == '__main__':
    show_version()
