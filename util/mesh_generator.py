from netgen.libngpy._stl import *
from netgen.libngpy._csg import *
from netgen.libngpy._meshing import *
import os


def create_mesh_file(geo_file: str, output_mesh_file: str, delimiter: str='\t'):

    if not os.path.exists(geo_file):
        raise Exception("The geo file with path {} does not exits!!".format(geo_file))

    geo = CSGeometry(geo_file)
    geo.ntlo

    param = MeshingParameters()
    m1 = GenerateMesh(geo, param)

    points = m1.Points()
    elements3d = m1.Elements3D()

    number_of_points = len(points)
    number_of_tetrahedrons = len(elements3d)

    with open(output_mesh_file, 'w+') as output_file:
        output_file.write(str(number_of_points)+str('\n'))
        output_file.write(str(number_of_tetrahedrons)+str('\n'))

        for point in points:
            output_file.write(delimiter.join([str(p) for p in point.p])+str('\n'))

        for tetrahedron in elements3d:
            output_file.write(delimiter.join([str(p) for p in list(tetrahedron.vertices)+[tetrahedron.index]])+str('\n'))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_file", type=str, required=True, help="Geo File")

    parser.add_argument("--output_mesh_file", type=str, required=True, help="Output Mesh File")

    args = parser.parse_args()

    create_mesh_file(args.geo_file, args.output_mesh_file)
