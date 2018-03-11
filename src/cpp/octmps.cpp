#include "octmps.h"

#include <boost/python.hpp>

#include "boost_utils.hpp"
#include "cuda_utils.hpp"
#include "main.cuh"
#include "mesh.h"
#include "mesh.hpp"
#include "simulation.h"
#include "simulation.hpp"

int WrappedOCTMPSRun(boost::python::list& Graph,
                     boost::python::object& simulation,
                     boost::python::object& num_GPUs,
                     boost::python::object& number_of_vertices,
                     boost::python::object& number_of_faces,
                     boost::python::object& number_of_tetrahedrons,
                     boost::python::object& n_simulations) {

  Tetrahedron *_Graph = unwrap_python_list<Tetrahedron >(Graph);
  Simulation *_simulation = boost::python::extract<Simulation *>(simulation);
  unsigned int _num_GPUs = boost::python::extract<unsigned int>(num_GPUs);
  int _number_of_vertices = boost::python::extract<int>(number_of_vertices);
  int _number_of_faces = boost::python::extract<int>(number_of_faces);
  int _number_of_tetrahedrons = boost::python::extract<int>(number_of_tetrahedrons);
  int _n_simulations = boost::python::extract<int>(n_simulations);

  return octmps_run(_Graph,
                    _simulation,
                    _num_GPUs,
                    _number_of_vertices,
                    _number_of_faces,
                    _number_of_tetrahedrons,
                    _n_simulations);
}

BOOST_PYTHON_MODULE(octmps) {
  boost::python::object package = boost::python::scope();
  package.attr("__path__") = "octmps";


  package.attr("NUM_SUBSTEPS_RESOLUTION") = NUM_SUBSTEPS_RESOLUTION;


  InitMesh();
  InitCudaUtils();
  InitSimulation();

  boost::python::def("octmps_run", &WrappedOCTMPSRun);
}
