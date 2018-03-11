#include "simulation.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

#include "boost_utils.hpp"
#include "mesh.h"


/****************
**  Probe Data
*****************/

ProbeData *InitProbeData(double start_x,
                         double end_x,
                         double distance_Ascans) {
  ProbeData *pd = (ProbeData *) malloc(sizeof(ProbeData));
  pd->start_x = start_x;
  pd->end_x = end_x;
  pd->distance_Ascans = distance_Ascans;
  return pd;
}

boost::shared_ptr<ProbeData> ProbeDataWrappedInit(const boost::python::object& start_x,
    const boost::python::object& end_x,
    const boost::python::object& distance_Ascans) {

  double _start_x = boost::python::extract<double>(start_x);
  double _end_x = boost::python::extract<double>(end_x);
  double _distance_Ascans = boost::python::extract<double>(distance_Ascans);
  return boost::shared_ptr<ProbeData>(InitProbeData(_start_x, _end_x, _distance_Ascans));
}

void ProbeDataExportToPython() {
  boost::python::class_<ProbeData, boost::shared_ptr<ProbeData>, boost::noncopyable>("ProbeData", boost::python::no_init)
  .def("__init__", make_constructor(&ProbeDataWrappedInit, boost::python::default_call_policies(),
                                    (boost::python::arg("start_x"),
                                     boost::python::arg("end_x"),
                                     boost::python::arg("distance_Ascans"))))
  .def_readwrite("start_x", &ProbeData::start_x)
  .def_readwrite("end_x", &ProbeData::end_x)
  .def_readwrite("distance_Ascans", &ProbeData::distance_Ascans);
}

/**********************
**  Simulation Struct
**********************/

Simulation *InitSimulation(double probe_x, double probe_y, double probe_z,
                           ProbeData *probe,
                           unsigned int number_of_photons,
                           double r_specular,
                           short num_tetrahedrons,
                           short type_bias,
                           double backward_bias_coefficient,
                           double coherence_length_source,
                           double target_depth_min,
                           double target_depth_max,
                           double max_collecting_radius,
                           double max_collecting_angle_deg,
                           double probability_additional_bias,
                           double max_relative_contribution_to_bin_per_post_processed_sample,
                           short num_optical_depth_length_steps,
                           double optical_depth_shift,
                           unsigned int n_regions,
                           Region *regions) {

  Simulation *ss = (Simulation *) malloc(sizeof(Simulation));
  ss->probe_x = probe_x;
  ss->probe_y = probe_y;
  ss->probe_z = probe_z;
  ss->probe = probe;
  ss->number_of_photons = number_of_photons;
  ss->r_specular = r_specular;
  ss->num_tetrahedrons = num_tetrahedrons;
  ss->type_bias = type_bias;
  ss->backward_bias_coefficient = backward_bias_coefficient;
  ss->coherence_length_source = coherence_length_source;
  ss->target_depth_min = target_depth_min;
  ss->target_depth_max = target_depth_max;
  ss->max_collecting_radius = max_collecting_radius;
  ss->max_collecting_angle_deg = max_collecting_angle_deg;
  ss->probability_additional_bias = probability_additional_bias;
  ss->max_relative_contribution_to_bin_per_post_processed_sample = max_relative_contribution_to_bin_per_post_processed_sample;
  ss->num_optical_depth_length_steps = num_optical_depth_length_steps;
  ss->optical_depth_shift = optical_depth_shift;
  ss->n_regions = n_regions;
  ss->regions = regions;
  return ss;
}

boost::shared_ptr<Simulation> SimulationWrappedInit(const boost::python::object& probe_x,
    const boost::python::object& probe_y,
    const boost::python::object& probe_z,
    const boost::python::object& probe,
    const boost::python::object& number_of_photons,
    const boost::python::object& r_specular,
    const boost::python::object& num_tetrahedrons,
    const boost::python::object& type_bias,
    const boost::python::object& backward_bias_coefficient,
    const boost::python::object& coherence_length_source,
    const boost::python::object& target_depth_min,
    const boost::python::object& target_depth_max,
    const boost::python::object& max_collecting_radius,
    const boost::python::object& max_collecting_angle_deg,
    const boost::python::object& probability_additional_bias,
    const boost::python::object& max_relative_contribution_to_bin_per_post_processed_sample,
    const boost::python::object& num_optical_depth_length_steps,
    const boost::python::object& optical_depth_shift,
    const boost::python::object& n_regions,
    boost::python::list regions) {

  double _probe_x = boost::python::extract<double>(probe_x);
  double _probe_y = boost::python::extract<double>(probe_y);
  double _probe_z = boost::python::extract<double>(probe_z);
  ProbeData *_probe = boost::python::extract<ProbeData *>(probe);
  unsigned int _number_of_photons = boost::python::extract<unsigned int>(number_of_photons);
  double _r_specular = boost::python::extract<double>(r_specular);
  short _num_tetrahedrons = boost::python::extract<short>(num_tetrahedrons);
  short _type_bias = boost::python::extract<short >(type_bias);
  double _backward_bias_coefficient = boost::python::extract<double>(backward_bias_coefficient);
  double _coherence_length_source = boost::python::extract<double>(coherence_length_source);

  double _target_depth_min = boost::python::extract<double>(target_depth_min);
  double _target_depth_max = boost::python::extract<double>(target_depth_max);

  double _max_collecting_radius = boost::python::extract<double>(max_collecting_radius);
  double _max_collecting_angle_deg = boost::python::extract<double>(max_collecting_angle_deg);

  double _probability_additional_bias = boost::python::extract<double>(probability_additional_bias);
  double _max_relative_contribution_to_bin_per_post_processed_sample = boost::python::extract<double>(max_relative_contribution_to_bin_per_post_processed_sample);

  short _num_optical_depth_length_steps = boost::python::extract<short>(num_optical_depth_length_steps);
  double _optical_depth_shift = boost::python::extract<double>(optical_depth_shift);

  unsigned int _n_regions = boost::python::extract<unsigned int>(n_regions);
  Region *_regions = unwrap_python_list<Region>(regions);
  return boost::shared_ptr<Simulation>(InitSimulation(_probe_x, _probe_y, _probe_z,
                                       _probe,
                                       _number_of_photons,
                                       _r_specular,
                                       _num_tetrahedrons,
                                       _type_bias,
                                       _backward_bias_coefficient,
                                       _coherence_length_source,
                                       _target_depth_min,
                                       _target_depth_max,
                                       _max_collecting_radius,
                                       _max_collecting_angle_deg,
                                       _probability_additional_bias,
                                       _max_relative_contribution_to_bin_per_post_processed_sample,
                                       _num_optical_depth_length_steps,
                                       _optical_depth_shift,
                                       _n_regions,
                                       _regions));
}

void SimulationExportToPython() {

  boost::python::class_<Simulation, boost::shared_ptr<Simulation>, boost::noncopyable>("Simulation", boost::python::no_init)
  .def("__init__", boost::python::make_constructor(&SimulationWrappedInit,
       boost::python::default_call_policies(),
       (
         boost::python::arg("probe_x") = 0.0,
         boost::python::arg("probe_y") = 0.0,
         boost::python::arg("probe_z") = 0.0,
         boost::python::arg("probe"),
         boost::python::arg("number_of_photons"),
         boost::python::arg("r_specular") = 0.0,
         boost::python::arg("num_tetrahedrons"),
         boost::python::arg("type_bias"),
         boost::python::arg("backward_bias_coefficient"),
         boost::python::arg("coherence_length_source"),
         boost::python::arg("target_depth_min"),
         boost::python::arg("target_depth_max"),
         boost::python::arg("max_collecting_radius"),
         boost::python::arg("max_collecting_angle_deg"),
         boost::python::arg("probability_additional_bias"),
         boost::python::arg("max_relative_contribution_to_bin_per_post_processed_sample"),
         boost::python::arg("num_optical_depth_length_steps"),
         boost::python::arg("optical_depth_shift"),
         boost::python::arg("n_regions"),
         boost::python::arg("regions"))));
}

void InitSimulation() {
  boost::python::object simulation(boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("octmps.simulation"))));

  boost::python::scope().attr("simulation") = simulation;
  // set the current scope to the new sub-module
  boost::python::scope io_scope = simulation;

  ProbeDataExportToPython();
  SimulationExportToPython();
}
