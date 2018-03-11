#ifndef OCTMPS_INCLUDE_SIMULATION_HPP_
#define OCTMPS_INCLUDE_SIMULATION_HPP_

// Increase the maximum arity
#define BOOST_PYTHON_MAX_ARITY 24

#include <boost/python.hpp>

#include "octmps.h"
#include "simulation.h"

ProbeData *InitProbeData(double, double, double);

boost::shared_ptr<ProbeData> ProbeDataWrappedInit(const boost::python::object&,
                                                    const boost::python::object&,
                                                    const boost::python::object&);

void ProbeDataExportToPython();

Simulation *InitSimulation(double, double, double,
                                        ProbeData *,
                                        unsigned int,
                                        double,
                                        short ,
                                        short,
                                        double,
                                        double,
                                        double,
                                        double,
                                        double,
                                        double,
                                        double,
                                        double,
                                        short,
                                        double,
                                        unsigned int,
                                        boost::shared_ptr<Region> *);

boost::shared_ptr<Simulation> SimulationWrappedInit(const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  const boost::python::object&,
                                                  boost::python::list);

void SimulationExportToPython();

void InitSimulation();

#endif