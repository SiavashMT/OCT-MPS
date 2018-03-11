#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

#include <boost/python.hpp>
#include <boost/python/module.hpp>

int GetNumberOfAvailableGPUCards() {
	int n_devices;
	cudaGetDeviceCount(&n_devices);
	return n_devices;
}

void InitCudaUtils() {
	boost::python::object cuda_utils(boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("octmps.cuda_utils"))));

	boost::python::scope().attr("cuda_utils") = cuda_utils;

	boost::python::scope io_scope = cuda_utils;

	boost::python::def("get_number_of_available_gpu_cards", GetNumberOfAvailableGPUCards);
}

