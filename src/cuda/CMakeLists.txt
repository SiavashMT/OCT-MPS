### CMakeLists.txt for CUDA
project(octmps_kernel)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_HOST_COMPILATION_CPP OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
set(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive,--no-undefined,-z,defs,-static-libgcc")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# Finding Compute Capability
execute_process(COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/detect_compute_capability.sh
        OUTPUT_VARIABLE
        GENCODE)

# Fix for: build.make: *** missing separator. Stop.
string(STRIP ${GENCODE} GENCODE)
message("\nCompiling with CUDA Compute Capability " ${GENCODE} "\n")


find_package(CUDA QUIET REQUIRED)
message(STATUS "CUDA Found ${CUDA_FOUND}")

# Debugging flags
set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3;-DNDEBUG;-lineinfo")

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 ${GENCode} -rdc=true
)

find_library(octmps_core REQUIRED)

cuda_include_directories(/usr/local/cuda/include/
        /usr/local/cuda/samples/common/inc/)


file(GLOB SOURCES "*.cu")
cuda_add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} octmps_core)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        COMPONENT libraries)




