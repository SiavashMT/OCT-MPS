# You need to instalInvalid __global__ write of sizel cmake from the git repository
# Otherwise you will run into the following issue (even on 3.9.3)
# https://gitlab.kitware.com/cmake/cmake/issues/16411
cmake_minimum_required(VERSION 3.9.4)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# GCC and G++ versions higher than 5 is not supported
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    message(WARNING "CUDA does not support G++ version more than 5!\nSwitching to g++-4.9")
    set(CMAKE_CXX_COMPILER g++-4.9)
endif ()

if (CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.9)
    message(WARNING "CUDA does not support GCC version more than 5!\nSwitching to gcc-4.9")
    set(CMAKE_C_COMPILER gcc-4.9)
endif ()

project(octmps)

include_directories(include)

add_subdirectory(src/c)
add_subdirectory(src/cuda)
add_subdirectory(src/cpp)

# add_custom_command(
# 	TARGET octmps
# 	POST_BUILD
#     COMMAND ENV PYTHONPATH ${CMAKE_CURRENT_SOURCE_DIR} APPEND
# )
