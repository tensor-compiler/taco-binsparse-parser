include(FetchContent)

cmake_minimum_required(VERSION 3.20)
project(bsp-to-taco)

add_library(bsp-to-taco SHARED)

cmake_policy(SET CMP0079 NEW)

set(CMAKE_C_STANDARD 11)
target_compile_features(bsp-to-taco PUBLIC cxx_std_11)

#set(CMAKE_C_FLAGS "-O3 -march=native")
set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wno-narrowing -fPIC")

add_subdirectory(include)
add_subdirectory(src)

#### binsparse reference implementation configuration

FetchContent_Declare(
  binsparse
  GIT_REPOSITORY https://github.com/Binsparse/binsparse-reference-c
  GIT_TAG 13e56902c6f729822730297a5d13bbdbc62e619b
)
FetchContent_MakeAvailable(binsparse)
target_include_directories(bsp-to-taco PUBLIC ${binsparse_SOURCE_DIR}/include)

#### taco configuration

option(CUDA "Build for NVIDIA GPU (CUDA must be preinstalled)" OFF)
option(PYTHON "Build TACO for python environment" OFF)
option(OPENMP "Build with OpenMP execution support" OFF)
option(COVERAGE "Build with code coverage analysis" OFF)
set(TACO_FEATURE_CUDA 0)
set(TACO_FEATURE_OPENMP 0)
set(TACO_FEATURE_PYTHON 0)
if(CUDA)
  message("-- Searching for CUDA Installation")
  find_package(CUDA REQUIRED)
  add_definitions(-DCUDA_BUILT)
  set(TACO_FEATURE_CUDA 1)
endif(CUDA)
if(OPENMP)
  message("-- Will use OpenMP for parallel execution")
  add_definitions(-DUSE_OPENMP)
  set(TACO_FEATURE_OPENMP 1)
endif(OPENMP)
if(PYTHON)
  message("-- Will build Python extension")
  add_definitions(-DPYTHON)
  set(TACO_FEATURE_PYTHON 1)
endif(PYTHON)

FetchContent_Declare(
  taco
  GIT_REPOSITORY https://github.com/junikimm717/taco
  GIT_TAG master
)
FetchContent_MakeAvailable(taco)
configure_file(${taco_SOURCE_DIR}/include/taco/version.h.in ${taco_SOURCE_DIR}/include/taco/version.h @ONLY)
target_include_directories(${PROJECT_NAME} PUBLIC ${taco_SOURCE_DIR}/include)

#### Project Configuration

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_BINARY_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC binsparse taco)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  add_subdirectory(examples)
  add_subdirectory(test)
endif()
