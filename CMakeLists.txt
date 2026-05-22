cmake_minimum_required(VERSION 3.16)
project(turboquant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TURBOQUANT_BUILD_TESTS "Build unit tests" ON)
option(TURBOQUANT_BUILD_BENCH "Build microbenchmarks" ON)
option(TURBOQUANT_BUILD_DATASET_BENCH "Build HDF5 dataset bench (needs HDF5+BLAS)" ON)

# --- Highway ----------------------------------------------------------------
find_package(HWY QUIET)
if(NOT HWY_FOUND)
  find_path(HWY_INCLUDE_DIR hwy/highway.h)
  find_library(HWY_LIBRARY NAMES hwy)
  if(NOT HWY_INCLUDE_DIR OR NOT HWY_LIBRARY)
    include(FetchContent)
    FetchContent_Declare(
      highway
      GIT_REPOSITORY https://github.com/google/highway.git
      GIT_TAG 1.0.7
    )
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(highway)
    set(TURBOQUANT_HWY_TARGET hwy)
  else()
    add_library(turboquant_hwy_imported INTERFACE)
    target_include_directories(turboquant_hwy_imported INTERFACE ${HWY_INCLUDE_DIR})
    target_link_libraries(turboquant_hwy_imported INTERFACE ${HWY_LIBRARY})
    set(TURBOQUANT_HWY_TARGET turboquant_hwy_imported)
  endif()
else()
  set(TURBOQUANT_HWY_TARGET hwy::hwy)
endif()

# --- Library ----------------------------------------------------------------
add_library(turboquant
  src/turboquant.cc
  src/packing.cc
  src/rotation.cc
  src/kernels.cc
)
target_include_directories(turboquant
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(turboquant PUBLIC ${TURBOQUANT_HWY_TARGET})

if(NOT MSVC)
  target_compile_options(turboquant PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

# --- Tests ------------------------------------------------------------------
if(TURBOQUANT_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(turboquant_tests
      tests/test_packing.cc
      tests/test_rotation.cc
      tests/test_roundtrip.cc
      tests/test_adc.cc
    )
    target_include_directories(turboquant_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(turboquant_tests PRIVATE turboquant GTest::gtest GTest::gtest_main)
    if(NOT MSVC)
      target_compile_options(turboquant_tests PRIVATE -O2 -g)
    endif()
    add_test(NAME turboquant_tests COMMAND turboquant_tests)
  else()
    message(STATUS "GTest not found; skipping tests")
  endif()
endif()

# --- Microbenchmarks --------------------------------------------------------
if(TURBOQUANT_BUILD_BENCH)
  find_package(benchmark QUIET)
  if(benchmark_FOUND)
    add_executable(turboquant_bench bench/bench_micro.cc)
    target_link_libraries(turboquant_bench PRIVATE turboquant benchmark::benchmark)
    if(NOT MSVC)
      target_compile_options(turboquant_bench PRIVATE -O3)
    endif()
  else()
    message(STATUS "google-benchmark not found; skipping microbench")
  endif()
endif()

# --- Dataset bench (HDF5 + BLAS) -------------------------------------------
if(TURBOQUANT_BUILD_DATASET_BENCH)
  find_package(HDF5 COMPONENTS C QUIET)
  set(BLAS_VENDOR "OpenBLAS")
  find_package(BLAS QUIET)
  if(HDF5_FOUND AND BLAS_FOUND)
    add_executable(turboquant_dataset_bench bench/bench_dataset.cc)
    target_include_directories(turboquant_dataset_bench PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(turboquant_dataset_bench PRIVATE
      turboquant ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
    target_compile_definitions(turboquant_dataset_bench PRIVATE
      $<$<BOOL:${HDF5_DEFINITIONS}>:${HDF5_DEFINITIONS}>)
    if(NOT MSVC)
      target_compile_options(turboquant_dataset_bench PRIVATE -O3)
    endif()
  else()
    message(STATUS "HDF5 or BLAS not found; skipping dataset bench")
  endif()
endif()
