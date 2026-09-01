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
# Resolution order matters. TurboQuant is normally consumed via add_subdirectory
# or FetchContent from a parent that already builds Highway (cyborgdb-core does),
# and linking two different Highway versions into one binary is a real hazard —
# so an explicit choice from the parent must win over anything found on the host.
# Probing the system first would silently ignore the parent on any machine that
# happens to have a system libhwy installed.
#
# Default fetch tag matches the Highway cyborgdb-core builds against. Encoded
# payloads are byte-identical across 1.0.7, 1.3.0 and 1.4.0 (verified by
# tests/test_golden.cc), so the version does not affect the on-disk format.
set(TURBOQUANT_HWY_GIT_TAG "1.4.0" CACHE STRING
    "Highway version to fetch when no existing Highway is supplied or found")
# Skip all discovery and fetch TURBOQUANT_HWY_GIT_TAG unconditionally. CI needs
# this to actually test the version it asked for: CMAKE_DISABLE_FIND_PACKAGE_HWY
# suppresses find_package but not the find_path/find_library below, so a runner
# with a host libhwy would silently test that instead.
option(TURBOQUANT_FORCE_FETCH_HWY "Always fetch Highway, ignoring any host copy" OFF)

if(TURBOQUANT_FORCE_FETCH_HWY)
  include(FetchContent)
  FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG ${TURBOQUANT_HWY_GIT_TAG}
  )
  set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(highway)
  set(TURBOQUANT_HWY_TARGET hwy)
  set(TURBOQUANT_HWY_ORIGIN "force-fetched ${TURBOQUANT_HWY_GIT_TAG}")
elseif(TARGET hwy)
  # 1. Parent already declared a Highway target (FetchContent/add_subdirectory).
  #    Nothing to find: reuse it verbatim.
  set(TURBOQUANT_HWY_TARGET hwy)
  set(TURBOQUANT_HWY_ORIGIN "parent target 'hwy'")
elseif(TARGET hwy::hwy)
  set(TURBOQUANT_HWY_TARGET hwy::hwy)
  set(TURBOQUANT_HWY_ORIGIN "parent target 'hwy::hwy'")
elseif(HWY_INCLUDE_DIR AND HWY_LIBRARY)
  # 2. Parent pre-seeded the paths explicitly. Honour them without probing.
  add_library(turboquant_hwy_imported INTERFACE)
  target_include_directories(turboquant_hwy_imported INTERFACE ${HWY_INCLUDE_DIR})
  target_link_libraries(turboquant_hwy_imported INTERFACE ${HWY_LIBRARY})
  set(TURBOQUANT_HWY_TARGET turboquant_hwy_imported)
  set(TURBOQUANT_HWY_ORIGIN "caller-supplied HWY_INCLUDE_DIR=${HWY_INCLUDE_DIR}")
else()
  # 3. Standalone build: installed package, then system paths, then fetch.
  find_package(HWY QUIET)
  if(HWY_FOUND)
    set(TURBOQUANT_HWY_TARGET hwy::hwy)
    set(TURBOQUANT_HWY_ORIGIN "find_package(HWY)")
  else()
    find_path(HWY_INCLUDE_DIR hwy/highway.h)
    find_library(HWY_LIBRARY NAMES hwy)
    if(HWY_INCLUDE_DIR AND HWY_LIBRARY)
      add_library(turboquant_hwy_imported INTERFACE)
      target_include_directories(turboquant_hwy_imported INTERFACE ${HWY_INCLUDE_DIR})
      target_link_libraries(turboquant_hwy_imported INTERFACE ${HWY_LIBRARY})
      set(TURBOQUANT_HWY_TARGET turboquant_hwy_imported)
      set(TURBOQUANT_HWY_ORIGIN "system ${HWY_LIBRARY}")
    else()
      include(FetchContent)
      FetchContent_Declare(
        highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG ${TURBOQUANT_HWY_GIT_TAG}
      )
      set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
      set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
      FetchContent_MakeAvailable(highway)
      set(TURBOQUANT_HWY_TARGET hwy)
      set(TURBOQUANT_HWY_ORIGIN "fetched ${TURBOQUANT_HWY_GIT_TAG}")
    endif()
  endif()
endif()

# Say which one won. The failure mode this guards against is silent, so make the
# resolution visible in every configure log.
message(STATUS "TurboQuant Highway: ${TURBOQUANT_HWY_ORIGIN}")

# --- Library ----------------------------------------------------------------
add_library(turboquant
  src/turboquant.cc
  src/packing.cc
  src/rotation.cc
  src/kernels.cc
  src/codebook.cc
  src/rotator_mixed.cc
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
  # Prefer an installed GTest; fall back to fetching a pinned one so a bare
  # checkout can run the suite without a system package (mirrors the Highway
  # handling above).
  find_package(GTest QUIET)
  if(NOT GTest_FOUND AND NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  add_executable(turboquant_tests
    tests/test_packing.cc
    tests/test_rotation.cc
    tests/test_roundtrip.cc
    tests/test_golden.cc
    tests/test_robustness.cc
  )
  target_include_directories(turboquant_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  # gtest_main already carries gtest; naming both makes the linker warn about a
  # duplicate archive.
  target_link_libraries(turboquant_tests PRIVATE turboquant GTest::gtest_main)
  if(NOT MSVC)
    target_compile_options(turboquant_tests PRIVATE -O2 -g)
  endif()
  # If GTest was found outside the default dyld search path (e.g. conda or a
  # custom prefix), bake the library's directory into the test binary's rpath
  # so `ctest` works without DYLD_LIBRARY_PATH gymnastics.
  # Only meaningful for an installed GTest; a FetchContent'd one is built in
  # this tree and LOCATION is not readable on non-imported targets.
  set(_gtest_loc "")
  if(GTest_FOUND)
    get_target_property(_gtest_loc GTest::gtest LOCATION)
  endif()
  if(_gtest_loc)
    get_filename_component(_gtest_dir "${_gtest_loc}" DIRECTORY)
    set_target_properties(turboquant_tests PROPERTIES
      BUILD_RPATH "${_gtest_dir}"
      INSTALL_RPATH "${_gtest_dir}")
  endif()
  add_test(NAME turboquant_tests COMMAND turboquant_tests)
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

  # Prefer Apple's Accelerate framework on macOS; fall back to whatever BLAS
  # is available elsewhere (OpenBLAS / MKL / system). Users can override by
  # setting BLA_VENDOR before invoking cmake.
  set(TURBOQUANT_BLAS_FOUND FALSE)
  set(TURBOQUANT_BLAS_LIBS "")
  set(TURBOQUANT_BLAS_INCLUDES "")
  set(TURBOQUANT_BLAS_DEFS "")
  if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
      set(TURBOQUANT_BLAS_FOUND TRUE)
      set(TURBOQUANT_BLAS_LIBS ${ACCELERATE_FRAMEWORK})
      set(TURBOQUANT_BLAS_DEFS TURBOQUANT_USE_ACCELERATE=1)
      message(STATUS "Dataset bench: using Accelerate framework for BLAS")
    endif()
  endif()
  if(NOT TURBOQUANT_BLAS_FOUND)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      set(TURBOQUANT_BLAS_FOUND TRUE)
      set(TURBOQUANT_BLAS_LIBS ${BLAS_LIBRARIES})
      # cblas.h is not standardized across distros; probe the usual spots.
      find_path(CBLAS_INCLUDE_DIR cblas.h
                HINTS /usr/include /usr/local/include /opt/homebrew/include
                      /usr/include/openblas /usr/include/x86_64-linux-gnu)
      if(CBLAS_INCLUDE_DIR)
        set(TURBOQUANT_BLAS_INCLUDES ${CBLAS_INCLUDE_DIR})
      endif()
      message(STATUS "Dataset bench: using BLAS=${BLAS_LIBRARIES}")
    endif()
  endif()

  if(HDF5_FOUND AND TURBOQUANT_BLAS_FOUND)
    add_executable(turboquant_dataset_bench bench/bench_dataset.cc)
    target_include_directories(turboquant_dataset_bench PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${HDF5_INCLUDE_DIRS} ${TURBOQUANT_BLAS_INCLUDES})
    target_link_libraries(turboquant_dataset_bench PRIVATE
      turboquant ${HDF5_LIBRARIES} ${TURBOQUANT_BLAS_LIBS})
    target_compile_definitions(turboquant_dataset_bench PRIVATE
      $<$<BOOL:${HDF5_DEFINITIONS}>:${HDF5_DEFINITIONS}>
      ${TURBOQUANT_BLAS_DEFS})
    if(NOT MSVC)
      target_compile_options(turboquant_dataset_bench PRIVATE -O3)
    endif()
  else()
    message(STATUS "HDF5 or BLAS not found; skipping dataset bench")
  endif()
endif()
