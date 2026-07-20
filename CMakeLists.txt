cmake_minimum_required(VERSION 3.25)
project(tinyllama_inference_engine VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TLIE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(TLIE_ENABLE_CUDA "Build the FP16 CUDA runtime, tests, and benchmarks" OFF)
set(TLIE_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/tinyllama-chat-v1.0" CACHE PATH
    "Prepared TinyLlama model directory")
set(TLIE_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/data/generated/tinyllama-chat-v1.0" CACHE PATH
    "Generated full-model golden directory")

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  sentencepiece
  URL https://codeload.github.com/google/sentencepiece/tar.gz/31646a467d2051eb904e0b45de3a73e91fe1c1e3
  URL_HASH SHA256=b4eb17e6bea5c9380ddd79d04d01d6aa5e280be8f3e3cef1f745a80e6d9451ce
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(
  nlohmann_json
  URL https://codeload.github.com/nlohmann/json/tar.gz/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
  URL_HASH SHA256=0dbc5e40a01ff142e7e68c03e85247a4dcede2f592d12d3677dee3664d17975a
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

add_library(tlie_cpu
  src/error.cpp
  src/kv_cache.cpp
  src/model.cpp
  src/model_config.cpp
  src/operators.cpp
  src/sha256.cpp
  src/tokenizer.cpp
  src/weight_store.cpp)
target_include_directories(tlie_cpu PUBLIC include)
target_include_directories(tlie_cpu SYSTEM PRIVATE ${sentencepiece_SOURCE_DIR}/src)
target_link_libraries(tlie_cpu PUBLIC sentencepiece-static nlohmann_json::nlohmann_json)
target_compile_options(tlie_cpu PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(APPLE)
  target_link_libraries(tlie_cpu PUBLIC "-framework Accelerate")
  target_compile_definitions(tlie_cpu PRIVATE TLIE_USE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
endif()

if(TLIE_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "TLIE sanitizers require Clang or GCC")
  endif()
  target_compile_options(tlie_cpu PUBLIC -O1 -fno-omit-frame-pointer -fsanitize=address,undefined)
  target_link_options(tlie_cpu PUBLIC -fno-omit-frame-pointer -fsanitize=address,undefined)
endif()

add_executable(tinyllama_reference app/reference_cli.cpp)
target_link_libraries(tinyllama_reference PRIVATE tlie_cpu)

if(BUILD_TESTING)
  function(tlie_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE tlie_cpu)
    add_test(NAME ${target} COMMAND ${target})
  endfunction()

  tlie_add_test(tlie_unit_tests tests/unit_tests.cpp)
  target_compile_definitions(tlie_unit_tests PRIVATE
    TLIE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    TLIE_TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")

  tlie_add_test(tlie_operator_tests tests/operator_tests.cpp)
  target_compile_definitions(tlie_operator_tests PRIVATE TLIE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

  tlie_add_test(tlie_tokenizer_tests tests/tokenizer_tests.cpp)
  target_compile_definitions(tlie_tokenizer_tests PRIVATE
    TLIE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    TLIE_MODEL_DIR="${TLIE_MODEL_DIR}")

  tlie_add_test(tlie_m1_integration_tests tests/m1_integration_tests.cpp)
  target_compile_definitions(tlie_m1_integration_tests PRIVATE
    TLIE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    TLIE_MODEL_DIR="${TLIE_MODEL_DIR}"
    TLIE_GOLDEN_DIR="${TLIE_GOLDEN_DIR}")
  set_tests_properties(tlie_m1_integration_tests PROPERTIES LABELS "integration;model")

  add_test(NAME tlie_reference_cli_rejects_invalid_max_tokens
    COMMAND ${CMAKE_COMMAND}
      -DEXECUTABLE=$<TARGET_FILE:tinyllama_reference>
      -DMODEL_DIR=${TLIE_MODEL_DIR}
      -DMAX_TOKENS=not-a-number
      -P ${CMAKE_SOURCE_DIR}/tests/expect_cli_failure.cmake)
  set_tests_properties(tlie_reference_cli_rejects_invalid_max_tokens PROPERTIES
    LABELS "cli;failure")

  add_test(NAME tlie_reference_cli_rejects_invalid_expected_token
    COMMAND ${CMAKE_COMMAND}
      -DEXECUTABLE=$<TARGET_FILE:tinyllama_reference>
      -DMODEL_DIR=${TLIE_MODEL_DIR}
      -DMAX_TOKENS=1
      -DEXPECTED_TOKEN=not-a-number
      -P ${CMAKE_SOURCE_DIR}/tests/expect_cli_failure.cmake)
  set_tests_properties(tlie_reference_cli_rejects_invalid_expected_token PROPERTIES
    LABELS "cli;failure")
endif()

if(TLIE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  FetchContent_Declare(
    cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG 5814e121dfb5049f72a5c3956c3c8961b40da78b
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(cpp_httplib)

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_library(tlie_cuda
    src/cuda/kernels.cu
    src/cuda/model.cu
    src/cuda/weight_store.cpp)
  target_include_directories(tlie_cuda PUBLIC include)
  target_link_libraries(tlie_cuda PUBLIC tlie_cpu CUDA::cudart CUDA::cublas)
  target_compile_options(tlie_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--expt-relaxed-constexpr>)
  set_target_properties(tlie_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(tinyllama_cuda app/cuda_reference.cu)
  target_link_libraries(tinyllama_cuda PRIVATE tlie_cuda)
  target_compile_definitions(tinyllama_cuda PRIVATE JSON_HAS_RANGES=0)
  if(TARGET CUDA::nvtx3)
    target_link_libraries(tinyllama_cuda PRIVATE CUDA::nvtx3)
    target_compile_definitions(tinyllama_cuda PRIVATE TLIE_USE_NVTX3=1)
  elseif(TARGET CUDA::nvToolsExt)
    target_link_libraries(tinyllama_cuda PRIVATE CUDA::nvToolsExt)
  else()
    message(FATAL_ERROR "The CUDA toolkit does not expose an NVTX CMake target")
  endif()

  add_executable(kernel_benchmarks benchmarks/kernel_benchmarks.cu)
  target_link_libraries(kernel_benchmarks PRIVATE tlie_cuda)
  target_compile_definitions(kernel_benchmarks PRIVATE JSON_HAS_RANGES=0)
  set_target_properties(kernel_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")

  add_executable(batch_benchmark app/batch_benchmark.cu)
  target_link_libraries(batch_benchmark PRIVATE tlie_cuda)
  target_compile_definitions(batch_benchmark PRIVATE
    JSON_HAS_RANGES=0
    TLIE_MODEL_DIR="${TLIE_MODEL_DIR}")
  set_target_properties(batch_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")

  add_executable(tinyllama_server app/server.cpp)
  target_link_libraries(tinyllama_server PRIVATE tlie_cuda httplib::httplib)
  target_compile_definitions(tinyllama_server PRIVATE JSON_HAS_RANGES=0)

  if(BUILD_TESTING)
    add_executable(kernel_tests tests/cuda/kernel_tests.cu)
    target_link_libraries(kernel_tests PRIVATE tlie_cuda)
    set_target_properties(kernel_tests PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_test(NAME tlie_cuda_kernel_tests COMMAND kernel_tests)
    set_tests_properties(tlie_cuda_kernel_tests PROPERTIES LABELS "cuda;kernel")
    add_test(NAME tlie_cuda_batch_tests COMMAND batch_benchmark --test)
    set_tests_properties(tlie_cuda_batch_tests PROPERTIES LABELS "cuda;batch;scheduler")
  endif()
endif()
