cmake_minimum_required(VERSION 3.16)
project(tflite_grid_sample_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TFLITE_BUILD_SHARED_RUNTIME "Build TensorFlow Lite runtime DLLs." ON)

# Keep third-party dependencies static; only tensorflowlite_c is built as DLL.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

# Keep Windows builds stable in this repository setup.
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "Disable XNNPACK for easier MSVC builds" FORCE)
set(TFLITE_ENABLE_EXTERNAL_DELEGATE OFF CACHE BOOL "Disable external delegate support" FORCE)

include(FetchContent)
FetchContent_Declare(
  tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        v2.16.1
  SOURCE_SUBDIR  tensorflow/lite
)
FetchContent_MakeAvailable(tensorflow-lite)

if(NOT TARGET tensorflow-lite)
  message(FATAL_ERROR "Expected target 'tensorflow-lite' from FetchContent.")
endif()

set(TFLITE_UPSTREAM_DIR "${tensorflow-lite_SOURCE_DIR}")
set(TFLITE_REGISTER_CC "${TFLITE_UPSTREAM_DIR}/tensorflow/lite/core/kernels/register.cc")

if(NOT EXISTS "${TFLITE_REGISTER_CC}")
  message(FATAL_ERROR "Could not locate register.cc at ${TFLITE_REGISTER_CC}")
endif()

file(READ "${TFLITE_REGISTER_CC}" TFLITE_REGISTER_CONTENT)

if(NOT TFLITE_REGISTER_CONTENT MATCHES "Register_GRID_SAMPLE")
  if(TFLITE_REGISTER_CONTENT MATCHES "TfLiteRegistration\\* Register_DETECTION_POSTPROCESS\\(\\);")
    string(REGEX REPLACE
      "TfLiteRegistration\\* Register_DETECTION_POSTPROCESS\\(\\);"
      "TfLiteRegistration* Register_DETECTION_POSTPROCESS();\nTfLiteRegistration* Register_GRID_SAMPLE();"
      TFLITE_REGISTER_CONTENT
      "${TFLITE_REGISTER_CONTENT}"
    )
  endif()
endif()

if(NOT TFLITE_REGISTER_CONTENT MATCHES "AddCustom\\(\"TFLiteGridSample\"")
  string(REPLACE
    "  may_directly_contain_user_defined_ops_ = false;"
    "  AddCustom(\"TFLiteGridSample\", tflite::ops::custom::Register_GRID_SAMPLE());\n  may_directly_contain_user_defined_ops_ = false;"
    TFLITE_REGISTER_CONTENT
    "${TFLITE_REGISTER_CONTENT}"
  )
endif()

if(NOT TFLITE_REGISTER_CONTENT MATCHES "Register_GRID_SAMPLE")
  message(FATAL_ERROR "Failed to inject Register_GRID_SAMPLE declaration into BuiltinOpResolver source.")
endif()

if(NOT TFLITE_REGISTER_CONTENT MATCHES "AddCustom\\(\"TFLiteGridSample\"")
  message(FATAL_ERROR "Failed to inject TFLiteGridSample AddCustom entry into BuiltinOpResolver source.")
endif()

file(WRITE "${TFLITE_REGISTER_CC}" "${TFLITE_REGISTER_CONTENT}")

target_sources(tensorflow-lite PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_sample.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_sample.h
)
target_include_directories(tensorflow-lite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
  set_target_properties(tensorflow-lite PROPERTIES OUTPUT_NAME "tensorflowlite")
endif()

add_library(tensorflowlite_c SHARED
  ${TFLITE_UPSTREAM_DIR}/tensorflow/lite/core/c/c_api.cc
  ${TFLITE_UPSTREAM_DIR}/tensorflow/lite/core/c/c_api_experimental.cc
  ${TFLITE_UPSTREAM_DIR}/tensorflow/lite/core/c/common.cc
  ${TFLITE_UPSTREAM_DIR}/tensorflow/lite/core/c/registration_external.cc
)

target_include_directories(tensorflowlite_c
  PUBLIC
    ${TFLITE_UPSTREAM_DIR}
    ${TFLITE_UPSTREAM_DIR}/tensorflow/lite
)

target_compile_definitions(tensorflowlite_c PRIVATE TFL_COMPILE_LIBRARY)
if(MSVC)
  target_compile_options(tensorflowlite_c PRIVATE /UTFL_STATIC_LIBRARY_BUILD)
else()
  target_compile_options(tensorflowlite_c PRIVATE -UTFL_STATIC_LIBRARY_BUILD)
endif()

target_link_libraries(tensorflowlite_c PRIVATE tensorflow-lite)

if(MSVC)
  target_compile_options(tensorflow-lite PRIVATE /EHsc)
  target_compile_options(tensorflowlite_c PRIVATE /EHsc)
endif()