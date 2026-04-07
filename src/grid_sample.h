#ifndef GRID_SAMPLE_H_
#define GRID_SAMPLE_H_

#include "tensorflow/lite/core/c/common.h"

#ifdef _WIN32
#define GRID_SAMPLE_EXPORT __declspec(dllexport)
#else
#define GRID_SAMPLE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

GRID_SAMPLE_EXPORT TfLiteRegistration* Register_GRID_SAMPLE();

#ifdef __cplusplus
}

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_GRID_SAMPLE();

}  // namespace custom
}  // namespace ops
}  // namespace tflite
#endif

#endif  // GRID_SAMPLE_H_
