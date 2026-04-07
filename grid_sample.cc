#include "grid_sample.h"

#include <algorithm>
#include <cmath>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace grid_sample {

constexpr int kInputTensor = 0;
constexpr int kThetaTensor = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool align_corners;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  op_data->align_corners = true;
  (void)context;
  (void)buffer;
  (void)length;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
  (void)context;
}

bool HasSupportedThetaShape(const TfLiteTensor* theta) {
  if (NumDimensions(theta) == 2) {
    return SizeOfDimension(theta, 1) == 6;
  }
  if (NumDimensions(theta) == 3) {
    return SizeOfDimension(theta, 1) == 2 && SizeOfDimension(theta, 2) == 3;
  }
  return false;
}

const float* GetThetaForBatch(const TfLiteTensor* theta, const float* theta_data,
                              int batch_index) {
  if (NumDimensions(theta) == 2) {
    return theta_data + batch_index * 6;
  }
  return theta_data + batch_index * 2 * 3;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* theta = GetInput(context, node, kThetaTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE(context, HasSupportedThetaShape(theta));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, theta->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  TF_LITE_ENSURE_EQ(context, SizeOfDimension(theta, 0), SizeOfDimension(input, 0));

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* theta = GetInput(context, node, kThetaTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const float* input_data = GetTensorData<float>(input);
  const float* theta_data = GetTensorData<float>(theta);
  float* output_data = GetTensorData<float>(output);

  const int batch = SizeOfDimension(input, 0);
  const int height = SizeOfDimension(input, 1);
  const int width = SizeOfDimension(input, 2);
  const int channels = SizeOfDimension(input, 3);

  const OpData* op_data = reinterpret_cast<const OpData*>(node->user_data);
  const bool align_corners = op_data != nullptr ? op_data->align_corners : true;

  const float width_scale =
      align_corners ? static_cast<float>(std::max(width - 1, 1))
                    : static_cast<float>(std::max(width, 1));
  const float height_scale =
      align_corners ? static_cast<float>(std::max(height - 1, 1))
                    : static_cast<float>(std::max(height, 1));

  for (int b = 0; b < batch; ++b) {
    const float* affine = GetThetaForBatch(theta, theta_data, b);
    const float t0 = affine[0];
    const float t1 = affine[1];
    const float t2 = affine[2];
    const float t3 = affine[3];
    const float t4 = affine[4];
    const float t5 = affine[5];

    for (int y_out = 0; y_out < height; ++y_out) {
      float norm_y = 0.0f;
      if (align_corners) {
        norm_y = (height > 1) ? (2.0f * y_out / static_cast<float>(height - 1) - 1.0f)
                              : 0.0f;
      } else {
        norm_y = 2.0f * (static_cast<float>(y_out) + 0.5f) / height_scale - 1.0f;
      }

      for (int x_out = 0; x_out < width; ++x_out) {
        float norm_x = 0.0f;
        if (align_corners) {
          norm_x = (width > 1) ? (2.0f * x_out / static_cast<float>(width - 1) - 1.0f)
                               : 0.0f;
        } else {
          norm_x = 2.0f * (static_cast<float>(x_out) + 0.5f) / width_scale - 1.0f;
        }

        const float src_x = t0 * norm_x + t1 * norm_y + t2;
        const float src_y = t3 * norm_x + t4 * norm_y + t5;

        const float x = align_corners ? 0.5f * (src_x + 1.0f) * width_scale
                                      : ((src_x + 1.0f) * width_scale - 1.0f) * 0.5f;
        const float y = align_corners ? 0.5f * (src_y + 1.0f) * height_scale
                                      : ((src_y + 1.0f) * height_scale - 1.0f) * 0.5f;

        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        const float dx = x - x0;
        const float dy = y - y0;

        for (int c = 0; c < channels; ++c) {
          const auto sample = [&](int py, int px) {
            if (px < 0 || px >= width || py < 0 || py >= height) {
              return 0.0f;
            }
            return input_data[((b * height + py) * width + px) * channels + c];
          };

          const float p00 = sample(y0, x0);
          const float p01 = sample(y0, x1);
          const float p10 = sample(y1, x0);
          const float p11 = sample(y1, x1);

          output_data[((b * height + y_out) * width + x_out) * channels + c] =
              p00 * (1.0f - dx) * (1.0f - dy) + p01 * dx * (1.0f - dy) +
              p10 * (1.0f - dx) * dy + p11 * dx * dy;
        }
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace grid_sample
}  // namespace custom
}  // namespace ops
}  // namespace tflite

extern "C" {
GRID_SAMPLE_EXPORT TfLiteRegistration* Register_GRID_SAMPLE() {
  static TfLiteRegistration reg = {
      tflite::ops::custom::grid_sample::Init,
      tflite::ops::custom::grid_sample::Free,
      tflite::ops::custom::grid_sample::Prepare,
      tflite::ops::custom::grid_sample::Invoke,
  };
  reg.custom_name = "TFLiteGridSample";
  reg.version = 1;
  return &reg;
}
}

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_GRID_SAMPLE() { return ::Register_GRID_SAMPLE(); }

}  // namespace custom
}  // namespace ops
}  // namespace tflite
