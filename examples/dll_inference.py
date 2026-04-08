from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


PathLike = Union[str, os.PathLike[str]]


def _tensor_shape(lib: ctypes.CDLL, tensor: ctypes.c_void_p) -> list[int]:
    dims = lib.TfLiteTensorNumDims(tensor)
    return [int(lib.TfLiteTensorDim(tensor, i)) for i in range(dims)]


def _dtype_from_tflite_type(type_id: int) -> np.dtype:
    # TfLiteType enum mapping for common numeric tensor types.
    mapping = {
        1: np.float32,
        2: np.int32,
        3: np.uint8,
        4: np.int64,
        6: np.bool_,
        7: np.int16,
        9: np.int8,
        10: np.float16,
        11: np.float64,
        13: np.uint64,
        16: np.uint32,
        17: np.uint16,
    }
    if type_id not in mapping:
        raise RuntimeError(f"Unsupported TfLite tensor type id: {type_id}")
    return np.dtype(mapping[type_id])


def _configure_c_api(lib: ctypes.CDLL) -> None:
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_size_t = ctypes.c_size_t

    lib.TfLiteVersion.restype = ctypes.c_char_p

    lib.TfLiteModelCreateFromFile.argtypes = [ctypes.c_char_p]
    lib.TfLiteModelCreateFromFile.restype = c_void_p
    lib.TfLiteModelDelete.argtypes = [c_void_p]

    lib.TfLiteInterpreterOptionsCreate.restype = c_void_p
    lib.TfLiteInterpreterOptionsDelete.argtypes = [c_void_p]
    lib.TfLiteInterpreterOptionsSetNumThreads.argtypes = [c_void_p, c_int]

    lib.TfLiteInterpreterCreate.argtypes = [c_void_p, c_void_p]
    lib.TfLiteInterpreterCreate.restype = c_void_p
    lib.TfLiteInterpreterDelete.argtypes = [c_void_p]

    lib.TfLiteInterpreterAllocateTensors.argtypes = [c_void_p]
    lib.TfLiteInterpreterAllocateTensors.restype = c_int
    lib.TfLiteInterpreterInvoke.argtypes = [c_void_p]
    lib.TfLiteInterpreterInvoke.restype = c_int

    lib.TfLiteInterpreterGetInputTensorCount.argtypes = [c_void_p]
    lib.TfLiteInterpreterGetInputTensorCount.restype = c_int
    lib.TfLiteInterpreterGetOutputTensorCount.argtypes = [c_void_p]
    lib.TfLiteInterpreterGetOutputTensorCount.restype = c_int
    lib.TfLiteInterpreterGetInputTensor.argtypes = [c_void_p, c_int]
    lib.TfLiteInterpreterGetInputTensor.restype = c_void_p
    lib.TfLiteInterpreterGetOutputTensor.argtypes = [c_void_p, c_int]
    lib.TfLiteInterpreterGetOutputTensor.restype = c_void_p

    lib.TfLiteTensorNumDims.argtypes = [c_void_p]
    lib.TfLiteTensorNumDims.restype = c_int
    lib.TfLiteTensorDim.argtypes = [c_void_p, c_int]
    lib.TfLiteTensorDim.restype = c_int
    lib.TfLiteTensorType.argtypes = [c_void_p]
    lib.TfLiteTensorType.restype = c_int
    lib.TfLiteTensorByteSize.argtypes = [c_void_p]
    lib.TfLiteTensorByteSize.restype = c_size_t
    lib.TfLiteTensorCopyFromBuffer.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.TfLiteTensorCopyFromBuffer.restype = c_int
    lib.TfLiteTensorCopyToBuffer.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.TfLiteTensorCopyToBuffer.restype = c_int


def _coerce_image_to_shape(image: np.ndarray, expected_shape: Sequence[int]) -> np.ndarray:
    if tuple(image.shape) == tuple(expected_shape):
        return image
    if len(expected_shape) == image.ndim + 1 and expected_shape[0] == 1:
        if tuple(expected_shape[1:]) == tuple(image.shape):
            return np.expand_dims(image, axis=0)
    raise ValueError(
        f"Image shape {tuple(image.shape)} does not match expected input shape {tuple(expected_shape)}"
    )


def run_inference_from_dll(
    dll_path: PathLike,
    tflite_model_path: PathLike,
    image: np.ndarray,
    theta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run inference with tensorflowlite_c.dll and return the first output tensor.

    Required inputs:
    - dll_path: Path to tensorflowlite_c.dll.
    - tflite_model_path: Path to the .tflite model.
    - image: Image tensor as a NumPy array.

    Optional:
    - theta: Affine theta tensor. If omitted and the model expects shape [1, 6],
      an identity theta is used.
    """
    dll_path = Path(dll_path).resolve()
    model_path = Path(tflite_model_path).resolve()

    if not dll_path.exists():
        raise FileNotFoundError(f"DLL not found: {dll_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    image_np = np.ascontiguousarray(image)

    if os.name == "nt":
        os.add_dll_directory(str(dll_path.parent))

    lib = ctypes.CDLL(str(dll_path))
    _configure_c_api(lib)

    k_tflite_ok = 0
    c_void_p = ctypes.c_void_p

    model_handle = lib.TfLiteModelCreateFromFile(str(model_path).encode("utf-8"))
    if not model_handle:
        raise RuntimeError("TfLiteModelCreateFromFile failed")

    options = lib.TfLiteInterpreterOptionsCreate()
    interpreter = None

    try:
        lib.TfLiteInterpreterOptionsSetNumThreads(options, 1)
        interpreter = lib.TfLiteInterpreterCreate(model_handle, options)
        if not interpreter:
            raise RuntimeError("TfLiteInterpreterCreate failed")

        status = lib.TfLiteInterpreterAllocateTensors(interpreter)
        if status != k_tflite_ok:
            raise RuntimeError(f"TfLiteInterpreterAllocateTensors failed with status {status}")

        input_count = int(lib.TfLiteInterpreterGetInputTensorCount(interpreter))
        if input_count < 1:
            raise RuntimeError("Model does not expose input tensors")

        image_assigned = False

        for i in range(input_count):
            tensor = lib.TfLiteInterpreterGetInputTensor(interpreter, i)
            shape = _tensor_shape(lib, tensor)
            dtype = _dtype_from_tflite_type(int(lib.TfLiteTensorType(tensor)))

            if not image_assigned:
                try:
                    image_value = _coerce_image_to_shape(image_np, shape)
                    image_value = np.ascontiguousarray(image_value.astype(dtype, copy=False))
                    status = lib.TfLiteTensorCopyFromBuffer(
                        tensor,
                        image_value.ctypes.data_as(c_void_p),
                        image_value.nbytes,
                    )
                    if status != k_tflite_ok:
                        raise RuntimeError(f"Failed to copy image into input {i}, status {status}")
                    image_assigned = True
                    continue
                except ValueError:
                    pass

            if tuple(shape) == (1, 6):
                theta_value = theta
                if theta_value is None:
                    theta_value = np.array([[1, 0, 0, 0, 1, 0]], dtype=dtype)
                theta_value = np.ascontiguousarray(np.asarray(theta_value, dtype=dtype))

                if tuple(theta_value.shape) != (1, 6):
                    raise ValueError(
                        f"Theta input expects shape (1, 6), got {tuple(theta_value.shape)}"
                    )

                status = lib.TfLiteTensorCopyFromBuffer(
                    tensor,
                    theta_value.ctypes.data_as(c_void_p),
                    theta_value.nbytes,
                )
                if status != k_tflite_ok:
                    raise RuntimeError(f"Failed to copy theta into input {i}, status {status}")
                continue

            raise RuntimeError(
                "Encountered an additional model input that could not be auto-filled. "
                f"Input index {i} has shape {shape}."
            )

        if not image_assigned:
            raise RuntimeError("Could not map the provided image to any model input tensor")

        status = lib.TfLiteInterpreterInvoke(interpreter)
        if status != k_tflite_ok:
            raise RuntimeError(f"TfLiteInterpreterInvoke failed with status {status}")

        output_tensor = lib.TfLiteInterpreterGetOutputTensor(interpreter, 0)
        output_shape = _tensor_shape(lib, output_tensor)
        output_dtype = _dtype_from_tflite_type(int(lib.TfLiteTensorType(output_tensor)))

        output_nbytes = int(lib.TfLiteTensorByteSize(output_tensor))
        output = np.empty(output_nbytes // output_dtype.itemsize, dtype=output_dtype)

        status = lib.TfLiteTensorCopyToBuffer(
            output_tensor,
            output.ctypes.data_as(c_void_p),
            output_nbytes,
        )
        if status != k_tflite_ok:
            raise RuntimeError(f"Failed to copy output tensor, status {status}")

        return output.reshape(output_shape)
    finally:
        if interpreter:
            lib.TfLiteInterpreterDelete(interpreter)
        if options:
            lib.TfLiteInterpreterOptionsDelete(options)
        if model_handle:
            lib.TfLiteModelDelete(model_handle)


def _load_image_from_path(image_path: PathLike) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path)
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected a NumPy array in {path}")
        return image

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Loading non-.npy images requires Pillow. Install with: pip install pillow"
        ) from exc

    with Image.open(path) as img:
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr


if __name__ == "__main__":
    # Direct-run example: edit these paths and run this file.
    example_dll = Path("build/Release/tensorflowlite_c.dll")
    example_model = Path("artifacts/grid_sample_custom.tflite")
    example_image = Path("artifacts/image.npy")
    example_theta: Optional[Path] = None  # e.g. Path("artifacts/theta.npy")

    image = _load_image_from_path(example_image)
    theta = np.load(example_theta) if example_theta else None

    output = run_inference_from_dll(
        dll_path=example_dll,
        tflite_model_path=example_model,
        image=image,
        theta=theta,
    )

    print("Inference succeeded")
    print("Output shape:", tuple(output.shape))
    print("Output dtype:", output.dtype)
    print("Output min/max:", float(np.min(output)), float(np.max(output)))
