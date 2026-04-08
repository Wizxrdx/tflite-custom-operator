from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from examples.dll_inference import _load_image_from_path, run_inference_from_dll
except ModuleNotFoundError:
    from dll_inference import _load_image_from_path, run_inference_from_dll


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflowlite_c.dll and a .tflite model."
    )
    parser.add_argument("--dll", required=True, help="Path to tensorflowlite_c.dll")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (.npy preferred, or standard image file if Pillow is installed)",
    )
    parser.add_argument(
        "--theta",
        default=None,
        help="Optional .npy path for theta tensor (shape [1, 6])",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional .npy path to save output tensor",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = _load_image_from_path(args.image)
    theta = np.load(args.theta) if args.theta else None

    output = run_inference_from_dll(
        dll_path=args.dll,
        tflite_model_path=args.model,
        image=image,
        theta=theta,
    )

    print("Inference succeeded")
    print("Output shape:", tuple(output.shape))
    print("Output dtype:", output.dtype)
    print("Output min/max:", float(np.min(output)), float(np.max(output)))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, output)
        print("Saved output:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
