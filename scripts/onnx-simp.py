#!/usr/bin/env python3

import os

import onnx
from onnxsim import simplify


def simp(in_file, out_file):
    model = onnx.load(in_file)

    input_shapes = {}
    # input_shapes = { None: [1, 3, 220, 220] }

    model_simp, check = simplify(
        model,
        check_n=1,
        skip_shape_inference=True,
        input_shapes=input_shapes
    )

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, out_file)


def main():
    in_dir = os.environ.get('SIMP_IN_DIR', './models/tmp_in')
    out_dir = os.environ.get('SIMP_OUT_DIR', './models/tmp_out')
    for filename in os.listdir(in_dir):
        basename, ext = os.path.splitext(filename)
        if ext == '.onnx':
            in_file = os.path.join(in_dir, filename)
            out_file = os.path.join(out_dir, filename)
            simp(in_file, out_file)


if __name__ == '__main__':
    main()
