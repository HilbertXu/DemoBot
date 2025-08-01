import trimesh
import argparse
import numpy as np


def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="ROS Image Saver with Synchronization")

    # Add arguments
    parser.add_argument("--input_f", type=str, default=None, help="path to the input model")
    parser.add_argument("--output_f", type=str, default=None, help="path to the output model")
    parser.add_argument("--scale_factor", type=float, nargs='+', default=None, help='factor to scale the 3D object model')
    parser.add_argument("--color_name", type=str, default=None, help='factor to scale the 3D object model')

    return parser.parse_args()

RGBA_CODE = {
    "red": np.asarray([255, 0, 0, 255]),
    "green": np.asarray([0, 255, 0, 255]),
    "brown": np.asarray([150, 75, 0, 255]),
    'blue': np.asarray([125, 249, 255, 255]),
    'yellow': np.asarray([255, 255, 0, 255]),
}

args = parse_args()

mesh = trimesh.load(args.input_f)
size_x, size_y, size_z = mesh.extents
print(f"Mesh size (XYZ): {size_x:.3f}, {size_y:.3f}, {size_z:.3f}")

if args.scale_factor is not None:
    mesh.apply_scale(args.scale_factor)
    size_x, size_y, size_z = mesh.extents
    print(f"Mesh size (XYZ) after rescale to {args.scale_factor}: {size_x:.4f}, {size_y:.4f}, {size_z:.4f}")

if args.color_name is not None:
    assert args.color_name in RGBA_CODE.keys(), "Please select valid color"
    mesh.visual.vertex_colors = np.tile(RGBA_CODE[args.color_name], (len(mesh.vertices), 1))

size_x, size_y, size_z = mesh.extents
print(f"Mesh size (XYZ) after rescale to {args.scale_factor}: {size_x:.4f}, {size_y:.4f}, {size_z:.4f}")
mesh.show()
mesh.export(args.output_f)