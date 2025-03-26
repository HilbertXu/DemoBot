import trimesh
import argparse


def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="ROS Image Saver with Synchronization")

    # Add arguments
    parser.add_argument("--input_f", type=str, help="path to the input model")
    parser.add_argument("--output_f", type=str, help="path to the output model")
    parser.add_argument("--scale_factor", type=float, help='factor to scale the 3D object model')

    return parser.parse_args()

args = parse_args()

mesh = trimesh.load(args.input_f)
size_x, size_y, size_z = mesh.extents

print(f"Mesh size (XYZ): {size_x:.3f}, {size_y:.3f}, {size_z:.3f}")
mesh.apply_scale(args.scale_factor)

size_x, size_y, size_z = mesh.extents
print(f"Mesh size (XYZ) after rescale to {args.scale_factor}: {size_x:.4f}, {size_y:.4f}, {size_z:.4f}")
mesh.export(args.output_f)