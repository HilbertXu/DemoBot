import trimesh
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy



def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="ROS Image Saver with Synchronization")

    # Add arguments
    parser.add_argument("--object_mesh_f", type=str, help="path to the input model")
    parser.add_argument("--object_mesh_op_f", type=str, default=None, help="path to operate part of the input model")
    parser.add_argument("--output_dir", type=str, help='output folder')
    parser.add_argument("--vis", type=int, help='visualize the results')

    return parser.parse_args()

args = parse_args()

# Load the hammer head mesh
mesh = trimesh.load(args.object_mesh_f, process=True)
if args.object_mesh_op_f is not None:
    mesh_op = trimesh.load(args.object_mesh_op_f, process=True)
else:
    mesh_op = deepcopy(mesh)

object_name = args.object_mesh_f.split("/")[-1].split(".")[0]

op_vertices = mesh_op.vertices
all_vertices = mesh.vertices

if args.object_mesh_op_f is not None:
    outer_keypoints = []
    inner_keypoints = op_vertices

    for vertice in all_vertices:
        flag = False
        for p in op_vertices:
            if np.isclose(vertice, p, atol=1e-2).all():
                flag = True
        if not flag:
            outer_keypoints.append(vertice)
else:
    outer_keypoints = inner_keypoints = all_vertices


def visualize_keypoints(mesh, outer_pts, inner_pts=None):
    # Convert to Open3D point cloud
    pcd_outer = o3d.geometry.PointCloud()
    pcd_outer.points = o3d.utility.Vector3dVector(outer_pts)
    pcd_outer.paint_uniform_color([1, 0, 0])  # Red for outer keypoints
    
    # Convert mesh to Open3D format
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    
    geoms = [mesh_o3d, pcd_outer]
    
    if inner_pts is not None:
        pcd_inner = o3d.geometry.PointCloud()
        pcd_inner.points = o3d.utility.Vector3dVector(inner_pts)

        # Set different colors
        pcd_inner.paint_uniform_color([0, 1, 0])  # Green for inner ring keypoints

        geoms.append(pcd_inner)
    
    # Show visualization
    o3d.visualization.draw_geometries(geoms)

np.savez(
    f"{args.output_dir}/{object_name}_keypoints.npz",
    outer_keypoints=np.stack(outer_keypoints, axis=0),
    inner_keypoints=inner_keypoints
)

# Run visualization
if args.vis:
    visualize_keypoints(mesh, inner_keypoints, outer_keypoints)

