import trimesh
import argparse
import os
import shutil
import numpy as np
from glob import glob

def parse_cli_args():
    """Parse the input command line arguments."""
    # add argparse arguments
    parser = argparse.ArgumentParser("Utility to generate keypoints for the given mesh file.")
    parser.add_argument("--mesh_f", type=str, help="The input mesh file.")
    parser.add_argument("--primitive_type", type=str, help="the primitive shape for simplify the keypoint sampling")
    parser.add_argument("--insert_mesh_f", type=str, default=None)
    parser.add_argument("--grasp_mesh_f", type=str, default=None)
    parser.add_argument("--visualize", action='store_true', default=False)
    args_cli = parser.parse_args()
    # return arguments
    return args_cli


def get_keypoint_cylinder_x(min_xyz, max_xyz):
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz
    keypoints = [
        [max_x, (min_y+max_y)/2., (min_z+max_z)/2.],
        [(min_x+max_x)/2., (min_y+max_y)/2., (min_z+max_z)/2.],
        [min_x, (min_y+max_y)/2., (min_z+max_z)/2.]
    ]
    
    return np.asarray(keypoints)



def get_keypoint_cylinder_y(min_xyz, max_xyz):
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz
    keypoints = [
        [(min_x+max_x)/2., max_y, (min_z+max_z)/2.],
        [(min_x+max_x)/2., (min_y+max_y)/2., (min_z+max_z)/2.],
        [(min_x+max_x)/2., min_y, (min_z+max_z)/2.]
    ]
    
    return np.asarray(keypoints)


def get_keypoint_cylinder_z(min_xyz, max_xyz):
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz
    keypoints = [
        [(min_x+max_x)/2., (min_y+max_y)/2., max_z],
        [(min_x+max_x)/2., (min_y+max_y)/2., (min_z+max_z)/2.],
        [(min_x+max_x)/2., (min_y+max_y)/2., min_z]
    ]
    
    return np.asarray(keypoints)


def get_keypoint_cube(mesh):
    return mesh.bounding_box.vertices



def get_insert_keypoint_from_mesh(mesh, axis='y'):
    # we assume the insert hole is a cylinder to simplify the problem
    axis_to_index = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    vertices = mesh.vertices
    all_indices = np.arange(vertices.shape[0])
    upper_indices = vertices[:, axis_to_index[axis]] == np.min(vertices[:, axis_to_index[axis]])
    bottom_indices = vertices[:, axis_to_index[axis]] == np.max(vertices[:, axis_to_index[axis]])
    if upper_indices.sum() + bottom_indices.sum() < vertices.shape[0]:
        middle_indices = np.logical_not(np.logical_or(upper_indices, bottom_indices))
    else:
        middle_indices = all_indices
    
    
    return np.asarray([
        vertices[upper_indices, :].mean(axis=0),
        vertices[middle_indices, :].mean(axis=0),
        vertices[bottom_indices, :].mean(axis=0)

    ]) # [upper_point, middle_point, bottom_point]





def visualize(mesh, keypoints, pointcloud, insert_keypoints=None):
    spheres = []
    color_mapping = {
        0: [1.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0],
        2: [0.0, 0.0, 1.0],
        3: [1.0, 1.0, 0.0],
        4: [0.6, 0.35, 0.7],
        5: [0.9, 0.494, 0.133],
        6: [0.1, 0.737, 0.612],
        7: [0.584, 0.647, 0.651]
    }
    for idx, vertex in enumerate(keypoints):
        # Create a small sphere at the vertex location
        sphere = trimesh.creation.icosphere(radius=0.01, color=color_mapping[idx])  # Adjust radius as needed
        sphere.apply_translation(vertex)
        spheres.append(sphere)
    
    if insert_keypoints is not None:
        for idx, vertex in enumerate(insert_keypoints):
            # Create a small sphere at the vertex location
            sphere = trimesh.creation.icosphere(radius=0.001, color=color_mapping[idx])  # Adjust radius as needed
            sphere.apply_translation(vertex)
            spheres.append(sphere)

    # Create a Scene with the mesh and all spheres
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    for sphere in spheres:
        scene.add_geometry(sphere)
    
    # Show the scene
    scene.show()

    trimesh.points.PointCloud(pointcloud).show()
    

args = parse_cli_args()

mesh = trimesh.load(args.mesh_f)
# Get the Axis-Aligned Bounding Box (AABB)
aabb = mesh.bounds  # shape (2, 3): [min_xyz, max_xyz]

if args.primitive_type == 'cylinder_x':
    keypoints = get_keypoint_cylinder_x(aabb[0], aabb[1])
elif args.primitive_type == 'cylinder_y':
    keypoints = get_keypoint_cylinder_y(aabb[0], aabb[1])
elif args.primitive_type == 'cylinder_z':
    keypoints = get_keypoint_cylinder_z(aabb[0], aabb[1])
elif args.primitive_type == 'cube':
    keypoints = get_keypoint_cube(mesh)
    

if args.insert_mesh_f is not None:
    insert_mesh = trimesh.load(args.insert_mesh_f)
    insert_keypoints = get_insert_keypoint_from_mesh(insert_mesh, axis='y')
else:
    insert_keypoints = keypoints

if args.grasp_mesh_f is not None:
    grasp_mesh = trimesh.load(args.grasp_mesh_f)
    pointcloud, face_indices = trimesh.sample.sample_surface_even(grasp_mesh, count=500, radius=0.005)
else:
    pointcloud, face_indices = trimesh.sample.sample_surface_even(mesh, count=500, radius=0.005)


visualize(mesh, keypoints, pointcloud, insert_keypoints)

output_f = args.mesh_f.replace(".obj", ".npz")
np.savez(
    output_f, 
    object_keypoints=keypoints,
    insert_keypoints=insert_keypoints
)
    
