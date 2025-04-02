# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from estimater import *
from datareader import *
import argparse
from scipy.spatial.transform import Rotation as R


def remove_axis_rotation(T, axis=0):
    # Extract rotation and translation
    rot_matrix = T[:3, :3]
    translation = T[:3, 3]

    # Convert to Euler angles (XYZ convention)
    r = R.from_matrix(rot_matrix)
    euler = r.as_euler('xyz', degrees=False)

    # Zero out the Z rotation
    euler[axis] = 0.0

    # Rebuild rotation matrix without Z rotation
    new_rot_matrix = R.from_euler('xyz', euler).as_matrix()

    # Build new transform
    new_T = np.eye(4)
    new_T[:3, :3] = new_rot_matrix
    new_T[:3, 3] = translation

    return new_T


if __name__=='__main__':
  
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--mask_folder', type=str, default=None)
  parser.add_argument('--ignore_x_axis', type=int, default=0, help='ignore the rotation along x axis')
  parser.add_argument('--ignore_y_axis', type=int, default=0, help='ignore the rotation along y axis')
  parser.add_argument('--ignore_z_axis', type=int, default=0, help='ignore the rotation along z axis')
  args = parser.parse_args()

  print(args.mesh_file)

  set_logging_format()
  set_seed(0)
  
  os.makedirs(f'{args.test_scene_dir}/pose', exist_ok=True)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf, mask_folder=args.mask_folder)
  poses = []
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
    
    
    if args.ignore_x_axis:
      pose = remove_axis_rotation(pose, axis=0)
    
    if args.ignore_y_axis:
      pose = remove_axis_rotation(pose, axis=1)
    
    if args.ignore_z_axis:
      pose = remove_axis_rotation(pose, axis=2)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    poses.append(pose.reshape(4,4))
    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  poses = np.stack(poses, axis=0)
  obj_name = args.mesh_file.split("/")[-1].split(".")[0]
  os.makedirs(f'{args.test_scene_dir}/processed/object', exist_ok=True)
  np.save(f'{args.test_scene_dir}/processed/object/{obj_name}_pose_cam.npy', poses)
