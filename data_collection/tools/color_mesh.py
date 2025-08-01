import trimesh
import numpy as np

# Load the mesh
mesh = trimesh.load('/home/advr/projects/DemoBot/assets/hammer/hammer_handle.obj')

# Set all vertex colors to green (RGBA)
green = np.array([0, 255, 0, 255], dtype=np.uint8)  # R, G, B, A
mesh.visual.vertex_colors = np.tile(green, (len(mesh.vertices), 1))

# Export the mesh with vertex color
mesh.export('/home/advr/projects/DemoBot/assets/hammer/hammer_handle_colored.obj')
mesh.show()



# import trimesh

# # Load your mesh
# mesh = trimesh.load('model.obj')

# # Create a simple green material (color values are in [0, 1])
# green_material = trimesh.visual.material.SimpleMaterial(color=[0, 1.0, 0])

# # Assign the material to the mesh
# mesh.visual.material = green_material

# # Export the mesh; trimesh will generate a corresponding .mtl file.
# mesh.export('model_green.obj')
# mesh.show()
