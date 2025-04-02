from glob import glob
from PIL import Image
import numpy as np
import os
import numpy as np
import os.path as op
from tqdm import tqdm
import sys

# @TODO
# check the object masks and target object masks
# to remove the overlapped regions between object masks and target object masks


SEGM_IDS = {"bg": 0, "object": 50, "right": 150, "left": 250}


def process_mask(data_dir, seq_name, flag):
    print(f"{data_dir}/{seq_name}/processed/sam/{flag}/images_masks/*.png")
    mask_ps = sorted(glob(f"{data_dir}/{seq_name}/processed/sam/{flag}/images_masks/*.png"))
    print(f"Processing {seq_name} {flag} with {len(mask_ps)} masks")
    for mask_p in mask_ps:
        mask = Image.open(mask_p)
        mask_np = np.array(mask)
        mask_np[mask_np > 0] = 1
        out_mask = mask_np
        out_mask = out_mask.astype(np.uint8) * 255
        out_p = mask_p.replace("/images_masks/", "/masks_processed/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(out_mask).save(out_p)


def validate_mask(data_dir, seq_name):
    print(f"Processing {seq_name}")

    # Step 1: Prepare file paths and load bounding boxes
    rgb_ps = sorted(glob(f"{data_dir}/{seq_name}/images/*"))

    # Step 1: format masks
    process_mask(data_dir, seq_name, "right")
    process_mask(data_dir, seq_name, "left")
    process_mask(data_dir, seq_name, "object")

    right_mask_ps = sorted(
        glob(f"{data_dir}/{seq_name}/processed/sam/right/images_masks/*.png")
    )
    left_mask_ps = sorted(
        glob(f"{data_dir}/{seq_name}/processed/sam/left/images_masks/*.png")
    )
    object_mask_ps = sorted(
        glob(f"{data_dir}/{seq_name}/processed/sam/object/images_masks/*.png")
    )
    
    target_mask_ps = sorted(
        glob(f"{data_dir}/{seq_name}/processed/sam/target/images_masks/*.png")
    )
    
    if len(left_mask_ps) > 0:
        assert len(left_mask_ps) == len(object_mask_ps)

    if len(right_mask_ps) > 0:
        assert len(right_mask_ps) == len(object_mask_ps)

    # rgb image with only object pixels
    rgb_ps = sorted(glob(f"{data_dir}/{seq_name}/images/*"))
    object_mask_ps = sorted(
        glob(f"{data_dir}/{seq_name}/processed/sam/object/masks_processed/*.png")
    )
    assert len(rgb_ps) == len(object_mask_ps)
    for rgb_p, object_mask_p in zip(rgb_ps, object_mask_ps):
        rgb_np = np.array(Image.open(rgb_p))
        object_mask_np = np.array(Image.open(object_mask_p))
        rgb_np[object_mask_np == 0] = 255

        out_p = rgb_p.replace("/images/", "/processed/images_object/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(rgb_np).save(out_p)
        
    
    for (obj_mask_f, tgt_mask_f) in zip(object_mask_ps, target_mask_ps):
        obj_mask = np.asarray(Image.open(obj_mask_f))
        tgt_mask = np.asarray(Image.open(tgt_mask_f))
        
        obj_mask[obj_mask > 0] = 1
        tgt_mask[tgt_mask > 0] = 1
        
        out_mask = np.clip((tgt_mask - obj_mask), a_max=1.0, a_min=0.0).astype(int) * 255
        out_f = tgt_mask_f.replace("images_masks", "masks_processed")
        Image.fromarray(out_mask).save(out_f)
    
    
    

    # merge the three masks
    object_mask_ps = sorted(
        glob(f"{data_dir}/{seq_name}/processed/sam/object/masks_processed/*.png")
    )
    for object_p in object_mask_ps:

        object_mask = np.array(Image.open(object_p))

        right_p = object_p.replace("/object/", "/right/")
        left_p = object_p.replace("/object/", "/left/")

        out_mask = np.zeros_like(object_mask)

        if op.exists(right_p):
            right_mask = np.array(Image.open(right_p))
            out_mask[right_mask > 0] = SEGM_IDS["right"]
        if op.exists(left_p):
            left_mask = np.array(Image.open(left_p))
            out_mask[left_mask > 0] = SEGM_IDS["left"]
        # object mask overwrites hands
        out_mask[object_mask > 0] = SEGM_IDS["object"]

        out_p = object_p.replace(
            "/processed/sam/object/masks_processed/", "/processed/masks/"
        )
        os.makedirs(os.path.dirname(out_p), exist_ok=True)

        Image.fromarray(out_mask).save(out_p)
    print("Done!")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seq_name", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    validate_mask(args.data_dir, args.seq_name)
