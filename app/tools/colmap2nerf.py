import argparse
import json
from pathlib import Path

import numpy as np
from colmap_parsing_utils import (
    parse_colmap_camera_params,
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="convert colmap to transforms.json")

    parser.add_argument("--recon_dir", type=str, default="landmark/dataset/your_dataset/sparse/0")
    parser.add_argument("--output_dir", type=str, default="landmark/dataset/your_dataset")
    parser.add_argument("--holdout", type=int, default=50)

    args = parser.parse_args()
    return args


def colmap_to_json(recon_dir, output_dir, holdout):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db

    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = im_data.name

        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }

        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    out = parse_colmap_camera_params(cam_id_to_camera[1])

    out_test = out

    frames_test = [f for i, f in enumerate(frames) if i % holdout == 0]

    out["frames"] = frames
    out_test["frames"] = frames_test

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1

    out["applied_transform"] = applied_transform.tolist()
    out_test["applied_transform"] = applied_transform.tolist()

    with open(output_dir / "transforms_train.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    with open(output_dir / "transforms_test.json", "w", encoding="utf-8") as f:
        json.dump(out_test, f, indent=4)

    return len(frames)


if __name__ == "__main__":
    init_args = parse_args()
    Recondir = Path(init_args.recon_dir)
    Outputdir = Path(init_args.output_dir)
    Holdout = init_args.holdout
    colmap_to_json(Recondir, Outputdir, Holdout)
