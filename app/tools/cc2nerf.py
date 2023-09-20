import argparse
import json
import os

import numpy as np
import torch
from cc_parsing_utils import auto_orient_and_center_poses


def parse_args():
    parser = argparse.ArgumentParser(description="convert cc json to transforms.json for nerf")

    parser.add_argument("--path", type=str, default="landmark/datasets/your_dataset")
    parser.add_argument("--input_transforms", type=str, default="transforms.json")
    parser.add_argument("--same_intri", action="store_true", default=True)
    parser.add_argument("--downsample", type=int, default=1)

    parser.add_argument(
        "--orientation_method",
        type=str,
        default="none",
        choices=["pca", "none"],
        help="The method to use for orientation",
    )
    parser.add_argument(
        "--center_method",
        type=str,
        default="poses",
        choices=["poses", "none"],
        help="The method to use to center the poses",
    )

    parser.add_argument("--auto_scale_poses", action="store_false", default=True)

    parser.add_argument("--train_skip", type=int, default=50, help="index%train_skip==0 -> test")

    init_args = parser.parse_args()
    return init_args


if __name__ == "__main__":
    args = parse_args()

    INPUT_PATH = args.path
    OUTPUT_PATH = INPUT_PATH
    DOWNSAMPLE = args.downsample
    TRAIN_SKIP = args.train_skip

    with open(os.path.join(INPUT_PATH, f"{args.input_transforms}"), "r", encoding="utf8") as f:
        tj = json.load(f)

    if args.same_intri:
        frame_1 = tj["0"]
        rot_mat = np.array(frame_1["rot_mat"])

        w = int(rot_mat[1, -1] / DOWNSAMPLE)
        h = int(rot_mat[0, -1] / DOWNSAMPLE)
        fl_x = rot_mat[2, -1] / DOWNSAMPLE
        fl_y = rot_mat[2, -1] / DOWNSAMPLE

        cx = w / 2
        cy = h / 2
        k1 = 0
        k2 = 0
        k3 = 0
        p1 = 0
        p2 = 0
        angle_x = 2 * np.arctan(w / (2 * fl_x))

        train_transforms = {"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": w, "h": h, "frames": []}

        test_transforms = {"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": w, "h": h, "frames": []}

        all_frames = []

        c2ws = []
        for i in range(len(tj)):
            frame = tj[f"{i}"]
            c2w = np.array(frame["rot_mat"])
            c2w = c2w[..., :-1]
            c2ws.append(c2w)

        c2ws = np.array(c2ws)

        poses = torch.from_numpy(np.array(c2ws).astype(np.float32))
        poses, transform_matrix = auto_orient_and_center_poses(
            poses,  # type: ignore
            method=args.orientation_method,
            center_method=args.center_method,
        )

        scale_factor = 1.0
        if args.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))

        poses[:, :3, 3] *= scale_factor

        c2ws = poses.numpy()

        for i in range(len(tj)):
            frame = tj[f"{i}"]
            file_path = frame["path"].split("/")[-1]
            suffix = "images" if DOWNSAMPLE == 1 else f"images_{DOWNSAMPLE}"
            file_path = os.path.join(f"{suffix}", file_path)

            c2w = np.concatenate((c2ws[i], np.array([[0, 0, 0, 1]])), axis=0)
            all_frames.append({"file_path": file_path, "transform_matrix": c2w.tolist()})

        for i, frame in enumerate(all_frames):
            if i % TRAIN_SKIP == 0:
                test_transforms["frames"].append(frame)
            else:
                train_transforms["frames"].append(frame)

    else:
        transforms = {"camera_model": "SIMPLE_PINHOLE", "orientation_override": "none", "frames": []}
        all_frames = []
        with open(os.path.join(INPUT_PATH, f"{args.input_transforms}"), "r", encoding="utf8") as f:
            tj = json.load(f)
        keys = tj.keys()
        c2ws = []

        for key in keys:
            frame = tj[key]
            c2w = np.array(frame["rot_mat"])
            c2w = c2w[..., :-1]
            c2ws.append(c2w)

        poses = torch.from_numpy(np.array(c2ws).astype(np.float32))
        poses[..., :3, 3] -= torch.mean(poses[..., :3, 3], dim=0)

        scale_factor = 1.0
        if args.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        print("scale factor:", scale_factor)
        poses[:, :3, 3] *= scale_factor

        c2ws = poses.numpy()

        for i, key in enumerate(keys):
            frame = tj[key]
            file_path = frame["path"]
            rot_mat = np.array(frame["rot_mat"])

            w = int(rot_mat[1, -1] / DOWNSAMPLE)
            h = int(rot_mat[0, -1] / DOWNSAMPLE)
            fl_x = rot_mat[2, -1] / DOWNSAMPLE
            fl_y = rot_mat[2, -1] / DOWNSAMPLE

            cx = w / 2
            cy = h / 2

            suffix = "images" if DOWNSAMPLE == 1 else f"images_{DOWNSAMPLE}"
            file_path = os.path.join(f"{suffix}", file_path)
            c2w = np.concatenate((c2ws[i], np.array([[0, 0, 0, 1]])), axis=0)
            frame_dict = {
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "file_path": file_path,
                "transform_matrix": c2w.tolist(),
            }
            all_frames.append(frame_dict)

        for i, frame in enumerate(all_frames):
            transforms["frames"].append(frame)

    with open(os.path.join(OUTPUT_PATH, "transforms_train.json"), "w", encoding="utf8") as outfile:
        json.dump(train_transforms, outfile, indent=4)
    with open(os.path.join(OUTPUT_PATH, "transforms_test.json"), "w", encoding="utf8") as outfile:
        json.dump(test_transforms, outfile, indent=4)
