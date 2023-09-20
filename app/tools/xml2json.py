import argparse
import json
import os
from xml.dom import minidom

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="convert cc.xml to transforms.json")

    parser.add_argument("--path", type=str, default="landmark/datasets/your_dataset")
    parser.add_argument("--input_xml", type=str, default="cc.xml")

    init_args = parser.parse_args()
    return init_args


if __name__ == "__main__":
    args = parse_args()

results = {}

doc = minidom.parse(os.path.join(args.path, args.input_xml))
photo_subgroup = doc.getElementsByTagName("Photo")

for idx, photo in enumerate(tqdm(photo_subgroup)):

    path = photo.getElementsByTagName("ImagePath")[0].firstChild.data

    H = int(photo.getElementsByTagName("Height")[0].firstChild.data)
    W = int(photo.getElementsByTagName("Width")[0].firstChild.data)

    # if FocalLenth is presented in pixels
    focal_px = float(photo.getElementsByTagName("FocalLength")[0].firstChild.data)

    # if FocalLenthPixel is given
    # focal_px = float(photo.getElementsByTagName("FocalLengthPixel")[0].firstChild.data)

    # if SensorSize is given
    # focal = float(photo.getElementsByTagName("FocalLength")[0].firstChild.data)
    # sensor = float(photo.getElementsByTagName("SensorSize")[0].firstChild.data)
    # focal_px = focal / sensor * W

    photo_subgroup = photo.getElementsByTagName("Photo")

    rot = photo.getElementsByTagName("Rotation")[0]
    cet = photo.getElementsByTagName("Center")[0]

    rot_mat = np.zeros((4, 4))
    rot_mat[-1, -1] = 1

    for i in range(3):
        for j in range(3):
            rot_mat[i][j] = float(rot.getElementsByTagName(f"M_{i}{j}")[0].firstChild.data)

    rot_mat[:3, :3] = np.linalg.inv(rot_mat[:3, :3])

    rot_mat[:3, 1] = -rot_mat[:3, 1]
    rot_mat[:3, 2] = -rot_mat[:3, 2]

    x = float(cet.getElementsByTagName("x")[0].firstChild.data)
    y = float(cet.getElementsByTagName("y")[0].firstChild.data)
    z = float(cet.getElementsByTagName("z")[0].firstChild.data)

    rot_mat[0, -1] = x
    rot_mat[1, -1] = y
    rot_mat[2, -1] = z

    c2w = np.hstack([rot_mat[:3, :4], np.array([[H, W, focal_px]]).T])
    results[idx] = {
        "path": path,
        "rot_mat": c2w.tolist(),
    }

with open(os.path.join(args.path, "transforms.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
