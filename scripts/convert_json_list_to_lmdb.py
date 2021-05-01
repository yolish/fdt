import argparse
import lmdb
import msgpack
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils.image_operations import bbox_is_dict
from utils.pose_operations import get_pose, pose_bbox_to_full_image


class FramesJsonList(ImageFolder):
    def __init__(self, threed_5_points, threed_68_points, json_list, dataset_path=None):
        self.samples = []
        self.bboxes = []
        self.landmarks = []
        self.threed_5_points = threed_5_points
        self.threed_68_points = threed_68_points
        self.dataset_path = dataset_path

        image_paths = pd.read_csv(json_list, delimiter=" ", header=None)
        image_paths = np.asarray(image_paths).squeeze()

        print("Loading frames paths...")
        for image_path in tqdm(image_paths):
            with open(image_path) as f:
                image_json = json.load(f)

            # path to the image
            img_path = image_json["image_path"]
            # if not absolute path, append the dataset path
            if self.dataset_path is not None:
                img_path = os.path.join(self.dataset_path, img_path)
            self.samples.append(img_path)

            # landmarks used to create pose labels
            self.landmarks.append(image_json["landmarks"])

            # load bboxes
            self.bboxes.append(image_json["bboxes"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = Path(self.samples[index])

        img = Image.open(image_path)

        (w, h) = img.size
        global_intrinsics = np.array(
            [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
        )
        bboxes = self.bboxes[index]
        landmarks = self.landmarks[index]

        bbox_labels = []
        landmark_labels = []
        pose_labels = []
        global_pose_labels = []

        for i in range(len(bboxes)):
            bbox = np.asarray(bboxes[i])[:4].astype(int)
            landmark = np.asarray(landmarks[i])[:, :2].astype(float)

            # remove samples that do not have height ot width or are negative
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue

            if -1 in landmark:
                global_pose_labels.append([-9, -9, -9, -9, -9, -9])
                pose_labels.append([-9, -9, -9, -9, -9, -9])

            else:
                landmark[:, 0] -= bbox[0]
                landmark[:, 1] -= bbox[1]

                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])

                bbox_intrinsics = np.array(
                    [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]]
                )

                if len(landmark) == 5:
                    P, pose = get_pose(self.threed_5_points, landmark, bbox_intrinsics)
                else:
                    P, pose = get_pose(
                        self.threed_68_points,
                        landmark,
                        bbox_intrinsics,
                    )

                pose_labels.append(pose.tolist())

                global_pose = pose_bbox_to_full_image(
                    pose, global_intrinsics, bbox_is_dict(bbox)
                )

                global_pose_labels.append(global_pose.tolist())

            bbox_labels.append(bbox.tolist())
            landmark_labels.append(self.landmarks[index][i])

        with open(image_path, "rb") as f:
            raw_img = f.read()

        return (
            raw_img,
            global_pose_labels,
            bbox_labels,
            pose_labels,
            landmark_labels,
        )


class JsonLoader(DataLoader):
    def __init__(
        self, workers, json_list, threed_5_points, threed_68_points, dataset_path=None
    ):
        self._dataset = FramesJsonList(
            threed_5_points, threed_68_points, json_list, dataset_path
        )

        super(JsonLoader, self).__init__(
            self._dataset, num_workers=workers, collate_fn=lambda x: x
        )

def json_list_to_lmdb(args):
    cpu_available = os.cpu_count()
    if args.num_workers > cpu_available:
        args.num_workers = cpu_available

    threed_5_points = np.load(args.threed_5_points)
    threed_68_points = np.load(args.threed_68_points)

    print("Loading dataset from %s" % args.json_list)
    data_loader = JsonLoader(
        args.num_workers,
        args.json_list,
        threed_5_points,
        threed_68_points,
        args.dataset_path,
    )

    name = f"{os.path.split(args.json_list)[1][:-4]}.lmdb"
    lmdb_path = os.path.join(args.dest, name)
    isdir = os.path.isdir(lmdb_path)

    if os.path.isfile(lmdb_path):
        os.remove(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    size = len(data_loader) * 1200 * 1200 * 3
    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(f"Total number of samples: {len(data_loader)}")

    all_pose_labels = []

    txn = db.begin(write=True)

    total_samples = 0

    for idx, data in tqdm(enumerate(data_loader)):
        image, event_class, global_pose_labels, bboxes, pose_labels, landmarks = data[0]

        if len(bboxes) == 0:
            continue

        has_pose = False
        for pose_label in pose_labels:
            if pose_label[0] != -9:
                all_pose_labels.append(pose_label)
                has_pose = True

        if not has_pose:
            continue

        txn.put(
            "{}".format(total_samples).encode("ascii"),
            msgpack.dumps((image, global_pose_labels, bboxes, pose_labels, landmarks, event_class)),
        )
        if idx % args.write_frequency == 0:
            print(f"[{idx}/{len(data_loader)}]")
            txn.commit()
            txn = db.begin(write=True)

        total_samples += 1

    print(total_samples)

    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(total_samples)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    if args.train:
        print("Saving pose mean and std dev.")
        all_pose_labels = np.asarray(all_pose_labels)
        pose_mean = np.mean(all_pose_labels, axis=0)
        pose_stddev = np.std(all_pose_labels, axis=0)

        save_file_path = os.path.join(args.dest, os.path.split(args.json_list)[1][:-4])
        np.save(f"{save_file_path}_pose_mean.npy", pose_mean)
        np.save(f"{save_file_path}_pose_stddev.npy", pose_stddev)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_list",
        type=str,
        required=True,
        help="List of json files that contain frames annotations",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset images",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--write_frequency", help="Frequency to save to file.", type=int, default=5000
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Path to save the lmdb file."
    )
    parser.add_argument(
        "--train", action="store_true", help="Dataset will be used for training."
    )
    parser.add_argument(
        "--threed_5_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_5_points_trans.npy",
    )

    parser.add_argument(
        "--threed_68_points",
        type=str,
        help="Reference 3D points to compute pose.",
        default="./pose_references/reference_3d_68_points_trans.npy",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    return args


if __name__ == "__main__":
    args = parse_args()

    json_list_to_lmdb(args)
