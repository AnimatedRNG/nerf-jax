#!/usr/bin/env python3

import os
import struct
import io
import argparse
from collections import defaultdict

import numpy as np
import cv2
import tensorboard.compat.proto.event_pb2 as event_pb2
import imageio
from tqdm import tqdm


def extract_images(event, args, imgs):
    if event.HasField("summary"):
        for value in event.summary.value:
            if value.tag in args.tags:
                assert value.HasField("image")
                img = value.image
                img = imageio.imread(io.BytesIO(img.encoded_image_string))
                imgs[value.tag][event.step] = img


def encode(imgs, output_folder, fps, upscale):
    for img_tag, img_dict in imgs.items():
        img_tag = img_tag.replace("/", "_")
        out_path = os.path.join(output_folder, f"{img_tag}.mp4")

        writer = imageio.get_writer(out_path, fps=fps)

        steps = sorted(list(img_dict.keys()))
        for step in tqdm(steps):
            img = cv2.resize(
                img_dict[step],
                None,
                fx=upscale,
                fy=upscale,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            writer.append_data(img)
        writer.close()


# loosely-adapted from https://github.com/lanpa/tensorboard-dumper/blob/554c23270fdd68bc91677946d40e5b29ce459dda/dump.py
# this works surprisingly well even without JIT-compilation!
def get_event_strs(data):
    offset = 0

    events_str = []

    while offset < len(data):
        header = 0
        for ci in range(8):
            c = data[offset + 7 - ci]
            header <<= 8
            header |= c

        event_str = data[offset + 12 : offset + 12 + header]  # 8+4
        offset += 12 + header + 4

        events_str.append(event_str)
    return events_str


def main(args):
    try:
        with open(args.input, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"unable to read {args.input}")
        exit()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        for f in os.listdir(args.output):
            path = os.path.join(args.output, f)
            os.remove(path)

    imgs = defaultdict(dict)

    event_strs = get_event_strs(data)

    for event_str in event_strs:
        event = event_pb2.Event()

        event.ParseFromString(event_str)
        images = extract_images(event, args, imgs)
    encode(imgs, args.output, args.fps, args.upscale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate_videos.py")

    parser.add_argument("--input", help="saved tensorboard file to read from")
    parser.add_argument("--fps", type=int, default=10, help="fps of generated videos")
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="nearest neighbor upscale factor for images",
    )
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument(
        "--tags",
        metavar="TAGS",
        required=True,
        type=str,
        nargs="+",
        help="tags to extract",
    )

    args = parser.parse_args()

    main(args)
