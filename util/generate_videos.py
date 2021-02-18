#!/usr/bin/env python3

import os
import struct
import io
import argparse
from collections import defaultdict

import tensorboard.compat.proto.event_pb2 as event_pb2
import imageio


# adapted from https://github.com/lanpa/tensorboard-dumper/blob/554c23270fdd68bc91677946d40e5b29ce459dda/dump.py
def read_pb2(data):
    header = struct.unpack("Q", data[:8])

    event_str = data[12 : 12 + int(header[0])]  # 8+4
    data = data[12 + int(header[0]) + 4 :]
    return data, event_str


def extract_images(event, args, imgs):
    if event.HasField("summary"):
        for value in event.summary.value:
            if value.tag in args.tags:
                assert value.HasField("image")
                img = value.image
                img = imageio.imread(io.BytesIO(img.encoded_image_string))
                imgs[value.tag][event.step] = img


def encode(imgs, output_folder, fps):
    for img_tag, img_dict in imgs.items():
        img_tag = img_tag.replace("/", "_")
        out_path = os.path.join(output_folder, f"{img_tag}.mp4")

        writer = imageio.get_writer(out_path, fps=fps)

        steps = sorted(list(img_dict.keys()))
        for step in steps:
            writer.append_data(img_dict[step])
        writer.close()

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

    while data:
        data, event_str = read_pb2(data)
        event = event_pb2.Event()

        event.ParseFromString(event_str)
        images = extract_images(event, args, imgs)
    encode(imgs, args.output, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate_videos.py")

    parser.add_argument("--input", help="saved tensorboard file to read from")
    parser.add_argument("--fps", type=int, default=10, help="fps of generated videos")
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
