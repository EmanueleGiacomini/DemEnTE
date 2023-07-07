from importlib import metadata
import pandas as pd
import numpy as np
import cv2
import os
import json
import argparse
import shutil
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser('PorcodioDemente')
    parser.add_argument('--output', dest='path_output',
                        required=True, type=str)
    parser.add_argument('-d', '--dirs', dest="dirs", nargs='+',
                        help='<Required> Set flag', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    merge_dict = {
        "RGB": [],
        "BB": list(),
        "Mask": [],
        "Labels": []
    }

    for d in args.dirs:
        metadata_file = os.path.join(d, "metadata.json")
        with open(metadata_file) as json_file:
            metadata_dict = json.load(json_file)

        for k in metadata_dict.keys():
            if (k in ["BB", "Labels"]):
                for item in metadata_dict[k]:
                    merge_dict[k].append(item)
                continue

            if (k == "RGB"):
                for i in metadata_dict[k]:
                    merge_dict[k].append(os.path.join(args.path_output, i))
                continue

            for i in metadata_dict[k]:
                elements = []
                for item in i:
                    elements.append(os.path.join(args.path_output, item))
                merge_dict[k].append(elements)

        shutil.copytree(d, os.path.join(args.path_output, d))

    with open(os.path.join(args.path_output, 'metadata.json'), 'w') as f:
        f.write(json.dumps(merge_dict))
