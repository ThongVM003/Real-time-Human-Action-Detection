from utils.convert import to_ucf24
from rich import print
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert data to UCF24 format")
    parser.add_argument(
        "--source",
        type=str,
        help="Path to the image or video",
    )
    parser.add_argument(
        "--dest",
        type=str,
        help="Path to the image or video",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="ucf24",
        help="dataset type",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    to_ucf24(args.source, args.dest)
