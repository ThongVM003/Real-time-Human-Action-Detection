from utils.detect import detect, init_net
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Human Action Detection")
    parser.add_argument(
        "--source",
        type=str,
        default="resources/test/res.mp4",
        help="Path to the image or video",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="video",
        help="Type of the source (image or video)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the output video",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the output video",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold",
    )
    return parser.parse_args()


def main():
    # Load the model

    args = parse_args()
    net = init_net()
    # net("/workspaces/rtsptoweb/Real-time-Human-Action-Detection/resources/test/res.mp4")
    # Detect objects in a video
    detect(
        args.source,
        net,
        type=args.type,
        show=args.show,
        save=args.save,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
