import argparse
import sys

import cv2
from detectron2.data.transforms import ResizeShortestEdge

from src.appearance import get_model as get_encoder
from src.deep_sort import nn_matching
from src.deep_sort.detection import Detection
from src.deep_sort.tracker import Tracker
from src.detector import get_model as get_detector
from src.utils import visualize_tracks, visualize_detections
from src.appearance.model import Net


def get_options(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, default='data/video.mp4',
                        help="Path to the video to do inference on")
    parser.add_argument("-c", "--confidence-threshold", default=None,
                        help="Choose the confidence threshold for Testing models.")
    parser.add_argument("-t", "--max_eucledian_distance", default=0.5,
                        help="Choose the maximum eucledian distance.")
    parser.add_argument("-mcd", "--nn_budget", default=None,
                        help="Choose the nn_budget.")
    options = parser.parse_args(args)
    return options


IMG_INPUT = None
IMG_OUTPUT = None


def main():
    print("***********************************************")
    args = get_options()
    print(args)

    cap = cv2.VideoCapture(args.video) if args.video is not None else cv2.VideoCapture(2)
    _, img = cap.read()
    resizer = ResizeShortestEdge(600).get_transform(img)

    detector = get_detector()
    encoder = get_encoder()
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", float(args.max_eucledian_distance), args.nn_budget)
    tracker = Tracker(metric)

    while True:
        _, img = cap.read()
        img = resizer.apply_image(img)
        classes, confidences, boxes = detector(img)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = encoder(img, boxes)

        boxes, confidences, classes = map(lambda x: x.detach().cpu(), (boxes, confidences, classes))

        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        detections = [
            Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
            zip(boxes, confidences, classes, features)
        ]

        tracker.predict()
        tracker.update(detections)

        img = visualize_tracks(tracker, img)
        img = visualize_detections(detections, classes, img)

        cv2.imshow('---', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
