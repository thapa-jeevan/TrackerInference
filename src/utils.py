import cv2


def visualize_tracks(tracker, img):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        cv2.putText(img, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                    1.5e-3 * img.shape[0], (0, 255, 0), 1)
    return img


def visualize_detections(detections, classes, img):
    for det in detections:
        bbox = det.to_tlbr()
        # score = "%.2f" % round(det.confidence * 100, 2) + "%"
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        # if len(classes) > 0:
        #     # cls = det.cls
        #     cv2.putText(img, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
        #                 1.5e-3 * img.shape[0], (0, 255, 0), 1)
    return img
