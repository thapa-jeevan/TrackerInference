def IOU(bbox_p, bbox_t, p_xywh_format=False, t_xywh_format=False):
    x1_p, y1_p, x2_p, y2_p = bbox_p.tolist()
    x1_t, y1_t, x2_t, y2_t = bbox_t

    if t_xywh_format:
        x2_t, y2_t = (x1_t + x2_t, y1_t + y2_t)
    if p_xywh_format:
        x2_p, y2_p = (x1_p + x2_p, y1_p + y2_p)

    x1, y1 = max(x1_p, x1_t), max(y1_p, y1_t)
    x2, y2 = min(x2_p, x2_t), min(y2_p, y2_t)

    area_bbox_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_bbox_t = (x2_t - x1_t) * (y2_t - y1_t)

    w_inter = (x2 - x1)
    h_inter = (y2 - y1)
    if w_inter <= 0 or h_inter <= 0:
        return 0

    area_intersection = w_inter * h_inter

    _IOU = area_intersection / (area_bbox_p + area_bbox_t - area_intersection)

    return _IOU
