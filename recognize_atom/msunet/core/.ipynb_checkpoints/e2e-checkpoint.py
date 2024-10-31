import os
import glob
import json
import scipy
import numpy as np
import pandas as pd

def center_coords_to_bbox(gt_coord):
    box_rwidth, box_rheight = 10, 10
    gt_bbox = (
        gt_coord[0] - box_rwidth,
        gt_coord[0] + box_rwidth + 1,
        gt_coord[1] - box_rheight,
        gt_coord[1] + box_rheight + 1
    )
    return gt_bbox


def get_coord_to_bboxes(gt_coordinates_dict):
    gt_bboxes_list = []
    for gt_coord in gt_coordinates_dict:
        gt_bbox = center_coords_to_bbox(gt_coord)
        gt_bboxes_list.append(gt_bbox)
    return gt_bboxes_list


def bbox_iou(bb1, bb2):
    assert bb1[0] <= bb1[1]
    assert bb1[2] <= bb1[3]
    assert bb2[0] <= bb2[1]
    assert bb2[2] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def match_bboxes(iou_matrix, IOU_THRESH=0.5):
    n_true, n_pred = iou_matrix.shape
    MIN_IOU = 0.0
    MAX_DIST = 1.0

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((diff, n_pred), MIN_IOU)),
                                    axis=0)

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((n_true, diff), MIN_IOU)),
                                    axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label


def eval_matches(gt_bboxes, pd_bboxes, iou_threshold):
        iou_matrix = np.zeros((len(gt_bboxes), len(pd_bboxes))).astype(np.float32)

        for gt_idx, gt_bbox in enumerate(gt_bboxes):
            for pd_idx, pd_bbox in enumerate(pd_bboxes):
                iou = bbox_iou(gt_bbox, pd_bbox)
                iou_matrix[gt_idx, pd_idx] = iou
                
        idxs_true, idxs_pred, ious, labels = match_bboxes(iou_matrix, IOU_THRESH=iou_threshold)
        return idxs_true, idxs_pred, ious, labels
    
def eval_metrics(n_matches, n_gt, n_pred):
        precision = n_matches / n_pred if n_pred > 0 else 0.0
        if n_gt == 0:
            raise RuntimeError("No ground truth atoms???")
        recall = n_matches / n_gt
        
        return precision, recall
    
    
def get_metrics(gt, pred, iou_threshold):
    h, w = np.where(gt != 0)
    gt_coords = list(zip(h.flatten(), w.flatten()))
    gt_bboxes = get_coord_to_bboxes(gt_coords)

    h, w = np.where(pred != 0)
    pd_coords = list(zip(h.flatten(), w.flatten()))
    pd_bboxes = get_coord_to_bboxes(pd_coords)

    idxs_true, idxs_pred, ious, labels = eval_matches(gt_bboxes, pd_bboxes, iou_threshold)
    precision, recall = eval_metrics(n_matches=len(idxs_pred), n_gt=len(gt_coords), n_pred=len(pd_bboxes))
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1_score
