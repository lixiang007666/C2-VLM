import numpy as np
from scipy import ndimage

def _compute_bounding_box(mask):
    num_dims = len(mask.shape)
    bbox_min = np.zeros(num_dims, np.int64)
    bbox_max = np.zeros(num_dims, np.int64)

    proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return None, None

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    for axis in range(1, num_dims):
        max_over_axes = list(range(num_dims))
        max_over_axes.pop(axis)
        proj = np.amax(mask, axis=tuple(max_over_axes))
        idx_nonzero = np.nonzero(proj)[0]
        bbox_min[axis] = np.min(idx_nonzero)
        bbox_max[axis] = np.max(idx_nonzero)

    return bbox_min, bbox_max


def _crop_to_bounding_box(mask, bbox_min, bbox_max):
    cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    num_dims = len(mask.shape)

    if num_dims == 2:
        cropmask[0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                    bbox_min[1]:bbox_max[1] + 1]
    elif num_dims == 3:
        cropmask[0:-1, 0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                          bbox_min[1]:bbox_max[1] + 1,
                                          bbox_min[2]:bbox_max[2] + 1]
    return cropmask


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    if mask_gt.shape != mask_pred.shape:
        raise ValueError("Shape mismatch between ground truth and prediction masks.")

    num_dims = len(mask_gt.shape)
    if num_dims == 2:
        kernel = np.array([[1, 2],
                           [4, 8]], dtype=np.uint8)
        full_true_neighbours = 0b1111
    elif num_dims == 3:
        kernel = np.array([
            [[1, 2], [4, 8]],
            [[16, 32], [64, 128]]
        ], dtype=np.uint8)
        full_true_neighbours = 0b11111111
    else:
        raise ValueError("Only 2D and 3D masks are supported.")

    bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    if bbox_min is None:
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
        }

    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    neighbour_code_map_gt = ndimage.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0
    )
    neighbour_code_map_pred = ndimage.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0
    )

    borders_gt = ((neighbour_code_map_gt != 0) &
                  (neighbour_code_map_gt != full_true_neighbours))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != full_true_neighbours))

    if borders_gt.any():
        distmap_gt = ndimage.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.inf * np.ones_like(borders_gt)

    if borders_pred.any():
        distmap_pred = ndimage.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.inf * np.ones_like(borders_pred)

    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
    }


def compute_average_surface_distance(surface_distances):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]

    if len(distances_gt_to_pred) > 0:
        asd_gt_to_pred = np.mean(distances_gt_to_pred)
    else:
        asd_gt_to_pred = np.inf

    if len(distances_pred_to_gt) > 0:
        asd_pred_to_gt = np.mean(distances_pred_to_gt)
    else:
        asd_pred_to_gt = np.inf

    return asd_gt_to_pred, asd_pred_to_gt
