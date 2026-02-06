from scipy.spatial.distance import directed_hausdorff
from collections import *
from statistics import mean

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from scipy.spatial import distance
from metrics_utils import compute_surface_distances, compute_average_surface_distance



class MetricsStatistics:
    def __init__(self, save_dir="./results/"):
        self.epsilon = 1e-6
        self.func_dct = {
            "Precision": self.cal_precision,
            "Recall": self.cal_recall,
            "Specificity": self.cal_specificity,
            "ASD": self.compute_asd,
            "Dice": self.cal_dice,
            "Hausdorff": self.cal_hausdorff,
        }
        self.save_dir = save_dir
        self.metric_values = defaultdict(list)
        self.metric_epochs = defaultdict(list)
        self.summary_writer = SummaryWriter(log_dir=save_dir)

    def cal_epoch_metric(self, metrics, label_type, label, pred):
        for x in metrics:
            self.metric_values["{}-{}".format(x, label_type)].append(
                self.func_dct[x](label, pred)
            )

    def record_result(self, epoch):
        self.metric_epochs["epoch"].append(epoch)

        # Update the set of all metric keys
        all_metric_keys = set(self.metric_epochs.keys()) - {"epoch"}
        current_metric_keys = set(self.metric_values.keys())
        all_metric_keys.update(current_metric_keys)

        # Initialize missing metric lists
        for k in all_metric_keys:
            if k not in self.metric_epochs:
                print(f"Initializing metric {k}")
                self.metric_epochs[k] = [float("nan")] * (
                    len(self.metric_epochs["epoch"]) - 1
                )

        # Append metric values or NaN
        for k in all_metric_keys:
            if k in self.metric_values:
                mean_value = mean(self.metric_values[k])
                self.summary_writer.add_scalar(k, mean_value, epoch)
                self.metric_epochs[k].append(mean_value)
            else:
                print("Metric {} not found".format(k))
                self.metric_epochs[k].append(float("nan"))

        pd.DataFrame(self.metric_epochs).to_excel(
            f"{self.save_dir}/metrics_statistics.xlsx", index=False
        )
        self.metric_values.clear()

    def close(self):
        self.summary_writer.close()

    def cal_confusion_matrix(self, pred, label):
        TP = ((pred == 1) & (label == 1)).sum().item()
        FP = ((pred == 0) & (label == 1)).sum().item()
        FN = ((pred == 1) & (label == 0)).sum().item()
        TN = ((pred == 0) & (label == 0)).sum().item()
        return TP, FP, FN, TN

    def cal_precision(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TP / (TP + FP + self.epsilon)

    def cal_recall(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TP / (TP + FN + self.epsilon)

    def cal_specificity(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TN / (TN + FP + self.epsilon)


    def compute_asd(self, pred, label):
        pred_np = pred.cpu().numpy().astype(bool)
        label_np = label.cpu().numpy().astype(bool)
        
        # 默认 spacing 是各向同性，可以根据实际情况传入 [x_spacing, y_spacing, z_spacing]
        spacing_mm = [1.0] * pred_np.ndim

        surface_distances = compute_surface_distances(label_np, pred_np, spacing_mm)
        asd_gt_to_pred, asd_pred_to_gt = compute_average_surface_distance(surface_distances)
        
        # 返回对称 ASD（可以改成单向）
        return (asd_gt_to_pred + asd_pred_to_gt) / 2


    def calculate_surface_points(self, binary_mask):
        # Find coordinates of non-zero (surface) points in the binary mask
        return np.argwhere(binary_mask == 1)
    
    def cal_dice(self, pred, label):
        intersection = (pred & label).sum().item()
        union = pred.sum().item() + label.sum().item()
        dice = 2 * intersection / (union + self.epsilon)
        return dice

    def cal_hausdorff(self, pred, label):
        array1 = pred.cpu().numpy()
        array2 = label.cpu().numpy()
        dist1 = directed_hausdorff(array1, array2)[0]
        dist2 = directed_hausdorff(array2, array1)[0]
        hausdorff_dist = max(dist1, dist2)
        return hausdorff_dist
