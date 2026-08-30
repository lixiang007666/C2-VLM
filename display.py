import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

alpha = 0.5

overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1 - alpha, 0)

to_blue = (
    lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)])
    .transpose((1, 2, 0))
    .astype(dtype=np.uint8)
)
to_red = (
    lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x])
    .transpose((1, 2, 0))
    .astype(dtype=np.uint8)
)
to_green = (
    lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)])
    .transpose((1, 2, 0))
    .astype(dtype=np.uint8)
)
to_light_green = (
    lambda x: np.array([np.zeros_like(x), x / 2, np.zeros_like(x)])
    .transpose((1, 2, 0))
    .astype(dtype=np.uint8)
)
to_yellow = (
    lambda x: np.array([np.zeros_like(x), x, x])
    .transpose((1, 2, 0))
    .astype(dtype=np.uint8)
)

to_3ch = lambda x: np.array([x, x, x]).transpose((1, 2, 0)).astype(dtype=np.uint8)


def show_result_sample_figure(image, label, pred, prompt_points):
    sz = image.shape[-1] // 250
    cvt_img = lambda x: x.astype(np.uint8)
    image, label, pred = map(cvt_img, (image, label, pred))
    if len(image.shape) == 2:
        image = to_3ch(image)
    else:
        image = image.transpose((1, 2, 0))
    print(image.shape)
    print(label.shape)
    print(pred.shape)
    print(image.shape[:2])
    target_height, target_width = image.shape[:2]
    target_size = (target_width, target_height)
    label, pred = cv2.resize(label, target_size), cv2.resize(pred, target_size)
    print(label.shape)
    print(pred.shape)
    # image = cv2.transpose(image, (2, 0, 1))
    # label = cv2.transpose(label)
    # pred = cv2.transpose(pred)
    label_img = overlay(image, to_light_green(label))
    pred_img = overlay(image, to_yellow(pred))

    def draw_points(img):
        if prompt_points.size > 0:
            for x, y, type in prompt_points:
                cv2.circle(img, (x, y), int(sz), (255, 0, 0), -1)
                if type:
                    cv2.circle(img, (x, y), sz, (0, 255, 0), -1)
                else:
                    cv2.circle(img, (x, y), sz, (0, 0, 255), -1)

    draw_points(label_img)
    draw_points(pred_img)
    return np.concatenate((image, label_img, pred_img), axis=1)


def show_prompt_points_image(
    image,
    positive_region,
    negative_region,
    positive_points,
    negative_points,
    save_file=None,
):
    overlay_img = overlay(to_red(negative_region), to_yellow(positive_region))
    overlay_img = overlay(to_3ch(image), overlay_img)

    if positive_points.size > 0:
        for x, y in positive_points:
            cv2.circle(overlay_img, (x, y), 4, (0, 255, 0), -1)
    if negative_points.size > 0:
        for x, y in negative_points:
            cv2.circle(overlay_img, (x, y), 4, (0, 0, 255), -1)

    if save_file:
        cv2.imwrite(save_file, overlay_img)

    return overlay_img


def view_result_samples(result_dir):

    save_dir = "sample_display/{}".format(result_dir[len("results/") :])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_names = [x.split("_")[-1][:-4] for x in os.listdir(result_dir) if "label" in x]
    print(file_names)
    data_name = [x[:-15] for x in os.listdir(result_dir) if "label" in x][0]
    print(data_name)
    for file_name in tqdm(file_names):
        label = np.load("{}/{}_label_{}.npy".format(result_dir, data_name, file_name))
        pred = np.load("{}/{}_pred_{}.npy".format(result_dir, data_name, file_name))
        prompt_info_path = "{}/{}_prompt_info_{}.npy".format(
            result_dir, data_name, file_name
        )
        if os.path.exists(prompt_info_path):
            prompt_info = np.load(prompt_info_path)
        else:
            prompt_info = np.array([])  # Handle the case when prompt_info is empty
        image = np.load("{}/{}_sample_{}.npy".format(result_dir, data_name, file_name))

        result = show_result_sample_figure(
            image * 255, label * 255, pred * 255, prompt_info
        )
        cv2.imwrite("{}/{}.png".format(save_dir, file_name), result)


if __name__ == "__main__":
    result_dir = (
        "/nvme_data/home/lixiang/ViTeUNet/results/2025-03-27-10-46-18/3M_LargeVessel_200_False_vit_h_OCTA-500/0/0001"
    )
    view_result_samples(result_dir)
