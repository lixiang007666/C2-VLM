from sam_lora_image_encoder import LoRA_Sam
from segment_anything import *
import numpy as np
import torch
import torch.optim as optim
from dataset import vessel_2d_dataset
from torch.nn import DataParallel
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from options import *
import itertools
from statistics import *
from loss_functions import *
import os
import random
import time
from display import *
from metrics import MetricsStatistics
from collections import *
from models.build_sam_unet import sam_unet_registry
from segment_anything.utils.transforms import ResizeLongestSide
from thop import profile
from torchsummary import summary


from torch.utils.data.sampler import Sampler

parser = argparse.ArgumentParser(description="training arguments")
add_training_parser(parser)
add_vessel_2d_parser(parser)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
print(args.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()

# for i in range(num_gpus):
#     gpu_name = torch.cuda.get_device_name(i)
#     print(f"GPU {i}: {gpu_name}")

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
print(time_str)

to_cuda = lambda x: x.to(torch.float).to(device)
import logging
from collections import defaultdict


def compute_and_print_model_stats(
    model, images, original_size, prompt_points, prompt_type
):
    model.eval()
    with torch.no_grad():
        macs, params = profile(
            model, inputs=(images, original_size, prompt_points, prompt_type)
        )
        params_m = params / 1e6
        gflops = macs / 1e9
        print(f"Params: {params_m:.2f}M")
        print(f"GFLOPs: {gflops:.2f}G")
        logging.info(f"Model Params: {params_m:.2f}M")
        logging.info(f"Model GFLOPs: {gflops:.2f}G")
    model.train()


class TrainManager_vessel:
    def __init__(self, dataset_train, dataset_val):
        self.dataset_train, self.dataset_val = dataset_train, dataset_val
        parameters = [
            args.fov,
            args.label_type,
            args.epochs,
            args.is_local,
            args.model_type,
            args.remark,
        ]
        # print(parameters)
        self.record_dir = "results/{}/{}".format(
            time_str, "_".join(map(str, parameters))
        )
        self.cpt_dir = "{}/checkpoints".format(self.record_dir)

        if not os.path.exists(self.cpt_dir):
            os.makedirs(self.cpt_dir)

        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(self.record_dir, "training.log"),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        logging.info(
            "Initialized TrainManager_vessel with parameters: {}".format(parameters)
        )

        sample_n = len(self.dataset_train)
        self.indices = list(range(sample_n))
        logging.info(f"Number of training samples: {sample_n}")
        # random.shuffle(self.indices)
        self.split = sample_n // args.k_fold

        if args.model_type == "vit_h":
            sam = sam_model_registry["vit_h"](
                checkpoint="sam_weights/sam_vit_h_4b8939.pth"
            )
        elif args.model_type == "vit_l":
            sam = sam_model_registry["vit_l"](
                checkpoint="sam_weights/sam_vit_l_0b3195.pth"
            )
        else:
            sam = sam_model_registry["vit_b"](
                checkpoint="sam_weights/sam_vit_b_01ec64.pth"
            )

        self.sam_transform = (
            ResizeLongestSide(224)
            if args.model_type == "vit_b"
            else ResizeLongestSide(1024)
        )

        model = sam_unet_registry["res50_sam_unet"](
            need_ori_checkpoint=True,
            sam_unet_checkpoint="sam_weights/sam_vit_h_4b8939.pth",
        )
        lora_sam = LoRA_Sam(model, 4)
        self.model = DataParallel(lora_sam).to(device)
        # self.model = lora_sam.to(device)
        torch.save(self.model.state_dict(), "{}/init.pth".format(self.cpt_dir))
        logging.info("Saved initial model checkpoint.")

        self.loss_func = DiceLoss()
        if args.label_type in ["Artery", "Vein", "LargeVessel"]:
            self.loss_func = lambda x, y: 0.8 * DiceLoss()(x, y) + 0.2 * clDiceLoss()(
                x, y
            )

        # Initialize best validation loss tracker
        self.best_val_loss = defaultdict(lambda: float("inf"))

    def get_dataloader(self, fold_i):
        # train_indices = (
        #     self.indices[: fold_i * self.split]
        #     + self.indices[(fold_i + 1) * self.split :]
        # )
        # val_indices = self.indices[fold_i * self.split : (fold_i + 1) * self.split]

        # Define the validation indices for the current fold
        val_indices = self.indices[fold_i * self.split : (fold_i + 1) * self.split]
        
        # Modify the training indices by including the validation indices as well
        train_indices = self.indices[: fold_i * self.split] + self.indices[(fold_i + 1) * self.split :] + val_indices


        # print(train_indices, val_indices)
        train_sampler, val_sampler = [
            SubsetRandomSampler(x) for x in (train_indices, val_indices)
        ]
        import multiprocessing

        # num_workers = min(32, multiprocessing.cpu_count())
        num_workers = 64
        train_loader = DataLoader(
            self.dataset_train,
            batch_size=32,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=32,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def reset(self):
        self.model.load_state_dict(torch.load("{}/init.pth".format(self.cpt_dir)))
        # print(self.model)
        # for param in self.model.module.unet.parameters():
        #     param.requires_grad = True
        # for p in self.model.module.unet.parameters():
        #     print(p.requires_grad)
        print("List of trainable parameters: ")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")
        pg = [p for p in self.model.parameters() if p.requires_grad]
        # self.optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
        # self.optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # lr_lambda = lambda x: max(
        #     1e-6,
        #     args.lr * 0.9 ** x,
        # )
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = args.epochs // 5
        lr_lambda = lambda x: max(1e-5, args.lr * x / epoch_p if x <= epoch_p else args.lr * 0.85 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logging.info("Optimizer and scheduler have been reset.")

    def record_performance(
        self, train_loader, val_loader, fold_i, epoch, metrics_statistics
    ):
        # Define paths for latest and best checkpoints
        latest_checkpoint_path = f"{self.cpt_dir}/fold-{fold_i}_latest.pth"
        best_checkpoint_path = f"{self.cpt_dir}/fold-{fold_i}_best.pth"
        save_dir = "{}/{}/{:0>4}".format(self.record_dir, fold_i, epoch)

        # Save the latest checkpoint
        torch.save(
            self.model.state_dict(),
            latest_checkpoint_path,
        )
        logging.info(f"Saved latest checkpoint for fold {fold_i} at epoch {epoch}.")

        # Record the current learning rate
        metrics_statistics.metric_values["learning rate"].append(
            self.optimizer.param_groups[0]["lr"]
        )

        # Initialize a variable to track validation loss
        current_val_loss = None

        def record_dataloader(dataloader, loader_type="val", is_complete=True):
            nonlocal current_val_loss
            for (
                images,
                selected_components,
                sample_ids,
            ) in dataloader:
                # print(loader_type)
                images, labels = map(to_cuda, (images, selected_components))

                if loader_type == "val":
                    start_time = time.time()
                    preds = self.model(images, None, None, None)
                    inference_time = time.time() - start_time
                    print(
                        f"[{loader_type}] Inference time for sample {sample_ids[0]}: {inference_time:.4f}s"
                    )
                else:
                    preds = self.model(images, None, None, None)
                loss = self.loss_func(preds, labels).cpu().item()
                # print(f"Epoch val loss: {loss.item():.4f}")
                metrics_statistics.metric_values["loss_" + loader_type].append(loss)

                if loader_type == "val":
                    # Update current_val_loss with the last batch's loss
                    current_val_loss = loss

                if is_complete:
                    preds = torch.gt(preds, 0.8).int()
                    sample_id = str(sample_ids[0])

                    image, label, pred = map(
                        lambda x: x[0][0].cpu().detach(), (images, labels, preds)
                    )
                    # if loader_type == "val":
                    #     prompt_info = None
                    # else:
                    #     prompt_points, prompt_type = (
                    #         prompt_points[0].cpu().detach(),
                    #         prompt_type[0].cpu().detach(),
                    #     )
                    #     prompt_info = np.concatenate(
                    #         (prompt_points, prompt_type[:, np.newaxis]), axis=1
                    #     ).astype(int)
                    metrics_statistics.cal_epoch_metric(
                        args.metrics,
                        "{}-{}".format(args.label_type, loader_type),
                        label.int(),
                        pred.int(),
                    )

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_sample_func = lambda x, y: np.save(
                        "/".join(
                            [
                                save_dir,
                                "{}_{}_{}.npy".format(args.label_type, x, sample_id),
                            ]
                        ),
                        y,
                    )
                    save_items = {
                        "sample": image / 255,
                        "label": label,
                        "pred": pred,
                    }
                    # if prompt_info is not None:
                    #     save_items["prompt_info"] = prompt_info
                    for x, y in save_items.items():
                        save_sample_func(x, y)

        # Record performance on training and validation sets
        # record_dataloader(train_loader, "train", False)
        record_dataloader(val_loader, "val", True)

        # Record metrics to the statistics object
        metrics_statistics.record_result(epoch)

        # Check if the current validation loss is the best so far for this fold
        if (
            current_val_loss is not None
            and current_val_loss < self.best_val_loss[fold_i]
        ):
            self.best_val_loss[fold_i] = current_val_loss
            # Save the best checkpoint
            torch.save(
                self.model.state_dict(),
                best_checkpoint_path,
            )
            logging.info(
                f"New best model for fold {fold_i} at epoch {epoch} with val_loss {current_val_loss:.4f}"
            )

    def train(self):
        for fold_i in range(args.k_fold):
            train_loader, val_loader = self.get_dataloader(fold_i)
            self.reset()
            metrics_statistics = MetricsStatistics(
                save_dir="{}/{}".format(self.record_dir, fold_i)
            )
            # self.record_performance(
            #     train_loader, val_loader, fold_i, 0, metrics_statistics
            # )
            for epoch in range(1, args.epochs + 1):
                print("Epoch:", epoch)
                for (
                    images,
                    selected_components,
                    sample_ids,
                ) in tqdm(train_loader, desc="Training"):
                    # print("Train...")
                    images, labels = map(to_cuda, (images, selected_components))
                    self.optimizer.zero_grad()

                    # compute_and_print_model_stats(self.model, images, original_size, prompt_points, prompt_type)

                    Trainable_params = sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    )
                    # print(f'Trainable params: {Trainable_params/ 1e6}M')

                    preds = self.model(images, None, None, None)
                    loss = self.loss_func(preds, labels)
                    loss.backward()
                    # self.print_unet_gradients()
                    self.optimizer.step()
                    # self.scheduler.step()

                    for param_group in self.optimizer.param_groups:
                        print(f"Batch Learning Rate: {param_group['lr']:.6f}")

                    print(f"Batch Loss: {loss.item():.4f}")

                self.scheduler.step()
                if epoch % args.check_interval == 0:
                    self.record_performance(
                        train_loader, val_loader, fold_i, epoch, metrics_statistics
                    )
            metrics_statistics.close()
            logging.info(f"Completed training for fold {fold_i}.")

    def print_unet_gradients(self):
        for name, param in self.model.module.unet.named_parameters():
            if param.grad is not None:
                print(f"gradients - {name}: {param.grad.norm()}")
            else:
                print(f"{name} no gradients")

    def make_prompts(self, images, prompt_points):
        original_size = tuple(images.shape[-2:])
        images = self.sam_transform.apply_image_torch(images)
        prompt_points = self.sam_transform.apply_coords_torch(
            prompt_points, original_size
        )

        return images, original_size, prompt_points

    def make_prompts_validation(self, images):
        original_size = tuple(images.shape[-2:])
        images = self.sam_transform.apply_image_torch(images)
        return images, original_size


if __name__ == "__main__":
    ppn, pnn = args.prompt_positive_num, args.prompt_negative_num
    dataset_params = [args.fov, args.label_type, ppn, pnn, args.is_local, True]
    dataset_train = vessel_2d_dataset(*dataset_params)
    dataset_params[-1] = False
    dataset_val = vessel_2d_dataset(*dataset_params)
    train_manager = TrainManager_vessel(dataset_train, dataset_val)
    train_manager.train()
