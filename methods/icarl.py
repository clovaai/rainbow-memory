# This code is reffered from following github url.
# https://github.com/donlee90/icarl
import logging
import os
import random

import PIL
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data
from utils.train_utils import select_optimizer

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class ICaRLNet(nn.Module):
    def __init__(self, model, feature_size, n_class):
        super().__init__()
        self.model = model
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_class, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x


class ICaRL(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.feature_size = kwargs["feature_size"]

        self.model.fc = nn.Linear(self.model.fc.in_features, self.feature_size)
        self.feature_extractor = self.model
        self.feature_extractor = self.feature_extractor.to(self.device)

        self.icarlnet = ICaRLNet(
            self.feature_extractor, self.feature_size, kwargs["n_init_cls"]
        )
        self.icarlnet = self.icarlnet.to(self.device)

        # Learning method
        self.dist_loss = nn.BCELoss()

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

        # Number of classes having been trained from prev iterations.
        self.num_learned_class = 0
        # Number of classes being trained
        self.num_learning_class = 0

        if kwargs["mem_manage"] == "default":
            self.mem_manage = "prototype"


    def before_task(self, datalist, cur_iter, init_model=False, init_opt=True):
        datalist_df = pd.DataFrame(datalist)
        incoming_classes = datalist_df["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))

        # learning_class increase monotonically
        self.num_learning_class = max(
            datalist_df["label"].max() + 1, self.num_learning_class
        )

        in_features = self.icarlnet.fc.in_features
        out_features = self.icarlnet.fc.out_features
        weight = self.icarlnet.fc.weight.data

        if init_model:
            # init model parameters in every iteration
            self.model.fc = nn.Linear(
                self.feature_extractor.fc.in_features, self.feature_size
            )
            self.feature_extractor = self.model

            self.icarlnet = ICaRLNet(
                self.feature_extractor, self.feature_size, self.num_learning_class
            )
        else:
            self.icarlnet.fc = nn.Linear(
                in_features, self.num_learning_class, bias=False
            )

        # keep weights for the old classes
        self.icarlnet.fc.weight.data[:out_features] = weight

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.icarlnet = self.icarlnet.to(self.device)

        if init_opt:
            # reinitialize the optimizer and scheduler
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        logger.info(
            f"Increasing heads of fc layer {out_features} --> {self.num_learning_class})"
        )

        self.already_mem_update = False

    def classify(self, x):
        """Classify images by nearest-means-of-exemplars
        Args:
            x: input image batch
        Returns:
            pred: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        # (n_classes, feature_size)
        means = torch.stack(self.exemplar_means)
        # (batch_size, n_classes, feature_size)
        means = torch.stack([means] * batch_size)
        # (batch_size, feature_size, n_classes)
        means = means.transpose(1, 2)

        self.feature_extractor.eval()
        # (batch_size, feature_size)
        feature = self.feature_extractor(x)
        for i in range(feature.size(0)):  # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        # (batch_size, feature_size, 1)
        feature = feature.unsqueeze(2)
        # (batch_size, feature_size, n_classes)
        feature = feature.expand_as(means)

        # (batch_size, n_classes)
        dists = (feature - means).pow(2).sum(1).squeeze()
        # _, pred = dists.min(1)
        _, pred = dists.topk(k=self.topk, dim=1, largest=False, sorted=True)
        return pred

    def train(self, cur_iter, n_epoch, batch_size, n_worker):
        # Loader
        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        q = dict()
        if self.num_learned_class > 0:
            old_class_list = [
                sample
                for sample in train_list
                if sample["label"] < self.num_learned_class
            ]
            _, old_class_loader = self.get_dataloader(
                batch_size, n_worker, None, old_class_list
            )
            # Store network outputs with pre-update parameters
            with torch.no_grad():
                self.icarlnet.eval()
                for data in old_class_loader:
                    images = data["image"].to(self.device)
                    image_names = data["image_name"]
                    g = torch.sigmoid(self.icarlnet(images))
                    for i, image_name in enumerate(image_names):
                        # q[image_name] = g[i].detach().cpu().tolist()
                        q[image_name] = g[i].detach()

        # TRAIN
        best_acc = 0.0
        self.icarlnet.train()
        for epoch in range(n_epoch):
            total_loss, total_cls_loss, total_dist_loss = 0.0, 0.0, 0.0

            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            for i, data in enumerate(train_loader):
                x = data["image"].to(self.device)
                y = data["label"].to(self.device)
                image_names = data["image_name"]

                old_cls_index = torch.nonzero(
                    y < self.num_learned_class, as_tuple=False
                ).squeeze()
                old_cls_index = (
                    old_cls_index.unsqueeze(0)
                    if old_cls_index.dim() == 0
                    else old_cls_index
                )
                new_cls_index = torch.nonzero(
                    y >= self.num_learned_class, as_tuple=False
                ).squeeze()
                new_cls_index = (
                    new_cls_index.unsqueeze(0)
                    if new_cls_index.dim() == 0
                    else new_cls_index
                )
                assert old_cls_index.size(0) + new_cls_index.size(0) == y.size(0)

                self.optimizer.zero_grad()

                g = self.icarlnet(x)

                # Classification loss for new classes
                # loss = torch.zeros(1).to(self.device)
                cls_loss = 0
                if new_cls_index.size(0) > 0:
                    do_cutmix = self.cutmix and np.random.rand(1) < 0.5
                    if do_cutmix:
                        x = x[new_cls_index]
                        y = y[new_cls_index]
                        x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                        g_ = self.icarlnet(x)
                        cls_loss += lam * self.criterion(g_, labels_a) + (
                            1 - lam
                        ) * self.criterion(g_, labels_b)
                    else:
                        cls_loss += self.criterion(g[new_cls_index], y[new_cls_index])
                    total_cls_loss += cls_loss.item()

                # Distillation loss for old classes
                dist_loss = 0
                if self.num_learned_class > 0 and old_cls_index.size(0) > 0:
                    g = torch.sigmoid(g)
                    q_i = []
                    for idx in old_cls_index:
                        name = image_names[idx]
                        q_i.append(q[name])
                    q_i = torch.stack(q_i, dim=0)

                    for y in range(self.num_learned_class):
                        dist_loss += self.dist_loss(g[old_cls_index, y], q_i[:, y])

                    total_dist_loss += dist_loss.item()

                loss = cls_loss + dist_loss
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            num_batches = len(train_loader)

            writer.add_scalar(
                f"task{cur_iter}/train/loss", total_loss / num_batches, epoch
            )
            writer.add_scalar(
                f"task{cur_iter}/train/cls_loss", total_cls_loss / num_batches, epoch
            )
            writer.add_scalar(
                f"task{cur_iter}/train/distilling", total_dist_loss / num_batches, epoch
            )
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            train_loss = total_loss / num_batches
            train_cls_loss = total_cls_loss / num_batches
            train_dist_loss = total_dist_loss / num_batches

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | "
                f"train_cls_loss {train_cls_loss:.4f} | train_distill_loss: {train_dist_loss:.4f} | "
                f"train_lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

        # Icarl requires feature update(=compute exemplars) before the evaluation
        self.compute_means = True
        self.update_memory(cur_iter, self.num_learning_class)
        eval_dict = self.icarl_evaluation(test_loader)
        best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        mask = pred == y.unsqueeze(1)
        mask = mask.sum(dim=1).bool()
        correct_xlabel = y.masked_select(mask)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def icarl_evaluation(self, eval_loader):
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)

        self.feature_extractor.eval()
        if self.compute_means:
            logger.info("Computing mean of classes for classification")
            with torch.no_grad():
                mem_df = pd.DataFrame(self.memory_list)
                exemplar_means = []
                for i in range(self.num_learning_class):
                    cls_df = mem_df[mem_df["label"] == i]
                    cls_data = cls_df.to_dict(orient="records")

                    if len(cls_data) == 0:
                        logger.warning(f"No samples for a class {i}")
                        exemplar_means.append(
                            torch.zeros(self.feature_size).to(self.device)
                        )
                        continue

                    features = []
                    for data in cls_data:
                        image = PIL.Image.open(
                            os.path.join("dataset", self.dataset, data["file_name"])
                        ).convert("RGB")
                        image = self.test_transform(image).to(self.device)
                        feature = self.feature_extractor(image.unsqueeze(0))
                        feature = feature.squeeze()
                        feature.data = feature.data / feature.data.norm()  # Normalize
                        features.append(feature)
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                    mu_y = mu_y / mu_y.norm()  # Normalize
                    exemplar_means.append(mu_y)
                self.exemplar_means = exemplar_means
                self.compute_means = False

        total_num_data, total_correct = 0.0, 0.0
        with torch.no_grad():
            for data in eval_loader:
                x = data["image"].to(self.device)
                y = data["label"]
                pred = self.classify(x)
                total_num_data += y.size(0)
                total_correct += torch.sum(pred.detach().cpu() == y.unsqueeze(1)).item()

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(
                    y.detach().cpu(), pred.detach().cpu()
                )
                correct_l += correct_xlabel_cnt
                num_data_l += xlabel_cnt
        avg_acc = total_correct / total_num_data
        logger.info("[icarl_eval] test acc: {acc:.4f}".format(acc=avg_acc))
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret
