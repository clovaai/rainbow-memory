################################
# This code is reffered by
# https://github.com/GT-RIPL/Continual-Learning-Benchmark
################################

import logging
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class L2(Finetune):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        # except for last layers.
        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.reg_coef = kwargs["reg_coef"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "reservoir"
        self.online_reg = "online" in kwargs["stream_env"]

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

    def update_model(self, inputs, targets, optimizer):
        out = self.model(inputs)
        loss = self.regularization_loss(out, targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.detach(), out

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

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
            # learning rate scheduling from
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                total_epochs=n_epoch,
            )
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )
            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            if best_acc < eval_dict["avg_acc"]:
                best_acc = eval_dict["avg_acc"]

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }
        logger.debug(f"# of reg_terms: {len(self.regularization_terms)}")

        return best_acc, eval_dict

    def _train(self, train_loader, optimizer, epoch, total_epochs):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (
                    1 - lam
                ) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            reg_loss = self.regularization_loss()

            loss += reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            _, preds = logit.topk(self.topk, 1, True, True)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        n_batches = len(train_loader)
        return total_loss / n_batches, correct / num_data

    def regularization_loss(
        self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.params.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss


class EWC(L2):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.n_fisher_sample = None
        self.empFI = False

    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        logger.debug("Computing EWC")

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        # Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            logger.info("Sample", self.n_fisher_sample, "for estimating the F matrix.")
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(
                subdata, shuffle=True, num_workers=2, batch_size=1
            )

        self.model.eval()
        # Accumulate the square of gradients
        for data in dataloader:
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.model(x)

            pred = torch.argmax(logit, dim=-1)
            if self.empFI:  # Use groundtruth label (default is without this)
                pred = y

            loss = self.criterion(logit, pred)
            reg_loss = self.regularization_loss()
            loss += reg_loss

            self.model.zero_grad()
            loss.backward()

            for n, p in importance.items():
                # Some heads can have no grad if no loss applied on them.
                if self.params[n].grad is not None:
                    p += (self.params[n].grad ** 2) * len(x) / len(dataloader.dataset)

        return importance


class RWalk(L2):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.score = []
        self.fisher = []
        self.n_fisher_sample = None
        self.empFI = False
        self.alpha = 0.5
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.params.items():
            self.epoch_score[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized

    def update_fisher_and_score(
        self, new_params, old_params, new_grads, old_grads, epsilon=0.001
    ):
        for n, _ in self.params.items():
            if n in old_grads:
                new_p = new_params[n]
                old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                self.epoch_score[n] += (new_grad - old_grad) / (
                    0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                if self.epoch_score[n].max() > 1000:
                    logger.debug(
                        "Too large score {} / {}".format(
                            new_grad - old_grad,
                            0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon,
                        )
                    )
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[
                        n
                    ] + self.alpha * new_grad ** 2

    def _train(self, train_loader, optimizer, epoch, total_epochs):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            old_params = {n: p.detach() for n, p in self.params.items()}
            old_grads = {
                n: p.grad.detach() for n, p in self.params.items() if p.grad is not None
            }

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (
                    1 - lam
                ) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            reg_loss = self.regularization_loss()
            loss += reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            new_params = {n: p.detach() for n, p in self.params.items()}
            new_grads = {
                n: p.grad.detach() for n, p in self.params.items() if p.grad is not None
            }

            self.update_fisher_and_score(new_params, old_params, new_grads, old_grads)

            _, preds = logit.topk(self.topk, 1, True, True)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def calculate_importance(self, dataloader):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.params.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.params.items():
            importance[n] = self.fisher[-1][n] + self.score[-1][n]
            self.epoch_score[n] = self.params[n].clone().detach().fill_(0)
        return importance

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

    def update_model(self, inputs, targets, optimizer):
        out = self.model(inputs)
        loss = self.regularization_loss(out, targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.detach(), out

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

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        for epoch in range(n_epoch):
            # learning rate scheduling from
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                epoch=epoch,
                total_epochs=n_epoch,
            )
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            if best_acc < eval_dict["avg_acc"]:
                best_acc = eval_dict["avg_acc"]

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }

        return best_acc, eval_dict

    def _train(self, train_loader, optimizer, epoch, total_epochs):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (
                    1 - lam
                ) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            reg_loss = self.regularization_loss()
            loss += reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            _, preds = logit.topk(self.topk, 1, True, True)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        n_batches = len(train_loader)
        logger.info(
            "[{}/{}]\tloss: {:.4f}\tacc: {:.4f}".format(
                epoch, total_epochs, total_loss / n_batches, correct / num_data
            )
        )

        return total_loss / n_batches, correct / num_data

    def regularization_loss(
        self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.params.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss


class EWC(L2):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.n_fisher_sample = None
        self.empFI = False

    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        logger.info("Computing EWC")

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        # Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            logger.info("Sample", self.n_fisher_sample, "for estimating the F matrix.")
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(
                subdata, shuffle=True, num_workers=2, batch_size=1
            )

        self.model.eval()
        # Accumulate the square of gradients
        for data in dataloader:
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.model(x)

            pred = torch.argmax(logit, dim=-1)
            if self.empFI:  # Use groundtruth label (default is without this)
                pred = y

            loss = self.criterion(logit, pred)
            reg_loss = self.regularization_loss()
            loss += reg_loss

            self.model.zero_grad()
            loss.backward()

            for n, p in importance.items():
                # Some heads can have no grad if no loss applied on them.
                if self.params[n].grad is not None:
                    p += (self.params[n].grad ** 2) * len(x) / len(dataloader.dataset)

        return importance


class RWalk(L2):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.score = []
        self.fisher = []
        self.n_fisher_sample = None
        self.empFI = False
        self.alpha = 0.5
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.params.items():
            self.epoch_score[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized

    def update_fisher_and_score(
        self, new_params, old_params, new_grads, old_grads, epsilon=0.001
    ):
        for n, _ in self.params.items():
            if n in old_grads:
                new_p = new_params[n]
                old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                self.epoch_score[n] += (new_grad - old_grad) / (
                    0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                if self.epoch_score[n].max() > 1000:
                    logger.debug(
                        "Too large score {} / {}".format(
                            new_grad - old_grad,
                            0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon,
                        )
                    )
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[
                        n
                    ] + self.alpha * new_grad ** 2

    def _train(self, train_loader, optimizer, epoch, total_epochs):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            old_params = {n: p.detach() for n, p in self.params.items()}
            old_grads = {
                n: p.grad.detach() for n, p in self.params.items() if p.grad is not None
            }

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (
                    1 - lam
                ) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            reg_loss = self.regularization_loss()
            loss += reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            new_params = {n: p.detach() for n, p in self.params.items()}
            new_grads = {
                n: p.grad.detach() for n, p in self.params.items() if p.grad is not None
            }

            self.update_fisher_and_score(new_params, old_params, new_grads, old_grads)

            _, preds = logit.topk(self.topk, 1, True, True)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def calculate_importance(self, dataloader):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.params.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.params.items():
            importance[n] = self.fisher[-1][n] + self.score[-1][n]
            self.epoch_score[n] = self.params[n].clone().detach().fill_(0)
        return importance
