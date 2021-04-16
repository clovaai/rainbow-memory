import logging
import random

from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from methods.finetune import Finetune
from utils.train_utils import select_model, select_optimizer

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class GDumb(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.n_tasks = kwargs["n_tasks"]

    def train(self, cur_iter, n_epoch, batch_size, n_worker):
        logger.info("Reset model parameters")
        self.model = select_model(
            self.model_name, self.dataset, self.num_learning_class
        )
        self.model = self.model.to(self.device)

        # Initialize the optimizer and scheduler
        logger.info("Reset the optimizer and scheduler states")
        self.optimizer, self.scheduler = select_optimizer(
            self.opt_name, self.lr, self.model, self.sched_name
        )

        # Use only memory-stored samples.
        train_list = self.memory_list
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
        eval_dict = {}
        self.model = self.model.to(self.device)
        for epoch in range(n_epoch):
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:
                self.scheduler.step()

            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch=epoch,
                total_epochs=n_epoch,
            )

            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train_acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test_acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])
        return best_acc, eval_dict

    def before_task(self, datalist, cur_iter, init_model, init_opt):
        # always init_mode=True
        super().before_task(datalist, cur_iter, True, init_opt)
        # Do post-task operations before the training
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        self.update_memory(cur_iter)

    def update_memory(self, cur_iter):
        num_class = self.num_learning_class

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = self.streamed_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                self.memory_list = self.equal_class_sampling(candidates, num_class)

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # memory update happens only once per task iteration.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

    def after_task(self, cur_iter):
        pass
