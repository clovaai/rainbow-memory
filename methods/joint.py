from methods.finetune import Finetune
from utils.train_utils import select_model, select_optimizer


class Joint(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.model = select_model(self.model_name, self.dataset, n_classes)
        self.optimizer, self.scheduler = select_optimizer(
            kwargs["opt_name"], kwargs["lr"], self.model
        )
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.num_learning_class = n_classes

    def before_task(self, datalist, cur_iter, init_model, init_opt):
        pass

    def after_task(self, cur_iter):
        pass

    def update_memory(self, cur_iter):
        pass
