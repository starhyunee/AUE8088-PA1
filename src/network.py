# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
from torchvision.models.resnet import ResNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting



# Depthwise Separable Convolution 
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
#         self.norm = nn.BatchNorm2d(out_ch)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return x



# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self):
        super().__init__(num_classes = 200)
            
        # [TODO] Modify feature extractor part in AlexNet


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# Modify version
# class MyNetwork(AlexNet):
#     def __init__(self):
#         super().__init__()
#         # Modify Feature extractor
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),

#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.BatchNorm2d(192),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),

#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Dropout2d(p=0.3), 
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x




# # advanced AlexNet 
# class MyNetwork(nn.Module):
#     def __init__(self, num_classes=200):
#         super().__init__()
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.SiLU(),
#             nn.MaxPool2d(3, stride=2)
#         )
#         self.stage2 = nn.Sequential(
#             DepthwiseSeparableConv(64, 128),
#             nn.MaxPool2d(2),
#         )
#         self.stage3 = nn.Sequential(
#             DepthwiseSeparableConv(128, 256),
#             nn.MaxPool2d(2),
#         )
#         self.stage4 = nn.Sequential(
#             DepthwiseSeparableConv(256, 256),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256, 512),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = self.classifier(x)
#         return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet101',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')

        if scheduler_type == 'OneCycleLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_params.pop('max_lr'),
                    steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
                    **scheduler_params
                ),
                'interval': 'step',
                'frequency': 1
            }
        else:
            scheduler = {
                'scheduler': getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params),
                'interval': 'epoch',
                'frequency': 1
            }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.f1score.update(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)


    def on_validation_epoch_end(self):
        f1score_per_class = self.f1score.compute()
        macro_f1 = f1score_per_class.mean()
        self.log("f1score/val", macro_f1, prog_bar=True, logger=True)
        #self.f1score.reset()

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])



