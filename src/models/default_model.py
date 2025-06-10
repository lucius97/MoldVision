"""
MoldVGG_Default
------------

A PyTorch-Lightning module that takes one standard 3-channel image

Usage:
    model = MoldVGG_Default(opt_lr=0.01, lr_pat=5, batch_size=32, num_epochs=20)
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchvision import models
from

class MoldVGG_Default(L.LightningModule):
    """
    VGG16-based classifier for standard 3-channel inputs.
    Uses a frozen VGG16 backbone, drops its classifier head, and adds a dynamic head and metrics.
    """

    def __init__(
        self,
        opt_lr: float,
        lr_pat: int,
        batch_size: int,
        num_epochs: int,
        dropout_p=0.5,
        out_features: int = 5,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        # Load VGG16, freeze features
        self.base = models.vgg16(weights='DEFAULT')
        for p in self.base.features.parameters():
            p.requires_grad = False

        # Replace the classifier's final layer
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.hparams.out_features, bias=True)
        self.model.classifier[6].requires_grad = True

        # 5) Loss Function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img_f, img_b):
        # Extract & pool features
        f1 = self.branch_top(img_f)
        f2 = self.branch_bottom(img_b)
        # Concatenate and classify
        return self.classifier(torch.cat([f1, f2], dim=1))

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.hparams.opt_lr, momentum=0.9, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     patience=self.hparams.lr_pat,
                                                     verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'val/loss',
            }
        }

    def training_step(self, batch, batch_idx):
        imgs, labels = batch[:2]
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch[:2]
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        self.log('val/loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch[:2]
        logits = self(imgs)
        return {'logits': logits, 'labels': labels}

    def predict_step(self, batch, batch_idx):
        imgs, labels, names = batch
        logits = self(imgs)
        preds = logits.argmax(dim=1)
        return {'preds': preds,
                'logits': logits,
                'labels': labels,
                'name': names}
