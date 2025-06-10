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

class MoldVGG_Default(L.LightningModule):
    """
    VGG16-based classifier for standard 3-channel inputs.
    Uses a frozen VGG16 backbone, drops its classifier head, and adds a dynamic head and metrics.
    """

    def __init__(
        self,
        config: dict  # All settings loaded from Defaults.yaml
    ) -> None:
        super().__init__()
        self.save_hyperparameters("config")

        # Unpack relevant parameters from config
        self.opt_lr = config['LEARNING_RATE']
        self.lr_pat = config['LR_PATIENCE']  # fallback if not in config
        self.batch_size = config['BATCH_SIZE']
        self.num_epochs = config['NUM_EPOCHS']
        self.dropout_p = config['DROPOUT_RATE']
        self.out_features = len(config['CLASS_NAMES'])
        self.weight_decay = config['WEIGHT_DECAY']

        # Load VGG16, freeze features
        self.base = models.vgg16(weights='DEFAULT')
        for p in self.base.features.parameters():
            p.requires_grad = False

        # Replace the classifier's final layer according to out_features
        self.base.classifier[6] = nn.Linear(in_features=4096, out_features=self.out_features, bias=True)
        self.base.classifier[6].requires_grad = True

        # Optionally adjust dropout
        self.base.classifier[5] = nn.Dropout(self.dropout_p)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.base(x)

    def configure_optimizers(self):
        opt = optim.SGD(self.parameters(), lr=self.opt_lr, momentum=0.9, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     patience=self.lr_pat,
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