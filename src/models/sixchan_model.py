"""
MoldVGG_6Chan
------------

A PyTorch-Lightning module that takes
one 6-channel image comprised out of 2 Images concatenated together

Usage:
    model = MoldVGG_6Chan(opt_lr=0.01, lr_pat=5, batch_size=32, num_epochs=20)
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchvision import models

class MoldVGG_6Chan(L.LightningModule):
    """
    VGG16-based classifier for 6-channel inputs.
    Adapts the VGG16 backbone to accept 6 channels, drops its classifier,
    and adds a lightweight head plus metrics.
    """

    def __init__(
        self,
        opt_lr: float,
        lr_pat: int,
        batch_size: int,
        num_epochs: int,
        dropout_p = 0.5,
        out_features: int = 5,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        # 1) Load VGG16, adapt to 6-channel input, and freeze features
        self.base = models.vgg16(weights='DEFAULT')
        self.base.features[0] = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        for p in self.base.features.parameters():
            p.requires_grad = False
        # Drop the original classifier
        self.base.classifier = nn.Identity()

        # 2) Compute flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 6, 224, 224)
            feat = self.base.features(dummy)
            flat_size = feat.numel()

        # 3) Define a lightweight head and final classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(flat_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )
        self.classifier = nn.Linear(4096, out_features)

        # 4) Loss Function
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
        imgs_f, imgs_b, labels = batch[:3]
        logits = self(imgs_f, imgs_b)
        loss = self.loss_fn(logits, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics(logits, labels),
                      on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs_f, imgs_b, labels = batch[:3]
        logits = self(imgs_f, imgs_b)
        loss = self.loss_fn(logits, labels)
        self.log('val/loss', loss, on_epoch=True)
        self.log_dict(self.val_metrics(logits, labels), on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs_f, imgs_b, labels = batch[:3]
        logits = self(imgs_f, imgs_b)
        return {'logits': logits, 'labels': labels}

    def test_epoch_end(self, outputs):
        all_logits = torch.cat([o['logits'] for o in outputs], dim=0)
        all_labels = torch.cat([o['labels'] for o in outputs], dim=0)
        self.log_dict(self.test_metrics(all_logits, all_labels),
                      on_epoch=True)

    def predict_step(self, batch, batch_idx):
        imgs_f, imgs_b, labels, names = batch
        logits = self(imgs_f, imgs_b)
        preds = logits.argmax(dim=1)
        return {'preds': preds,
                'logits': logits,
                'labels': labels,
                'name': names}