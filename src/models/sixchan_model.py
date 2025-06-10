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
            nn.Dropout(self.dropout_p),
        )
        self.classifier = nn.Linear(4096, self.out_features)

        # 4) Loss Function
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
        cat_imgs, labels = batch[:2]
        logits = self(cat_imgs)
        loss = self.loss_fn(logits, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cat_imgs, labels = batch[:2]
        logits = self(cat_imgs)
        loss = self.loss_fn(logits, labels)
        self.log('val/loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        cat_imgs, labels = batch[:2]
        logits = self(cat_imgs)
        return {'logits': logits, 'labels': labels}

    def predict_step(self, batch, batch_idx):
        cat_imgs, labels, names = batch
        logits = self(cat_imgs)
        preds = logits.argmax(dim=1)
        return {'preds': preds,
                'logits': logits,
                'labels': labels,
                'name': names}