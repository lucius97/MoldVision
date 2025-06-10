"""
MoldVision
------------

A PyTorch-Lightning module that takes two images,
extracts features via a VGG16 backbone (seperate for top and bottom images)
and classifies based on concatenated embeddings.

Usage:
    model = Moldvision(opt_lr=0.01, lr_pat=5, batch_size=32, num_epochs=20)
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)
"""


import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchvision import models


class MoldVision(L.LightningModule):
    """Twin‐branch VGG16 classifier for paired‐image inputs."""

    def __init__(self,
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

        # 1) Load & freeze VGG16 backbone, drop its classifier
        self.base = models.vgg16(weights='DEFAULT')
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.classifier = nn.Identity()

        # 2) Helper to build each branch head (features → Flatten → 4096 → ReLU → Dropout)
        def make_branch():
            return nn.Sequential(
                self.base.features,  # shared conv backbone
                nn.Flatten(),  # → (batch, 512*7*7)
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p)
            )

        # 3) instantiate two separate heads (their Linear layers do *not* share weights)
        self.branch_top = make_branch()
        self.branch_bottom = make_branch()

        # 4) Classification head: 4096×2 → 4096 → ReLU → Dropout → 4096 → ReLU → Dropout → out_features
        self.classifier = nn.Sequential(
            nn.Linear(4096 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(4096, self.out_features)
        )

        # 5) Loss Function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img_f, img_b):
        # Extract & pool features
        f1 = self.branch_top(img_f)
        f2 = self.branch_bottom(img_b)
        # Concatenate and classify
        return self.classifier(torch.cat([f1, f2], dim=1))

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
        imgs_f, imgs_b, labels = batch[:3]
        logits = self(imgs_f, imgs_b)
        loss = self.loss_fn(logits, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs_f, imgs_b, labels = batch[:3]
        logits = self(imgs_f, imgs_b)
        loss = self.loss_fn(logits, labels)
        self.log('val/loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs_f, imgs_b, labels = batch[:3]
        logits = self(imgs_f, imgs_b)
        return {'logits': logits, 'labels': labels}

    def test_epoch_end(self, outputs):
        all_logits = torch.cat([o['logits'] for o in outputs], dim=0)
        all_labels = torch.cat([o['labels'] for o in outputs], dim=0)


    def predict_step(self, batch, batch_idx):
        imgs_f, imgs_b, labels, names = batch
        logits = self(imgs_f, imgs_b)
        preds = logits.argmax(dim=1)
        return {'preds': preds,
                'logits': logits,
                'labels': labels,
                'name': names}