import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchvision import models
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, Specificity

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

        # Metrics
        metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", average="weighted", num_classes=self.hparams.out_features),
            'prec': Precision(task="multiclass", average="weighted",
                                           num_classes=self.hparams.out_features),
            'recall': Recall(task="multiclass", average="weighted", num_classes=self.hparams.out_features),
            'f1': F1Score(task="multiclass", average="weighted", num_classes=self.hparams.out_features),
        })
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

        perclass = MetricCollection({
            'acc': Accuracy(task="multiclass", average=None, num_classes=self.hparams.out_features),
            'prec': Precision(task="multiclass", average=None, num_classes=self.hparams.out_features),
            'recall': Recall(task="multiclass", average=None, num_classes=self.hparams.out_features),
            'f1': F1Score(task="multiclass", average=None, num_classes=self.hparams.out_features),
            'auroc': AUROC(task="multiclass", average=None, num_classes=self.hparams.out_features),
            'avprc': AveragePrecision(task="multiclass", average=None, num_classes=self.hparams.out_features),
            'spec': Specificity(task="multiclass", average=None, num_classes=self.hparams.out_features),
        })
        self.test_metrics = perclass.clone(prefix='test/')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.base.features(x)
        emb   = self.head(feats)
        logits = self.classifier(emb)
        return logits

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.opt_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.lr_pat,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc_epoch"
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss, on_epoch=True)
        self.log_dict(self.train_metrics(logits, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("val/loss", loss, on_epoch=True)
        self.log_dict(self.val_metrics(logits, y), on_epoch=True)
        return {"logits": logits, "labels": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("test/loss", loss, on_epoch=True)
        return {"logits": logits, "labels": y}

    def test_epoch_end(self, outputs):
        all_logits = torch.cat([o['logits'] for o in outputs], dim=0)
        all_labels = torch.cat([o['labels'] for o in outputs], dim=0)
        self.log_dict(self.test_metrics(all_logits, all_labels), on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y, names = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        return {"preds": preds, "logits": logits, "labels": y, "names": names}