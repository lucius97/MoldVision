from src.models import MoldVGG_Twin, MoldVGG_6Chan, MoldVGG_Default
from src.data   import MoldDataModule
import argparse
import pytorch_lightning as pl

def train_one_fold(args, fold_idx):
    # --- 1) DataModule for this fold ---
    dm = MoldDataModule(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
        num_folds  = args.num_folds,
        fold_idx   = fold_idx,
        num_workers= args.num_workers,
        pin_memory = args.pin_memory,
        transforms = None,              # or your Compose([...])
        model_type = args.model        # 'twin','6chan','default'
    )
    dm.prepare_data()
    dm.setup()

    # --- 2) Pick model ---
    if args.model == "twin":
        model = MoldVGG_Twin(opt_lr=args.lr, lr_pat=args.patience, out_features=args.out_features)
    elif args.model == "6chan":
        model = MoldVGG_6Chan(opt_lr=args.lr, lr_pat=args.patience, out_features=args.out_features)
    else:
        model = MoldVGG_Default(opt_lr=args.lr, lr_pat=args.patience, out_features=args.out_features)

    # --- 3) Trainer & fit---
    trainer = pl.Trainer(
        max_epochs   = args.epochs,
        gpus         = 1 if pl.utilities.rank_zero_only.rank_zero_only() and pl.utilities.imports._IS_INTERACTIVE else 0,
        default_root_dir = f"{args.output_dir}/fold_{fold_idx}"
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["twin","6chan","default"], required=True)
    parser.add_argument("--data_dir",   type=str,   required=True)
    parser.add_argument("--output_dir", type=str,   default="outputs")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--out_features", type=int, default=5)
    parser.add_argument("--num_folds",  type=int,   default=5)
    parser.add_argument("--num_workers",type=int,   default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--fold_idx",   type=int,   default=None,
                        help="If set, trains only this fold; otherwise loops over all folds.")
    args = parser.parse_args()

    if args.fold_idx is None:
        # run every fold 0..num_folds-1
        for fi in range(args.num_folds):
            print(f"\n===== Fold {fi}/{args.num_folds-1} =====")
            train_one_fold(args, fi)
    else:
        train_one_fold(args, args.fold_idx)

if __name__ == "__main__":
    main()