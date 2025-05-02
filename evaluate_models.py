import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.flop_counter import FlopCounterMode
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

def evaluate_models():
    model_names = [f'efficientnet_b{i}' for i in range(8)]
    results = []
    datamodule = TinyImageNetDatasetModule(batch_size=cfg.BATCH_SIZE)

    for model_name in model_names:
        print(f'=== Evaluating {model_name} ===')
        cfg.MODEL_NAME = model_name
        cfg.WANDB_NAME = f"{model_name}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS['type']}-OneCycleLR{cfg.SCHEDULER_PARAMS['max_lr']}"

        model = SimpleClassifier(
            model_name=model_name,
            num_classes=cfg.NUM_CLASSES,
            optimizer_params=cfg.OPTIMIZER_PARAMS,
            scheduler_params=cfg.SCHEDULER_PARAMS
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='accuracy/val',
            mode='max',
            save_top_k=1,
            filename='{epoch}-{accuracy/val:.4f}'
        )

        trainer = Trainer(
            accelerator=cfg.ACCELERATOR,
            devices=cfg.DEVICES,
            precision=cfg.PRECISION_STR,
            max_epochs=cfg.NUM_EPOCHS,
            check_val_every_n_epoch=cfg.VAL_EVERY_N_EPOCH,
            logger=False,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(model, datamodule=datamodule)
        val_metrics = trainer.validate(ckpt_path='best', datamodule=datamodule)[0]
        print(val_metrics)
        accuracy = val_metrics['accuracy/val']

        param_count = sum(p.numel() for p in model.parameters())

        results.append({
            'model': model_name,
            'params': param_count,
            'accuracy': accuracy
        })

    labels = [r['model'] for r in results]
    params = [r['params'] for r in results]
    acc = [r['accuracy'] for r in results]

    # Plot 1: Params vs Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(params, acc, marker='o')
    for i, label in enumerate(labels):
        plt.text(params[i], acc[i], label, fontsize=9, ha='right')
    plt.xscale('log')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Validation Accuracy')
    plt.title('Params vs Accuracy')
    plt.tight_layout()
    plt.savefig('params_vs_accuracy.png')
    print('Saved params_vs_accuracy.png')

if __name__ == "__main__":
    evaluate_models()
