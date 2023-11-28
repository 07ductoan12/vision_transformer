import torch
import torch.optim.lr_scheduler as lr_scheduler
from customdata import load_dataset
from model.vit import VisionTransformer
from model.mobilevit import MobileViT
from model.mobilenet3_vit_v2 import MobileViTv3_v2
from trainner import Tranner

TRAIN_DIR = "../../datasets/cifar10/train/"
VALID_DIR = "../../datasets/cifar10/valid/"
NUM_WORKERS = 4
BATCH_SIZE = 64
EPOCHS = 70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EXP_NAME = "mobilenetvit_flower"


config = {
    "patch_size": 8,
    "hidden_size": 96,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 96 * 4,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 64,
    "num_channels": 3,
    "qkv_bias": True,
}


def main():
    train_dataloader, valid_dataloader, num_classes = load_dataset(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )
    # vit = VisionTransformer(
    #     image_size=config["image_size"],
    #     patch_size=config["patch_size"],
    #     num_channels=config["num_channels"],
    #     num_classes=num_classes,
    #     hidden_size=config["hidden_size"],
    #     hidden_dropout=config["hidden_dropout_prob"],
    #     num_hidden_layers=config["num_hidden_layers"],
    #     intermediate_size=config["intermediate_size"],
    #     num_attention_heads=config["num_attention_heads"],
    #     attention_dropout=config["attention_probs_dropout_prob"],
    #     bias=config["qkv_bias"],
    # )
    # vit.to(DEVICE)

    model = MobileViTv3_v2(
        image_size=(224, 224),
        width_multiplier=1,
        num_classes=num_classes,
        patch_size=(2, 2),
    )
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.3, total_iters=10
    )

    trainner = Tranner(
        epochs=EPOCHS,
        device=DEVICE,
        disable_progress_bar=True,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model=model,
        config=config,
        exp_name=EXP_NAME,
        scheduler=scheduler,
    )
    trainner.train(train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
