from typing import Optional
import zipfile
import os
import torch
import requests
import json
from cog import BasePredictor, Path, Input, BaseModel
from super_gradients.training import models, Trainer
from super_gradients.common.object_names import Models

class TrainingOutput(BaseModel):
    weights: Path

def train(
    training_dataset: Optional[Path] = Input(description="Zip file containing the training dataset. If not provided, base weights will be used.", default=None),
    epochs: int = Input(description="Number of fine-tuning epochs", default=10),
    batch_size: int = Input(description="Batch size for training", default=16),
    learning_rate: float = Input(description="Learning rate for training", default=1e-4),
) -> TrainingOutput:
    """Fine-tune the YOLO-NAS model on a custom dataset."""
    # Unzip the training dataset
    if training_dataset is not None:
        training_dataset_path = "training_dataset.zip"
        training_dataset.rename(training_dataset_path)
        extracted_dataset_dir = "extracted_training_dataset"
        with zipfile.ZipFile(training_dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dataset_dir)
    else:
        print("No training dataset provided. Using base weights.")

    # Set up the trainer
    model = Trainer(experiment_name="yolo_nas_pose_fine_tune", ckpt_root_dir="checkpoints")

    # Define training parameters
    train_params = {
        "max_epochs": epochs,
        "batch_size": batch_size,
        "lr_mode": "cosine",
        "initial_lr": learning_rate,
        "optimizer": "Adam",
        "loss": "yolo_pose_loss",
    }
    if training_dataset is not None:
        train_params["dataset_params"] = {
            "train_data_path": extracted_dataset_dir,
            "val_data_path": extracted_dataset_dir,  # Assuming validation data is in the same directory
            "classes": ["class1", "class2"],  # Replace with actual class names
        }
    else:
        print("No training dataset provided. Using base weights.")
    torch.save(model.state_dict(), fine_tuned_weights_path)

    return TrainingOutput(weights=Path(fine_tuned_weights_path))
