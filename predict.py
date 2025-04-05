from typing import Optional
import zipfile
import os
import torch
import requests
import json
from cog import BasePredictor, Path, Input, BaseModel
from super_gradients.training import models, Trainer
from super_gradients.common.object_names import Models

class ModelOutput(BaseModel):
    prediction_json: Path
    prediction_image: Path

class Predictor(BasePredictor):
    def setup(self, num_classes: int = 17):
        """Initialize the YOLO-NAS model with downloaded weights."""
        self.weights_path = "yolo_nas_pose_l_coco_pose.pth"

        # Download the weights if they don't already exist
        if not os.path.exists(self.weights_path):
            print("Downloading YOLO-NAS-POSE weights...")
            url = "https://huggingface.co/bdsqlsz/YOLO_NAS/resolve/main/yolo_nas_pose_l_coco_pose.pth"
            response = requests.get(url, stream=True)
            with open(self.weights_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

        # Load the model with the downloaded weights
        self.model = models.get("yolo_nas_pose_l", checkpoint_path=self.weights_path, num_classes=num_classes)

    def predict(self,
                input_path: Path = Input(description="Path to the input image or video"),
                confidence: float = Input(description="Confidence threshold for predictions", default=0.6),
                ) -> ModelOutput:
        """Run YOLO-NAS predictions on the input."""
        input_path = str(input_path)

        # Ensure the model is loaded with the correct weights
        if not hasattr(self, 'weights_path') or self.weights_path is None:
            raise ValueError("Weights path is not set. Please ensure setup is called correctly.")

        result = self.model.predict(input_path, conf=confidence)

        coco_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle"
        ]

        coco_skeleton = [
            [0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
        ]

        keypoints_with_parents = []
        for instance in result.prediction.poses:
            instance_keypoints = []
            for idx, keypoint in enumerate(instance):
                parent = next((pair[0] for pair in coco_skeleton if pair[1] == idx), None)
                instance_keypoints.append({
                    "keypoint": coco_keypoints[idx],
                    "coordinates": keypoint[:2].tolist(),  # Convert ndarray to list
                    "confidence": float(keypoint[2]),  # Convert confidence to float
                    "parent": coco_keypoints[parent] if parent is not None else None
                })
            keypoints_with_parents.append(instance_keypoints)

        # Save the result image
        prediction_image = "prediction_image.png"
        result.save(prediction_image)

        # Combine predictions into a single output
        combined_predictions = {
            "bboxes": [bbox.tolist() for bbox in result.prediction.bboxes_xyxy],
            "keypoints": keypoints_with_parents,
            "scores": [float(score) for score in result.prediction.scores],
            "skeleton": coco_skeleton,
            "image": prediction_image
        }

        # Save combined predictions to JSON
        combined_prediction_json = "combined_prediction.json"
        with open(combined_prediction_json, "w") as json_file:
            json.dump(combined_predictions, json_file)

        return ModelOutput(
            prediction_json=Path(combined_prediction_json),
            prediction_image=Path(prediction_image)
        )
