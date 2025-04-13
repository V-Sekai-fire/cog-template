# cog-yolo-nas-pose

A serverless deployment of the state-of-the-art YOLO-NAS Pose model on Replicate.

## Overview

This repository contains the code necessary to deploy the YOLO-NAS Pose model as a serverless endpoint on Replicate. YOLO-NAS Pose integrates object detection and pose estimation into a single, efficient pass. Its advanced backbone and neck architecture, paired with a pose estimation head optimized by AutoNAC, deliver real-time performance with high accuracy.

For more details about YOLO-NAS Pose, see:

- [LearnOpenCV: YOLO-NAS Pose](https://learnopencv.com/yolo-nas-pose/)
- [SuperGradients YOLO-NAS Pose Documentation](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md)

## How to Use Custom Models

Customizing the deployment with your own model weights is simple:

1. Add your model weights file to the root of this repository.
2. Update the model initialization in [predict.py](predict.py). For example:

   ```python
   # Change this line in predict.py
   self.model = models.get("yolo_nas_pose_l", checkpoint_path="your-custom-model.pth", num_classes=17)
   ```

3. Follow the [Replicate deployment guide](https://replicate.com/docs/guides/deploy-a-custom-model) to publish your model.

## How to use with API

Learn more about the available API endpoints from the [Replicate API Documentation](https://replicate.com/hardikdava/rf-detr/api).

## Local Development and Testing

To test the model locally before deployment:

```bash
# Install cog if you haven't already
pip install cog

# Run a prediction with a local image
cog predict -i image=@/path/to/your/image.jpg
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Cog
- yolo-nas-pose

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/V-Sekai-fire/cog-yolo-nas-pose/blob/main/LICENSE) file for details.
