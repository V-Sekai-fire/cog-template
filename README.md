# cog-generic-template

A serverless deployment of the state-of-the-art Template Model on Replicate.

## Overview

This repository contains the code necessary to deploy the Template Model as a serverless endpoint on Replicate.

## How to Use Custom Models

Customizing the deployment with your own model weights is simple:

1. Add your model weights file to the root of this repository.
2. Update the model initialization in [predict.py](predict.py). For example:

   ```python
   # Change this line in predict.py
   self.model = models.get("generic_template_model", checkpoint_path="your-custom-model.pth", num_classes=17)
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

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/V-Sekai-fire/cog-template/blob/main/LICENSE) file for details.