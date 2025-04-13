from cog import BasePredictor, BaseModel

class ModelOutput(BaseModel):
    pass

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self) -> ModelOutput:
        """Run template predictions on the input."""
        return ModelOutput()
