from pydantic import BaseModel, HttpUrl
from typing import List, Any, Optional

class ImageGenerationRequest(BaseModel):
    prompt: str
    num_images: int = 1  # Default to generating 1 image

class ImageGenerationResponse(BaseModel):
    images: List[str]  # List of URLs to the generated images


class FineTuneRequest(BaseModel):
    model_name: str
    prompts: List[str]


class FineTuneResponse(BaseModel):
    message: str
    output: Any  # Can be any type depending on the fine-tuning model's output

class ErrorResponse(BaseModel):
    detail: str  # Error message details
