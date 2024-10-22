from fastapi import APIRouter, File, HTTPException, UploadFile
import replicate

from myapp.schemas import ErrorResponse, FineTuneRequest, FineTuneResponse, ImageGenerationRequest, ImageGenerationResponse
from myapp.services import fine_tune_model, generate_image


router = APIRouter()

@router.post("/generate", response_model=ImageGenerationResponse, responses={500: {"model": ErrorResponse}})
async def generate_image_endpoint(request: ImageGenerationRequest):
    """
    Generate images based on a given prompt.
    """
    try:
        images = await generate_image(request)
        return ImageGenerationResponse(images=images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@router.post("/fine-tune", response_model=FineTuneResponse)
async def fine_tune_model_endpoint(trigger_word: str, training_images: UploadFile = File(...)):
    """
    Fine-tune an image generation model with the provided training data.
    """
    try:
        output = await fine_tune_model(training_images, trigger_word)
        return FineTuneResponse(message="Fine-tuning started successfully", output=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# @router.post("/generate-from-file", response_model=ImageGenerationResponse, responses={500: {"model": ErrorResponse}})
# async def generate_image_from_file(image: UploadFile = File(...), prompt: str = ""):
#     """
#     Generate an image based on a local file input.
#     """
#     if not image.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

#     try:
#         contents = await image.read()  # Read the file content
#         output = replicate.run(
#             "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
#             input={"image": contents, "prompt": prompt}
#         )
#         return ImageGenerationResponse(images=output)  # Returns the model's output
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

    




# @router.post("/generate-from-url", response_model=ImageGenerationResponse, responses={500: {"model": ErrorResponse}})
# async def generate_image_from_url(image_url: str, prompt: str):
#     """
#     Generate an image based on a URL input.
#     """
#     try:
#         output = replicate.run(
#             "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
#             input={"image": image_url, "prompt": prompt}
#         )
#         return ImageGenerationResponse(images=output)  # Returns the model's output
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
