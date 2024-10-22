import os
from typing import List
from fastapi import HTTPException, UploadFile
import replicate
from myapp.config import settings
from myapp.schemas import FineTuneRequest, ImageGenerationRequest

# Initialize the Replicate client
replicate.Client(api_token=settings.REPLICATE_API_TOKEN)

async def generate_image(request: ImageGenerationRequest) -> List[str]:
    """
    Generate images based on a given prompt.
    """
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": request.prompt, "num_images": request.num_images}
    )
    return output



async def fine_tune_model(training_images: UploadFile, trigger_word: str):
    # Check if the training images zip file is provided
    if not training_images:
        raise HTTPException(status_code=400, detail="No training images zip file provided.")

    # Save the uploaded zip file temporarily
    temp_zip_path = f"./{training_images.filename}"  # Save in the current directory
    
    # Log the temporary path for debugging
    print(f"Saving file to: {temp_zip_path}")

    # Save the uploaded file
    with open(temp_zip_path, "wb") as buffer:
        content = await training_images.read()
        buffer.write(content)

    # Log to confirm the file exists after writing
    if not os.path.exists(temp_zip_path):
        raise HTTPException(status_code=400, detail="Training images zip file not found after upload.")


    try:
        # Use context manager to ensure the file is closed after use
        with open(temp_zip_path, "rb") as input_file:
            print("Calling Replicate API...")
            output = replicate.run(
                "ostris/flux-dev-lora-trainer",
                input={
                    "input_images": input_file,
                    "trigger_word": trigger_word,
                    "steps": 1000,
                    "public": False
                }
            )
            print("Replicate API call successful.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Log the error details
        raise HTTPException(status_code=500, detail=f"An error occurred during fine-tuning: {str(e)}")
    finally:
        # Clean up: remove the temporary zip file after processing
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    return output