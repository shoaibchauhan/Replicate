# Image Generation API

Welcome to the Image Generation API! This FastAPI application allows you to generate images based on prompts, fine-tune image generation models, and generate images from uploaded files.

## Features

- Generate images based on a prompt.
- Fine-tune image generation models with custom training data.
- Generate images from local file uploads.

## Technologies Used

- FastAPI
- Replicate API
- Pydantic for data validation

## Prerequisites

- Python 3.7 or higher
- Virtual Environment (recommended)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create and activate a virtual environment:**

    ```bash    
    python -m venv venv
    # For Windows
    venv\Scripts\activate
    # For macOS/Linux
    source venv/bin/activate

3. **Install the required packages:**

    ```bash     
    pip install -r requirements


4. **Run the application:**

    ```bash
    uvicorn main.myapp:app --reload

5. **Access the API documentation:**

    ```bash
    http://127.0.0.1:8000/docs    


