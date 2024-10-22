from fastapi import FastAPI
from myapp.controller import router

app = FastAPI()

app.include_router(router, prefix="/images", tags=["images"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Generation API!"}
