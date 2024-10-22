import os

class Settings:
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN")

settings = Settings()
