import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings validating configuration values from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    SUPABASE_URL: str
    SUPABASE_KEY: str
    GEMINI_API_KEY: str
    GROQ_API_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str
    EXPECTED_TELEGRAM_USER_ID: int
    GMAIL_USER: str = ""
    GMAIL_APP_PASSWORD: str = ""

# Instantiated settings object to be imported across the application
settings = Settings()
