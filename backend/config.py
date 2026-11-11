
import os
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------
# PROJECT BASE DIRECTORY
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env", override=True)


class Config:

    # ----------------------------------
    # FLASK SETTINGS
    # ----------------------------------

    BASE_DIR = BASE_DIR

    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "careermate-secret-key"
    )

    HOST = os.getenv(
        "HOST",
        "127.0.0.1"
    )

    PORT = int(
        os.getenv("PORT", 5000)
    )

    DEBUG = os.getenv(
        "DEBUG",
        "True"
    ).lower() in ("true", "1", "t")

    # ----------------------------------
    # GEMINI API SETTINGS
    # ----------------------------------

    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY",
        ""
    ).strip()

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite"
    ).strip()

    GEMINI_EMBEDDING_MODEL = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "text-embedding-004"
    ).strip()

    # ----------------------------------
    # UPLOAD SETTINGS
    # ----------------------------------

    UPLOAD_FOLDER = BASE_DIR / "uploads"

    ALLOWED_EXTENSIONS = {
        "pdf"
    }

    MAX_UPLOAD_SIZE_MB = int(
        os.getenv(
            "MAX_UPLOAD_SIZE_MB",
            16
        )
    )

    MAX_CONTENT_LENGTH = (
        MAX_UPLOAD_SIZE_MB * 1024 * 1024
    )

    # ----------------------------------
    # CHROMADB SETTINGS
    # ----------------------------------

    CHROMA_PERSIST_DIRECTORY = (
        BASE_DIR /
        os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "data/chromadb"
        )
    )

    # ----------------------------------
    # TEXT PROCESSING SETTINGS
    # ----------------------------------

    CHUNK_SIZE = 800

    CHUNK_OVERLAP = 150

    TOP_K_RETRIEVAL = 4

    # ----------------------------------
    # CAREER DOMAIN SETTINGS
    # ----------------------------------

    DOMAIN_REJECTION_MESSAGE = (
        "I'm CareerMate, your career guidance assistant. "
        "I can only answer questions related to careers, "
        "education, skills and professional development."
    )

    CAREER_DOMAINS = [

        "Career Guidance & Planning",

        "Higher Education",

        "Professional Development",

        "Job Preparation",

        "Technical & Soft Skills",

        "Resume Building",

        "Interview Preparation",

        "Data Science",

        "Artificial Intelligence",

        "Machine Learning",

        "Software Engineering",

        "Cybersecurity"

    ]


# ----------------------------------
# CREATE REQUIRED DIRECTORIES
# ----------------------------------

Config.UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

Config.CHROMA_PERSIST_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)
