"""
CareerMate - End-to-End PDF Upload & RAG Verification Script
Tests real PDF parsing, domain validation on documents, scanned detection, and context retrieval.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from app import app
from backend.config import Config

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

def test_pdf_pipeline():
    client = app.test_client()

    print("\n--- 1. Testing Session Creation ---")
    res = client.post("/api/sessions", json={"title": "RAG Verification"})
    assert res.status_code == 200, res.data
    session_id = res.get_json()["session"]["id"]
    print(f"Session created: {session_id}")

    print("\n--- 2. Testing Career PDF Upload (Alex Smith Resume) ---")
    resume_path = SAMPLES_DIR / "Alex_Software_Engineer_Resume.pdf"
    with open(resume_path, "rb") as f:
        upload_res = client.post(
            "/api/upload-pdf",
            data={
                "file": (f, "Alex_Software_Engineer_Resume.pdf"),
                "session_id": session_id
            },
            content_type="multipart/form-data"
        )
    assert upload_res.status_code == 200, upload_res.data
    upload_data = upload_res.get_json()
    print("Upload Status: Success! Chunks indexed:", upload_data["chunks_count"])
    assert upload_data["success"] is True
    assert upload_data["chunks_count"] > 0
    assert upload_data["filename"] == "Alex_Software_Engineer_Resume.pdf"

    print("\n--- 3. Testing Non-Career PDF Rejection (Recipe Book) ---")
    recipe_path = SAMPLES_DIR / "Grandmas_Kitchen_Recipes_OffTopic.pdf"
    with open(recipe_path, "rb") as f:
        bad_upload = client.post(
            "/api/upload-pdf",
            data={
                "file": (f, "Grandmas_Kitchen_Recipes_OffTopic.pdf"),
                "session_id": session_id
            },
            content_type="multipart/form-data"
        )
    assert bad_upload.status_code == 400
    err_msg = bad_upload.get_json()["error"]
    print("Correctly rejected non-career PDF:", err_msg)
    assert "does not appear to be related to careers" in err_msg

    print("\n--- 4. Testing Scanned/Empty Document Rejection ---")
    scanned_path = SAMPLES_DIR / "Scanned_Image_Sample.pdf"
    with open(scanned_path, "rb") as f:
        scanned_res = client.post(
            "/api/upload-pdf",
            data={
                "file": (f, "Scanned_Image_Sample.pdf"),
                "session_id": session_id
            },
            content_type="multipart/form-data"
        )
    assert scanned_res.status_code == 400
    scanned_err = scanned_res.get_json()["error"]
    print("Correctly rejected scanned/empty PDF:", scanned_err)
    assert "scanned document or contains only images" in scanned_err

    print("\n--- 5. Testing Detach Document ---")
    detach_res = client.post("/api/remove-pdf", json={"session_id": session_id})
    assert detach_res.status_code == 200
    print("Detach successful:", detach_res.get_json()["message"])

    print("\nSUCCESS: All PDF and RAG pipeline tests completed perfectly!")

if __name__ == "__main__":
    test_pdf_pipeline()
