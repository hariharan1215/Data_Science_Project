"""
CareerMate - AI-Powered Career Guidance Chatbot
Flask REST API Backend Application
"""
import os
import json
import logging
import time
from pathlib import Path
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.config import Config
from backend.domain_validator import DomainValidator
from backend.pdf_processor import PDFProcessor, PDFProcessingError
from backend.gemini_client import GeminiClient
from backend.rag_service import RAGService
from backend.session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CareerMate")

# Initialize Flask App
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
CORS(app)

# Initialize Core Services
gemini_client = GeminiClient()
rag_service = RAGService()
session_manager = SessionManager()

# -------------------------------------------------------------
# Web & UI Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    """Serves the main CareerMate chat interface."""
    return render_template("index.html")

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns application health and configuration status."""
    return jsonify({
        "status": "healthy",
        "app_name": "CareerMate",
        "version": "1.0.0",
        "api_key_configured": gemini_client.is_configured(),
        "model": gemini_client.model_name,
        "embedding_model": gemini_client.embedding_model_name,
        "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
        "max_upload_size_mb": Config.MAX_UPLOAD_SIZE_MB
    })

@app.route("/api/config/key", methods=["POST"])
def update_api_key():
    """Allows setting or updating the Gemini API key dynamically."""
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key", "").strip()

    if not api_key:
        return jsonify({"success": False, "error": "API key cannot be empty."}), 400

    # Update in runtime and in .env file
    gemini_client.api_key = api_key
    gemini_client._init_client()

    env_path = Config.BASE_DIR / ".env"
    try:
        content = ""
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
        if "GEMINI_API_KEY=" in content:
            import re
            content = re.sub(r"GEMINI_API_KEY=.*", f"GEMINI_API_KEY={api_key}", content)
        else:
            content += f"\nGEMINI_API_KEY={api_key}\n"
        env_path.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not persist API key to .env: {e}")

    return jsonify({
        "success": True,
        "message": "Gemini API key updated successfully.",
        "api_key_configured": gemini_client.is_configured()
    })

# --- Sessions & Conversation History ---

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """Lists all chat sessions."""
    sessions = session_manager.list_sessions()
    return jsonify({"success": True, "sessions": sessions})

@app.route("/api/sessions", methods=["POST"])
def create_session():
    """Creates a new conversation session."""
    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Career Chat")
    session = session_manager.create_session(title=title)
    return jsonify({"success": True, "session": session})

@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    """Retrieves session messages and state."""
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({"success": False, "error": "Session not found."}), 404
    return jsonify({"success": True, "session": session})

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Deletes a conversation session and clears associated RAG storage."""
    rag_service.delete_session_documents(session_id)
    success = session_manager.delete_session(session_id)
    return jsonify({"success": success})

@app.route("/api/sessions/<session_id>/clear", methods=["POST"])
def clear_session(session_id):
    """Clears all conversation messages within the current session."""
    session_manager.clear_session_messages(session_id)
    return jsonify({"success": True, "message": "Conversation history cleared."})

# --- Chat & Career Guidance ---

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint with domain enforcement, conversation memory,
    and RAG context retrieval.
    """
    chat_started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    query = data.get("message", "").strip()
    session_id = data.get("session_id", "").strip()

    if not query:
        return jsonify({
            "success": False,
            "error": "Query cannot be empty."
        }), 400

    # Ensure a valid session
    session = session_manager.get_session(session_id)
    session_id = session["id"]

    # 1. Domain Validation Check
    is_valid, validation_status = DomainValidator.validate_query(query)
    validation_ms = (time.perf_counter() - chat_started) * 1000

    if not is_valid:
        # Off-topic query rejected
        rejection_response = Config.DOMAIN_REJECTION_MESSAGE
        session_manager.add_message(session_id, sender="user", text=query)
        session_manager.add_message(session_id, sender="ai", text=rejection_response)
        logger.info(
            "Chat completed session=%s path=off_topic total_ms=%.1f",
            session_id,
            (time.perf_counter() - chat_started) * 1000
        )
        return jsonify({
            "success": True,
            "response": rejection_response,
            "is_off_topic": True,
            "citations": [],
            "session_id": session_id
        })

    # Record user message in session
    session_manager.add_message(session_id, sender="user", text=query)

    # 2. Check for friendly greeting
    if validation_status == "GREETING":
        greeting_reply = (
            "Hello! I'm CareerMate, your dedicated AI Career Guidance Assistant.\n\n"
            "I'm here to help you navigate your professional journey, education, and tech careers. "
            "Here are some ways we can work together:\n\n"
            "- Career Planning & Transitions: Explore pathways in AI, Data Science, Software Development, Cloud, or Cyber Security.\n"
            "- Resume & Portfolio Reviews: Upload your PDF resume or notes to get targeted feedback and ATS optimization tips.\n"
            "- Interview Preparation: Practice behavioral, technical, and coding interview strategies.\n"
            "- Skill & Certification Roadmaps: Discover the highest-impact skills, frameworks, and courses for your target roles.\n\n"
            "How can I assist your career growth today?"
        )
        session_manager.add_message(session_id, sender="ai", text=greeting_reply)
        logger.info(
            "Chat completed session=%s path=greeting total_ms=%.1f",
            session_id,
            (time.perf_counter() - chat_started) * 1000
        )
        return jsonify({
            "success": True,
            "response": greeting_reply,
            "citations": [],
            "session_id": session_id
        })

    # 3. Retrieve RAG context if a PDF is attached to this session
    active_document = session.get("uploaded_document")
    context_chunks = []
    rag_ms = 0.0

    if active_document:
        rag_started = time.perf_counter()
        try:
            logger.info("Retrieving RAG context for session %s.", session_id)
            context_chunks = rag_service.retrieve_relevant_chunks(
                session_id=session_id,
                query=query
            )
            logger.info("Retrieved %s document chunks.", len(context_chunks))
        except Exception as e:
            logger.exception("Document retrieval failed for session %s: %s", session_id, e)
        finally:
            rag_ms = (time.perf_counter() - rag_started) * 1000

    # Keep only the latest two prior messages needed by Gemini.
    history = session["messages"][-3:-1]
    logger.info(
        "Chat prepared session=%s model=%s document_attached=%s "
        "validation_ms=%.1f rag_ms=%.1f preparation_ms=%.1f",
        session_id,
        gemini_client.model_name,
        bool(active_document),
        validation_ms,
        rag_ms,
        (time.perf_counter() - chat_started) * 1000
    )

    if data.get("stream") is True:
        def generate_events():
            answer_parts = []
            first_token_logged = False
            try:
                for text in gemini_client.generate_response_stream(
                    prompt=query,
                    conversation_history=history,
                    context_chunks=context_chunks if context_chunks else None
                ):
                    if text is GeminiClient.stream_restart_token:
                        answer_parts.clear()
                        yield f"data: {json.dumps({'type': 'reset'})}\n\n"
                        continue

                    if not first_token_logged:
                        logger.info(
                            "Chat first token session=%s elapsed_ms=%.1f",
                            session_id,
                            (time.perf_counter() - chat_started) * 1000
                        )
                        first_token_logged = True
                    answer_parts.append(text)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text}, ensure_ascii=False)}\n\n"

                response_text = "".join(answer_parts).strip()
                if not response_text:
                    raise RuntimeError("Gemini returned an empty response.")

                citations = list(dict.fromkeys(
                    f"{chunk.get('source', 'Uploaded Document')} (Page {chunk.get('page', 1)})"
                    for chunk in context_chunks
                ))
                session_manager.add_message(
                    session_id=session_id,
                    sender="ai",
                    text=response_text,
                    citations=citations
                )
                logger.info(
                    "Chat completed session=%s streamed_chars=%s total_ms=%.1f",
                    session_id,
                    len(response_text),
                    (time.perf_counter() - chat_started) * 1000
                )
                yield f"data: {json.dumps({'type': 'done', 'citations': citations}, ensure_ascii=False)}\n\n"
            except Exception as error:
                logger.exception(
                    "Gemini streaming response failed session=%s elapsed_ms=%.1f",
                    session_id,
                    (time.perf_counter() - chat_started) * 1000
                )
                error_message = gemini_client.stream_error_message(error)
                partial_response = "".join(answer_parts).strip()
                if partial_response:
                    partial_response += f"\n\n{error_message}"
                else:
                    partial_response = error_message
                session_manager.add_message(
                    session_id=session_id,
                    sender="ai",
                    text=partial_response
                )
                yield f"data: {json.dumps({'type': 'error', 'message': error_message}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate_events()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    # 5. Generate Response via Gemini
    ai_result = gemini_client.generate_response(
        prompt=query,
        conversation_history=history,
        context_chunks=context_chunks if context_chunks else None
    )
    logger.info(
    "Chat completed session=%s path=non_stream total_ms=%.1f",
    session_id,
    (time.perf_counter() - chat_started) * 1000
    )

    response_text = ai_result.get("response", "I could not generate an answer at this moment.")
    citations = ai_result.get("citations", [])

    # Record AI message in session
    session_manager.add_message(
        session_id=session_id,
        sender="ai",
        text=response_text,
        citations=citations
    )

    return jsonify({
        "success": ai_result.get("success", False),
        "response": response_text,
        "citations": citations,
        "has_rag_context": bool(context_chunks),
        "session_id": session_id
    })

# --- PDF Upload & RAG Document Management ---

@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    """
    Handles PDF document uploads, validation, PyMuPDF extraction,
    chunking, and ChromaDB vector indexing.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request."}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id", "").strip()

    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not PDFProcessor.is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Only PDF documents (.pdf) are supported."
        }), 400

    session = session_manager.get_session(session_id)
    session_id = session["id"]

    safe_name = secure_filename(file.filename)
    if not safe_name:
        safe_name = "career_document.pdf"

    save_path = Config.UPLOAD_FOLDER / f"{session_id}_{safe_name}"

    try:
        file.save(str(save_path))
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return jsonify({"success": False, "error": f"Failed to save file: {str(e)}"}), 500

    # Process and Validate PDF
    try:
        # Extract text per page with PyMuPDF and validate domain/scanned
        pages_data = PDFProcessor.extract_text_by_pages(save_path)
        # Create semantic chunks with metadata
        chunks = PDFProcessor.chunk_document(pages_data, filename=safe_name)

        if not chunks:
            save_path.unlink(missing_ok=True)
            return jsonify({
                "success": False,
                "error": "No readable text content could be extracted from this PDF."
            }), 400

        # Index chunks in ChromaDB vector store
        indexed_count = rag_service.index_document(
            session_id=session_id,
            filename=safe_name,
            chunks=chunks
        )

        # Bind document to this conversation session
        session_manager.set_session_document(session_id, safe_name)

        # Post system confirmation message in chat
        confirmation_msg = (
            f"📄 **Document Attached**: `{safe_name}`\n\n"
            f"- **Pages Processed**: {len(pages_data)}\n"
            f"- **Knowledge Chunks Indexed**: {indexed_count}\n\n"
            f"You can now ask questions specifically about this document, such as:\n"
            f"- *'Summarize the key qualifications and technical skills in this document.'*\n"
            f"- *'What areas of improvement do you suggest for this resume?'*\n"
            f"- *'Generate 5 interview questions based on my background.'*"
        )
        session_manager.add_message(session_id, sender="ai", text=confirmation_msg)

        return jsonify({
            "success": True,
            "filename": safe_name,
            "pages_count": len(pages_data),
            "chunks_count": indexed_count,
            "session_id": session_id,
            "confirmation_message": confirmation_msg
        })

    except PDFProcessingError as pe:
        # Clean up temporary uploaded file if validation failed
        save_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": str(pe)}), 400
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal processing error: {str(e)}"}), 500

@app.route("/api/remove-pdf", methods=["POST"])
def remove_pdf():
    """Detaches active PDF document from session and clears its RAG vectors."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()

    if not session_id:
        return jsonify({"success": False, "error": "session_id required."}), 400

    current_doc = session_manager.get_session_document(session_id)
    rag_service.delete_session_documents(session_id)
    session_manager.set_session_document(session_id, None)

    msg = f"Detached `{current_doc or 'document'}`. Subsequent questions will use general career guidance knowledge."
    session_manager.add_message(session_id, sender="ai", text=msg)

    return jsonify({"success": True, "message": msg})

# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles file uploads exceeding max allowed size."""
    return jsonify({
        "success": False,
        "error": f"File is too large. Maximum allowed size is {Config.MAX_UPLOAD_SIZE_MB} MB."
    }), 413

@app.errorhandler(404)
def not_found_error(error):
    """Handles 404 routes."""
    return jsonify({"success": False, "error": "Endpoint not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handles unexpected server errors."""
    return jsonify({"success": False, "error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    logger.info(f"Starting CareerMate on http://{Config.HOST}:{Config.PORT} (Debug={Config.DEBUG})")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
