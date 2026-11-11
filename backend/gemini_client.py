
"""
CareerMate - Google Gemini API Client
Handles Gemini AI responses and document embeddings.
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional

from backend.config import Config

logger = logging.getLogger(__name__)


CAREERMATE_SYSTEM_INSTRUCTION = f"""
You are CareerMate, an intelligent, professional,
and empathetic AI career guidance assistant.

OBJECTIVE & DOMAIN RESTRICTION:

- Answer questions related to careers, education,
  professional development, job preparation,
  technical skills, resume building, interview preparation,
  Data Science, Artificial Intelligence, Machine Learning,
  software development and IT careers.

- If a user asks an unrelated question, politely reply:
  "{Config.DOMAIN_REJECTION_MESSAGE}"

- Only introduce yourself as CareerMate when the user greets you
    or asks who you are.
- For every other question, answer the question immediately. Do not add
    a greeting, self-introduction, repeat the question, or finish with a
    generic invitation to ask more questions.
- Give a direct, concise answer by default. Provide a detailed explanation
    with useful context and examples when the user explicitly asks for detail.

WHEN RETRIEVED DOCUMENT CONTEXT IS PROVIDED:

- Use the uploaded document context to answer questions.
- Always cite the source document and page number.
- Do not invent information that is not present
  in the uploaded document.

RESPONSE STYLE:

- Professional and supportive.
- Use Markdown formatting for clear headings, bold emphasis, and ordered or unordered lists.
- Keep formatting readable and use headings or examples only when they clarify the answer.
"""


class GeminiClient:

    fallback_model_name = "gemini-3.6-flash"
    request_timeout_ms = 45_000
    stream_restart_token = object()

    @staticmethod
    def _response_token_limit(prompt: str) -> int:
        """Use a smaller output budget by default, expanding for detail requests."""
        detail_requested = re.search(
            r"\b(detail|detailed|explain|elaborate|in depth|step by step)\b",
            prompt,
            re.IGNORECASE
        )
        brevity_requested = re.search(
            r"\b(brief|briefly|short|concise|summary)\b",
            prompt,
            re.IGNORECASE
        )
        return 512 if detail_requested and not brevity_requested else 192

    @staticmethod
    def _normalize_response(text: str) -> str:
        """Trim a response without altering Markdown formatting."""
        return text.strip()

    @staticmethod
    def stream_error_message(error: Exception) -> str:
        """Return a useful, credential-safe explanation for a failed Gemini stream."""
        error_text = str(error).lower()
        if "429" in error_text or "resource_exhausted" in error_text or "quota" in error_text:
            return "Gemini is rate-limited right now. Please wait a moment and try again."
        if "503" in error_text or "unavailable" in error_text:
            return "I couldn't finish that reply. Please send your question again."
        if (
            "504" in error_text
            or "deadline_exceeded" in error_text
            or "timeout" in error_text
            or "timed out" in error_text
        ):
            return "I couldn't finish that reply. Please send your question again."
        if "403" in error_text or "permission" in error_text:
            return "Gemini API access was denied. Check the API key and its permissions."
        if "404" in error_text or "not found" in error_text:
            return "The configured Gemini model is unavailable. Check the model setting."
        if "api is not configured" in error_text or "client is unavailable" in error_text:
            return "Gemini is not configured. Check the API key and SDK setup."
        return "Gemini could not complete the request. Check the server log for the error and try again."

    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or Config.GEMINI_API_KEY

        self.model_name = Config.GEMINI_MODEL

        self.embedding_model_name = (
            Config.GEMINI_EMBEDDING_MODEL
        )

        self._client = None
        self._legacy_genai = None

        self._init_client()

    # ----------------------------------
    # INITIALIZE GEMINI CLIENT
    # ----------------------------------

    def _init_client(self):

        if not self.api_key or self.api_key == "your_gemini_api_key_here":

            logger.warning("Gemini API key is not configured.")

            return

        try:

            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=self.request_timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
            )

            logger.info(
                "Google GenAI SDK initialized successfully."
            )

            return

        except ImportError:

            logger.warning(
                "google-genai SDK not found. "
                "Trying legacy SDK."
            )

        try:

            import google.generativeai as legacy_genai

            legacy_genai.configure(
                api_key=self.api_key
            )

            self._legacy_genai = legacy_genai

            logger.info(
                "Legacy Gemini SDK initialized."
            )

        except ImportError:

            logger.error(
                "Google Gemini SDK is not installed."
            )

    # ----------------------------------
    # CHECK API CONFIGURATION
    # ----------------------------------

    def is_configured(self) -> bool:

        return bool(
            self.api_key
            and self.api_key != "your_gemini_api_key_here"
            and (self._client or self._legacy_genai)
        )

    def _with_transient_retries(self, operation):

        max_attempts = 2

        for attempt in range(max_attempts):

            try:

                return operation()

            except Exception as error:

                error_text = str(error).lower()

                if "404" in error_text or "not found" in error_text:
                    raise

                retryable = any(
                    marker in error_text
                    for marker in (
                        "503",
                        "unavailable",
                        "429",
                        "resource_exhausted",
                        "504",
                        "deadline_exceeded",
                        "deadline expired"
                    )
                )

                if not retryable or attempt == max_attempts - 1:
                    raise

                delay = 2 ** attempt

                logger.warning(
                    "Temporary Gemini API error. Retrying in %s second(s) (%s/%s): %s",
                    delay,
                    attempt + 1,
                    max_attempts - 1,
                    error
                )

                time.sleep(delay)

    def _with_overload_fallback(self, operation, fallback_operation):

        try:

            return self._with_transient_retries(operation)

        except Exception as error:

            error_text = str(error).lower()

            if (
                self.model_name == self.fallback_model_name
                or not ("503" in error_text or "unavailable" in error_text)
            ):
                raise

            logger.warning(
                "Model %s remains overloaded; retrying with fallback model %s.",
                self.model_name,
                self.fallback_model_name
            )

            return fallback_operation()

    # ----------------------------------
    # GENERATE AI RESPONSE
    # ----------------------------------

    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:

        if not self.is_configured():

            return {
                "success": False,
                "error": "Gemini API key or SDK is missing.",
                "response": (
                    "Gemini API is not configured. "
                    "Please check your .env file and dependencies."
                )
            }

        augmented_prompt = prompt

        citations = []

        # ----------------------------------
        # ADD DOCUMENT CONTEXT
        # ----------------------------------

        if context_chunks:

            context_text_blocks = []

            for i, chunk in enumerate(context_chunks, 1):

                src = chunk.get(
                    "source",
                    "Uploaded Document"
                )

                pg = chunk.get("page", 1)

                text = chunk.get("text", "")

                citations.append(
                    f"{src} (Page {pg})"
                )

                context_text_blocks.append(
                    f"Context Block {i}\n"
                    f"File: {src}\n"
                    f"Page: {pg}\n"
                    f"{text}"
                )

            combined_context = "\n\n".join(
                context_text_blocks
            )

            augmented_prompt = f"""
RETRIEVED DOCUMENT CONTEXT:

{combined_context}

USER QUESTION:

{prompt}

INSTRUCTIONS:

1. Answer using the document context.
2. Mention source filename and page number.
3. If information is missing, clearly mention it.
4. Follow CareerMate guidelines.
"""

        try:

            # ----------------------------------
            # MODERN GOOGLE GENAI SDK
            # ----------------------------------

            if self._client:

                from google.genai import types

                contents = []

                # Conversation history

                if conversation_history:

                    for msg in conversation_history[-2:]:

                        role = (
                            "user"
                            if msg.get("sender") == "user"
                            else "model"
                        )

                        message_text = msg.get("text", "")

                        if message_text:

                            contents.append(
                                types.Content(
                                    role=role,
                                    parts=[
                                        types.Part.from_text(
                                            text=message_text
                                        )
                                    ]
                                )
                            )

                # Current user prompt

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=augmented_prompt
                            )
                        ]
                    )
                )

                config = types.GenerateContentConfig(

                    system_instruction=(
                        CAREERMATE_SYSTEM_INSTRUCTION
                    ),

                    temperature=1.0,

                    max_output_tokens=self._response_token_limit(prompt),

                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )

                )

                def generate_with_model(model_name):
                    return self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config
                    )

                response = self._with_overload_fallback(
                    lambda: generate_with_model(self.model_name),
                    lambda: generate_with_model(self.fallback_model_name)
                )

                answer = response.text

                if not answer:

                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return {

                    "success": True,

                    "response": self._normalize_response(answer),

                    "citations": list(set(citations))

                }

            # ----------------------------------
            # LEGACY GOOGLE GENAI SDK
            # ----------------------------------

            elif self._legacy_genai:

                model = self._legacy_genai.GenerativeModel(

                    model_name=self.model_name,

                    system_instruction=(
                        CAREERMATE_SYSTEM_INSTRUCTION
                    ),

                    generation_config={

                        "temperature": 1.0,

                        "max_output_tokens": self._response_token_limit(prompt)

                    }

                )

                history = []

                if conversation_history:

                    for msg in conversation_history[-2:]:

                        role = (
                            "user"
                            if msg.get("sender") == "user"
                            else "model"
                        )

                        message_text = msg.get("text", "")

                        if message_text:

                            history.append({

                                "role": role,

                                "parts": [message_text]

                            })

                chat = model.start_chat(
                    history=history
                )

                def send_with_model(model_name):
                    fallback_model = self._legacy_genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=CAREERMATE_SYSTEM_INSTRUCTION,
                        generation_config={
                            "temperature": 0.4,
                            "max_output_tokens": self._response_token_limit(prompt)
                        }
                    )
                    fallback_chat = fallback_model.start_chat(
                        history=history
                    )
                    return fallback_chat.send_message(augmented_prompt)

                response = self._with_overload_fallback(
                    lambda: chat.send_message(augmented_prompt),
                    lambda: send_with_model(self.fallback_model_name)
                )

                answer = response.text

                if not answer:

                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return {

                    "success": True,

                    "response": self._normalize_response(answer),

                    "citations": list(set(citations))

                }

        except Exception as e:

            logger.error(
                "Gemini API request failed. Model: %s. Error: %s",
                self.model_name,
                e,
                exc_info=True
            )

            err_str = str(e)

            if "404" in err_str or "not found" in err_str.lower():

                user_message = (
                    f"Gemini model '{self.model_name}' is unavailable. "
                    "Please check the model name and API access."
                )

            elif "403" in err_str or "permission" in err_str.lower():

                user_message = (
                    "Gemini API permission denied. "
                    "Please check your API key."
                )

            elif "429" in err_str:

                user_message = (
                    "Gemini API request limit reached. "
                    "Please try again later."
                )

            elif (
                "504" in err_str
                or "deadline_exceeded" in err_str.lower()
                or "deadline expired" in err_str.lower()
            ):

                user_message = (
                    "Gemini took too long to respond. "
                    "I retried once, but the request still timed out. "
                    "Please try again or shorten the question/document."
                )

            elif "503" in err_str or "unavailable" in err_str.lower():

                user_message = (
                    "I couldn't finish that reply. Please send your question again."
                )

            else:

                user_message = (
                    f"Gemini API error: {err_str}"
                )

            return {

                "success": False,

                "error": err_str,

                "response": self._normalize_response(user_message)

            }

        return {

            "success": False,

            "error": "Gemini client is unavailable.",

            "response": "Please check your Gemini SDK installation."

        }

    def generate_response_stream(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None
    ):
        """Yields answer text, restarting the UI if a stream fails mid-response."""
        if not self.is_configured():
            raise RuntimeError(
                "Gemini API is not configured. Please check your API key and SDK."
            )

        context_blocks = []
        if context_chunks:
            for index, chunk in enumerate(context_chunks, 1):
                context_blocks.append(
                    f"Context Block {index}\n"
                    f"File: {chunk.get('source', 'Uploaded Document')}\n"
                    f"Page: {chunk.get('page', 1)}\n"
                    f"{chunk.get('text', '')}"
                )
            augmented_prompt = (
                "RETRIEVED DOCUMENT CONTEXT:\n\n"
                + "\n\n".join(context_blocks)
                + f"\n\nUSER QUESTION:\n\n{prompt}\n\n"
                "Answer using the document context, cite its filename and page, "
                "and say clearly if the information is missing."
            )
        else:
            augmented_prompt = prompt

        history = conversation_history[-2:] if conversation_history else []
        model_names = [self.model_name]
        if self.model_name != self.fallback_model_name:
            model_names.append(self.fallback_model_name)

        modern_contents = None
        modern_config = None
        if self._client:
            from google.genai import types

            modern_contents = []
            for message in history:
                message_text = message.get("text", "")
                if message_text:
                    modern_contents.append(
                        types.Content(
                            role="user" if message.get("sender") == "user" else "model",
                            parts=[types.Part.from_text(text=message_text)]
                        )
                    )
            modern_contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=augmented_prompt)]
                )
            )
            modern_config = types.GenerateContentConfig(
                system_instruction=CAREERMATE_SYSTEM_INSTRUCTION,
                temperature=1.0,
                max_output_tokens=self._response_token_limit(prompt),
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL
                )
            )

        for model_index, model_name in enumerate(model_names):
            retry_limit = 2 if model_index == len(model_names) - 1 else 1
            for attempt_index in range(retry_limit):
                emitted_text = False
                try:
                    if self._client:
                        stream = self._client.models.generate_content_stream(
                            model=model_name,
                            contents=modern_contents,
                            config=modern_config
                        )
                    elif self._legacy_genai:
                        model = self._legacy_genai.GenerativeModel(
                            model_name=model_name,
                            system_instruction=CAREERMATE_SYSTEM_INSTRUCTION,
                            generation_config={
                                "temperature": 1.0,
                                "max_output_tokens": self._response_token_limit(prompt)
                            }
                        )
                        legacy_history = [
                            {
                                "role": "user" if message.get("sender") == "user" else "model",
                                "parts": [message["text"]]
                            }
                            for message in history
                            if message.get("text")
                        ]
                        stream = model.start_chat(
                            history=legacy_history
                        ).send_message(augmented_prompt, stream=True)
                    else:
                        raise RuntimeError("Gemini client is unavailable.")

                    for chunk in stream:
                        text = getattr(chunk, "text", None)
                        if text:
                            emitted_text = True
                            yield text

                    if not emitted_text:
                        raise RuntimeError("Gemini returned an empty response.")
                    return
                except Exception as error:
                    error_text = str(error).lower()
                    retryable_temporary_error = any(
                        marker in error_text
                        for marker in (
                            "503", "unavailable",
                            "504", "deadline_exceeded", "deadline expired"
                        )
                    )
                    can_fallback = retryable_temporary_error or any(
                        marker in error_text
                        for marker in (
                            "404", "not found",
                            "429", "resource_exhausted", "quota"
                        )
                    )

                    if (
                        model_index == len(model_names) - 1
                        and attempt_index + 1 < retry_limit
                        and retryable_temporary_error
                    ):
                        logger.warning(
                            "Gemini fallback model %s timed out or was unavailable; retrying once.",
                            model_name
                        )
                        if emitted_text:
                            yield self.stream_restart_token
                        continue

                    if not can_fallback or model_index == len(model_names) - 1:
                        raise

                    if emitted_text:
                        logger.warning(
                            "Gemini model %s failed mid-response; restarting with fallback model %s.",
                            model_name,
                            model_names[model_index + 1]
                        )
                        yield self.stream_restart_token
                    else:
                        logger.warning(
                            "Gemini model %s failed before streaming; trying fallback model %s.",
                            model_name,
                            model_names[model_index + 1]
                        )
                    break

    # ----------------------------------
    # GENERATE BATCH EMBEDDINGS
    # ----------------------------------

    def get_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        if not self.is_configured():

            raise ValueError(
                "Gemini API is not configured for embeddings."
            )

        if not texts:

            return []

        try:

            # Modern SDK

            if self._client:

                all_embeddings = []

                batch_size = 50

                for i in range(
                    0,
                    len(texts),
                    batch_size
                ):

                    batch = texts[
                        i:i + batch_size
                    ]

                    response = self._client.models.embed_content(

                        model=self.embedding_model_name,

                        contents=batch

                    )

                    for emb in response.embeddings:

                        all_embeddings.append(
                            emb.values
                        )

                return all_embeddings

            # Legacy SDK

            elif self._legacy_genai:

                all_embeddings = []

                for text in texts:

                    result = self._legacy_genai.embed_content(

                        model=f"models/{self.embedding_model_name}",

                        content=text,

                        task_type="retrieval_document"

                    )

                    all_embeddings.append(
                        result["embedding"]
                    )

                return all_embeddings

            raise RuntimeError(
                "No Gemini SDK client available."
            )

        except Exception as e:

            logger.error(
                "Embedding generation failed: %s",
                e,
                exc_info=True
            )

            raise

    # ----------------------------------
    # GENERATE QUERY EMBEDDING
    # ----------------------------------

    def get_query_embedding(
        self,
        query: str
    ) -> List[float]:

        embeddings = self.get_embeddings_batch(
            [query]
        )

        return embeddings[0] if embeddings else []
