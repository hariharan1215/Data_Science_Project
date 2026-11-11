
async function handleSendMessage(customPrompt = null) {

    if (isGenerating) return;

    const query = (
        customPrompt || messageInput.value
    ).trim();

    if (!query) return;

    if (!currentSessionId) {
        appendMessageUI(
            "ai",
            "Please create a new conversation and try again."
        );
        return;
    }

    // Clear input
    messageInput.value = "";
    autoResizeTextarea();

    sendBtn.disabled = true;

    welcomeScreen.style.display = "none";

    // Display user message
    appendMessageUI("user", query);
    scrollToBottom();

    // Start loading
    isGenerating = true;
    showTypingIndicator(true);

    const controller = new AbortController();

    // Keep browser timeout aligned with the backend request deadline.
    let timeout = setTimeout(() => {
        controller.abort();
    }, 50000);
    const resetTimeout = () => {
        clearTimeout(timeout);
        timeout = setTimeout(() => controller.abort(), 50000);
    };
    let streamingBubble = null;
    let streamingMarkdown = "";

    try {

        // Send request to Flask backend
        const res = await fetch("/api/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                session_id: currentSessionId,

                message: query,
                stream: true

            }),

            signal: controller.signal

        });
        resetTimeout();

        const contentType = res.headers.get("content-type") || "";
        let data;

        if (contentType.includes("text/event-stream")) {
            streamingBubble = appendMessageUI("ai", "");
            const reader = res.body?.getReader();
            if (!reader) throw new Error("Streaming response is unavailable.");

            const decoder = new TextDecoder();
            let pending = "";
            let completed = false;

            const handleEvent = (frame) => {
                const dataLine = frame
                    .split("\n")
                    .find((line) => line.startsWith("data:"));
                if (!dataLine) return;

                const event = JSON.parse(dataLine.slice(5).trim());
                if (event.type === "chunk") {
                    streamingMarkdown += event.text;
                    renderAssistantMarkdown(streamingBubble, streamingMarkdown);
                    scrollToBottom();
                } else if (event.type === "done") {
                    completed = true;
                } else if (event.type === "reset") {
                    streamingMarkdown = "";
                    streamingBubble.replaceChildren();
                    scrollToBottom();
                } else if (event.type === "error") {
                    throw new Error(event.message || "Gemini could not finish the answer.");
                }
            };

            while (true) {
                const { value, done } = await reader.read();
                if (value?.length) resetTimeout();
                pending += decoder.decode(value, { stream: !done });
                pending = pending.replace(/\r\n/g, "\n");

                let boundary = pending.indexOf("\n\n");
                while (boundary !== -1) {
                    handleEvent(pending.slice(0, boundary));
                    pending = pending.slice(boundary + 2);
                    boundary = pending.indexOf("\n\n");
                }

                if (done) break;
            }

            if (pending.trim()) handleEvent(pending);
            if (!completed) throw new Error("The answer stream ended unexpectedly.");
        } else {
            // Read JSON responses used for greetings and off-topic replies.
            data = await res.json().catch(() => ({}));

            if (!res.ok || data.success === false) {
                throw new Error(
                    data.error ||
                    data.response ||
                    "Server error: " + res.status
                );
            }

            const aiResponse =
                data.response ||
                data.message ||
                data.answer;

            appendMessageUI(
                "ai",
                aiResponse || "No response received from Gemini API. Please try again.",
                data.citations || []
            );
        }

        scrollToBottom();

        // Refresh chat history without delaying the next reply.
        loadSessions().catch((sessionError) => {
            console.error(
                "Session loading error:",
                sessionError
            );
        });

    } catch (err) {

        console.error(
            "CareerMate Chat Error:",
            err
        );

        let errorMessage;

        if (err.name === "AbortError") {

            errorMessage =
                "⚠️ I couldn't finish that reply. Please send your question again.";

        } else {

            errorMessage =
                "⚠️ " + err.message;

        }

        if (streamingBubble) {
            streamingMarkdown = streamingMarkdown
                ? `${streamingMarkdown}\n\n${errorMessage}`
                : errorMessage;
            renderAssistantMarkdown(streamingBubble, streamingMarkdown);
        } else {
            appendMessageUI("ai", errorMessage);
        }

        scrollToBottom();

    } finally {

        // Stop timeout
        clearTimeout(timeout);

        // Stop loading indicator
        showTypingIndicator(false);

        // Allow next message
        isGenerating = false;

        sendBtn.disabled = !messageInput.value.trim();

    }

}

const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const chatForm = document.getElementById("chatForm");
const messagesStream = document.getElementById("messagesStream");
const welcomeScreen = document.getElementById("welcomeScreen");
const typingIndicator = document.getElementById("typingIndicator");
const uploadModal = document.getElementById("uploadModal");
const pdfFileInput = document.getElementById("pdfFileInput");
const dropZone = document.getElementById("dropZone");
const selectedFilePreview = document.getElementById("selectedFilePreview");
const uploadProgressBox = document.getElementById("uploadProgressBox");
const uploadErrorAlert = document.getElementById("uploadErrorAlert");
const uploadErrorText = document.getElementById("uploadErrorText");
const confirmUploadBtn = document.getElementById("confirmUploadBtn");
const MAX_UPLOAD_BYTES = 16 * 1024 * 1024;

let currentSessionId = null;
let isGenerating = false;
let selectedPdf = null;

function appendMessageUI(sender, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${sender === "user" ? "user" : "ai"}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = sender === "user"
        ? '<i class="fa-solid fa-user"></i>'
        : '<i class="fa-solid fa-robot"></i>';

    const content = document.createElement("div");
    content.className = "message-content-box";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (sender === "user") {
        bubble.textContent = text;
    } else {
        renderAssistantMarkdown(bubble, text);
    }

    content.appendChild(bubble);
    wrapper.append(avatar, content);
    messagesStream.appendChild(wrapper);
    return bubble;
}

function renderAssistantMarkdown(element, markdown) {
    markdown = markdown
        .replace(
            /Gemini is temporarily busy\.\s*Please try again shortly\./gi,
            "I couldn't finish that reply. Please send your question again."
        )
        .replace(
            /Gemini is temporarily experiencing high demand\.[\s\S]*?Please try again in a moment\./gi,
            "I couldn't finish that reply. Please send your question again."
        );

    if (window.marked?.parse && window.DOMPurify?.sanitize) {
        const html = window.marked.parse(markdown, {
            async: false,
            breaks: true,
            gfm: true
        });
        element.innerHTML = window.DOMPurify.sanitize(html, {
            FORBID_ATTR: ["style"],
            FORBID_TAGS: ["audio", "embed", "form", "iframe", "img", "object", "video"],
            USE_PROFILES: { html: true }
        });
        return;
    }

    // Keep responses safe and free of visible Markdown markers if a CDN is unavailable.
    element.textContent = markdown.replace(/\*/g, "");
}

function scrollToBottom() {
    const chatContainer = document.getElementById("chatContainer");
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showTypingIndicator(show) {
    typingIndicator.style.display = show ? "flex" : "none";
    if (show) scrollToBottom();
}

function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
}

async function createSession() {
    const response = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Career Chat" })
    });
    const data = await response.json();
    if (!response.ok || !data.session?.id) {
        throw new Error(data.error || "Could not start a conversation.");
    }
    currentSessionId = data.session.id;
    document.getElementById("currentChatTitle").textContent = data.session.title;
    return data.session;
}

async function ensureSession() {
    if (!currentSessionId) await createSession();
}

async function loadSessions() {
    const response = await fetch("/api/sessions");
    const data = await response.json();
    if (!response.ok) return;

    const historyList = document.getElementById("historyList");
    historyList.replaceChildren();
    const sessions = data.sessions || [];

    if (!sessions.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = "No previous conversations";
        historyList.appendChild(empty);
        return;
    }

    sessions.forEach((session) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-item";
        button.textContent = session.title || "New Career Chat";
        button.addEventListener("click", () => openSession(session.id));
        historyList.appendChild(button);
    });
}

async function openSession(sessionId) {
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    const data = await response.json();
    if (!response.ok || !data.session) return;

    const session = data.session;
    currentSessionId = session.id;
    document.getElementById("currentChatTitle").textContent = session.title;
    messagesStream.replaceChildren();
    welcomeScreen.style.display = session.messages.length ? "none" : "flex";
    session.messages.forEach((message) => appendMessageUI(message.sender, message.text));
    setActiveDocument(session.uploaded_document);
    scrollToBottom();
}

function setActiveDocument(filename) {
    const active = Boolean(filename);
    document.getElementById("activeDocSection").style.display = active ? "block" : "none";
    document.getElementById("sidebarDocName").textContent = filename || "";
    document.getElementById("pdfContextBar").style.display = active ? "flex" : "none";
    document.getElementById("contextBarDocName").textContent = filename || "";
    document.getElementById("inputAttachedDoc").style.display = active ? "flex" : "none";
    document.getElementById("attachedFileName").textContent = filename || "";
}

function showUploadError(message) {
    uploadErrorText.textContent = message;
    uploadErrorAlert.style.display = "flex";
}

function resetUploadDialog() {
    selectedPdf = null;
    pdfFileInput.value = "";
    selectedFilePreview.style.display = "none";
    dropZone.querySelector(".drop-zone-content").style.display = "";
    uploadProgressBox.style.display = "none";
    uploadErrorAlert.style.display = "none";
    confirmUploadBtn.disabled = true;
}

function selectPdf(file) {
    resetUploadDialog();
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf") || file.type && file.type !== "application/pdf") {
        showUploadError("PDF files only. Please choose a .pdf document.");
        return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
        showUploadError("File is too large. Maximum allowed size is 16 MB.");
        return;
    }

    selectedPdf = file;
    document.getElementById("previewFilename").textContent = file.name;
    document.getElementById("previewFilesize").textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    dropZone.querySelector(".drop-zone-content").style.display = "none";
    selectedFilePreview.style.display = "flex";
    confirmUploadBtn.disabled = false;
}

async function uploadSelectedPdf() {
    if (!selectedPdf || confirmUploadBtn.disabled) return;

    confirmUploadBtn.disabled = true;
    uploadErrorAlert.style.display = "none";
    uploadProgressBox.style.display = "block";
    document.getElementById("progressStatus").textContent = "Uploading and indexing PDF...";

    try {
        await ensureSession();
        const formData = new FormData();
        formData.append("file", selectedPdf);
        formData.append("session_id", currentSessionId);

        const response = await fetch("/api/upload-pdf", {
            method: "POST",
            body: formData
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.success) {
            throw new Error(data.error || `Upload failed (${response.status}).`);
        }

        setActiveDocument(data.filename);
        welcomeScreen.style.display = "none";
        appendMessageUI("ai", data.confirmation_message || `PDF uploaded: ${data.filename}. You can now ask a question about it.`);
        uploadModal.style.display = "none";
        resetUploadDialog();
        await loadSessions();
        scrollToBottom();
    } catch (error) {
        uploadProgressBox.style.display = "none";
        showUploadError(error.message || "Could not upload this PDF.");
        confirmUploadBtn.disabled = false;
    }
}

async function detachDocument() {
    if (!currentSessionId) return;
    const response = await fetch("/api/remove-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId })
    });
    const data = await response.json().catch(() => ({}));
    if (response.ok && data.success) {
        setActiveDocument(null);
        appendMessageUI("ai", data.message || "Document detached.");
        scrollToBottom();
        await loadSessions();
    }
}

document.getElementById("openUploadModalBtn").addEventListener("click", async () => {
    try {
        await ensureSession();
        resetUploadDialog();
        uploadModal.style.display = "flex";
    } catch (error) {
        showUploadError(error.message);
        uploadModal.style.display = "flex";
    }
});

document.getElementById("inputUploadBtn").addEventListener("click", async () => {
    try {
        await ensureSession();
        resetUploadDialog();
        uploadModal.style.display = "flex";
    } catch (error) {
        appendMessageUI("ai", error.message);
    }
});

document.getElementById("browsePdfBtn").addEventListener("click", () => pdfFileInput.click());
pdfFileInput.addEventListener("change", () => selectPdf(pdfFileInput.files[0]));
document.getElementById("removeSelectedFileBtn").addEventListener("click", resetUploadDialog);
document.getElementById("confirmUploadBtn").addEventListener("click", uploadSelectedPdf);
document.getElementById("closeUploadModalBtn").addEventListener("click", () => {
    uploadModal.style.display = "none";
    resetUploadDialog();
});
document.getElementById("cancelUploadBtn").addEventListener("click", () => {
    uploadModal.style.display = "none";
    resetUploadDialog();
});
document.getElementById("detachDocBtn").addEventListener("click", detachDocument);
document.getElementById("contextBarRemoveBtn").addEventListener("click", detachDocument);
document.getElementById("removeAttachmentBtn").addEventListener("click", detachDocument);

dropZone.addEventListener("click", (event) => {
    if (event.target.closest("button") || selectedPdf) return;
    pdfFileInput.click();
});
dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    selectPdf(event.dataTransfer.files[0]);
});

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
        await ensureSession();
        await handleSendMessage();
    } catch (error) {
        appendMessageUI("ai", error.message || "Could not start a conversation.");
    }
});

messageInput.addEventListener("input", () => {
    autoResizeTextarea();
    sendBtn.disabled = !messageInput.value.trim() || isGenerating;
});

document.getElementById("newChatBtn").addEventListener("click", async () => {
    try {
        await createSession();
        messagesStream.replaceChildren();
        welcomeScreen.style.display = "flex";
        setActiveDocument(null);
        await loadSessions();
    } catch (error) {
        appendMessageUI("ai", error.message || "Could not start a new conversation.");
    }
});

document.querySelectorAll(".suggestion-chip").forEach((button) => {
    button.addEventListener("click", async () => {
        try {
            await ensureSession();
            await handleSendMessage(button.dataset.prompt);
        } catch (error) {
            appendMessageUI("ai", error.message || "Could not start a conversation.");
        }
    });
});

createSession()
    .then(loadSessions)
    .catch((error) => appendMessageUI("ai", error.message || "Could not connect to CareerMate."));
