import requests
import gradio as gr

API_BASE_URL = "http://127.0.0.1:8000"


def ask_question(question):

    if not question.strip():

        return "Please enter a question.", ""

    try:

        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json={"question": question, "top_k": 5},
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("answer", "No answer returned.")

        sources = data.get("sources", [])

        source_text = ""

        for index, source in enumerate(sources, start=1):

            metadata = source.get("metadata", {})

            filename = metadata.get("filename", "Unknown")

            page = metadata.get("page", "Unknown")

            score = source.get("score", 0)

            source_text += (
                f"**{index}. {filename}** — "
                f"Page {page} "
                f"(similarity: {score:.2f})\n\n"
            )

            source_text += f"> {source.get('text', '')[:500]}...\n\n"

        if not source_text:

            source_text = "No sources found."

        return answer, source_text

    except requests.RequestException as e:

        return (f"Unable to connect to backend: {e}", "")


def upload_document(file):

    if file is None:

        return "Please select a PDF."

    try:

        with open(file, "rb") as f:

            response = requests.post(
                f"{API_BASE_URL}/documents/upload", files={"file": f}, timeout=120
            )

        if response.status_code == 409:

            return "⚠️ This document is already indexed."

        response.raise_for_status()

        data = response.json()

        return (
            f"✅ **{data['filename']}** uploaded successfully.\n\n"
            f"Pages: {data['page_count']}\n\n"
            f"Chunks: {data['chunk_count']}"
        )

    except requests.RequestException as e:

        return f"❌ Upload failed: {e}"


def get_documents():

    try:

        response = requests.get(f"{API_BASE_URL}/documents/", timeout=30)

        response.raise_for_status()

        data = response.json()

        documents = data.get("documents", [])

        if not documents:

            return "No documents indexed."

        result = ""

        for document in documents:

            result += (
                f"- **{document['filename']}** "
                f"— {document['page_count']} pages, "
                f"{document['chunk_count']} chunks\n"
            )

        return result

    except requests.RequestException as e:

        return f"Unable to load documents: {e}"


with gr.Blocks(title="Intelligent Document Q&A") as demo:

    gr.Markdown("""
        # 📚 Intelligent Document Q&A Assistant

        Ask questions about your uploaded documents.
        """)

    gr.Markdown("## 📄 Upload Document")
    file_upload = gr.File(label="Select PDF", file_types=[".pdf"], type="filepath")

    upload_button = gr.Button("Upload Document")
    upload_status = gr.Markdown()

    upload_button.click(fn=upload_document, inputs=file_upload, outputs=upload_status)

    question = gr.Textbox(label="Question", placeholder=("e.g. What is AWS Lambda?"))

    ask_button = gr.Button("Ask", variant="primary")

    answer = gr.Markdown(label="Answer")

    sources = gr.Markdown(label="Sources")

    ask_button.click(fn=ask_question, inputs=question, outputs=[answer, sources])

    gr.Markdown("## 📚 Indexed Documents")

    document_list = gr.Markdown("Loading documents...")

    refresh_button = gr.Button("Refresh Documents")

    refresh_button.click(fn=get_documents, outputs=document_list)


if __name__ == "__main__":

    demo.launch()
