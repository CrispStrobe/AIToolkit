# 🌍 CrispTranslator - translate documents, preserve formatting

This **Document Translator** is a multi-backend translation engine that leverages Neural Machine Translation (NMT), optionally Large Language Models (LLMs), and neural word alignment to provide document translation.

Unlike standard translators that strip formatting, this system decomposes Word documents into their constituent elements, translates the text content, and meticulously reconstructs the original styling in the target language.

## ⚖️ License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

---

## 🛠 Project Structure

The system is divided into two primary components:

1. **`translator.py`**: The core translation engine. It handles document parsing, NMT/LLM routing, word alignment, and XML-level document reconstruction.
2. **`translator-app.py`**: A Gradio-based web interface designed for easy deployment (including Hugging Face Spaces). It provides a user-friendly GUI for uploading files, selecting models, and viewing real-time translation logs.

---

## 🏗 Core Engine Logic (`translator.py`)

The engine operates on a "Extract-Translate-Align-Reconstruct" lifecycle.

### 1. Document Parsing & Formatting Retention

The system uses `python-docx` to navigate the internal XML structure of a `.docx` file.

* **Font Hierarchy Resolution**: To prevent the document from reverting to default Word themes, the engine resolves the "resolved font" for every paragraph by checking individual runs and style hierarchies.
* **Run Extraction**: Text is extracted as `FormatRun` objects, which capture text alongside specific metadata: bold, italic, underline, font name, size, and RGB color.
* **Structural Elements**: The engine aggregates paragraphs from the main body, tables, headers, footers, and footnotes into a unified processing queue.

### 2. Neural Machine Translation (NMT) Backends

The engine supports multiple local and cloud-based translation backends, mostly optimized via **CTranslate2 (CT2)** for high-speed inference on both CPU (including Apple Silicon M1/M2/M3) and NVIDIA GPUs.

| Model | Source | Description |
| --- | --- | --- |
| **NLLB-200** | Meta | 200+ languages; best speed/RAM ratio. |
| **Madlad-400** | Google | 3B parameter model; superior for academic and formal texts. |
| **WMT21** | Facebook | Dense, high-quality models optimized for European languages. |
| **Opus-MT** | Helsinki-NLP | Lightweight, bilingual models for specific language pairs. |
| **LLM** | e.g. OpenAI/Claude | Might be used for natural, fluent translations and hybrid enhancement. |

### 3. Neural Word Alignment

To preserve inline formatting (e.g., if only the third word in a sentence is **bold**), the system must know where that word moved to in the translated sentence.

* **Awesome-Align (BERT)**: The system utilizes a CT2-optimized BERT model to extract high-precision word alignments using **Mutual Argmax (Intersection)** logic.
* **Fallback Chain**: If the primary neural aligner fails, the system automatically falls back to **SimAlign**, and can also leverage the **Lindat API**, or **fast_align** (which will only work tolerably well if additional snapshots are provided, though).

### 4. Footnote Handling & XML Commitment

Footnotes are notoriously difficult to translate because they rely on internal XML anchors (`w:footnoteReference`).

* The engine extracts these anchors before translation.
* It re-attaches them to the translated text.
* The final step involves a robust **XML Commitment**, where the updated footnote tree is serialized back into the document's binary blob.

---

## 🖥 The Web Interface (`translator-app.py`)

The application provides a robust wrapper around the engine, specifically optimized for high-latency or long-running tasks.

### Features:

* **Asynchronous Processing**: Uses `asyncio` to prevent the UI from freezing during long translations.
* **Thread Isolation**: To avoid conflicts with Gradio's internal event loop (preventing "Invalid file descriptor" errors), the translation is offloaded to a `ThreadPoolExecutor` with a fresh event loop.
* **Real-time Logging**: Captures standard Python logging output and streams it into a UI Textbox so users can monitor progress.
* **Environment Auto-Setup**: On startup, the app checks for required libraries (CTranslate2, etc) and API keys, providing a system status report directly in the GUI.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* `pip install python-docx torch ctranslate2 transformers gradio huggingface_hub tqdm lxml simalign`
* (Optional) `fast_align` binary in your PATH for legacy alignment support.

### Running the App

```bash
python translator-app.py

```

Access the interface at `http://localhost:7860`.

### Running via CLI

```bash
python translator.py input.docx output.docx -s en -t de --nmt nllb --mode hybrid

```

---

## 📖 Translation Strategy (Modes)

1. **NMT Only**: Uses local neural models (NLLB/Madlad) for maximum privacy and speed.
2. **LLM Plain**: High-quality natural language translation via API; loses granular word-level formatting.
3. **LLM with Alignment** (experimental): Uses an LLM for translation but applies local neural alignment to restore formatting.
4. **Hybrid (Recommended)**: Routes text through NMT engines first, using Aligners to optimize formatting.