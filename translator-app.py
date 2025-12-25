#!/usr/bin/env python3
"""
Gradio interface for Document Translator
Designed for Hugging Face Spaces deployment
"""

import gradio as gr
import asyncio
import os
import sys
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from translator.py
from translator import (
    UltimateDocumentTranslator,
    TranslationMode,
    TranslationBackend,
    AlignerBackend
)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def check_and_setup_environment():
    """
    Verify and setup required tools for Hugging Face Spaces.
    Returns status messages.
    """
    status_messages = []
    
    # 1. Check CTranslate2
    try:
        import ctranslate2
        status_messages.append("✓ CTranslate2 installed")
    except ImportError:
        status_messages.append("⚠ CTranslate2 not found - installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ctranslate2"], check=True)
            status_messages.append("✓ CTranslate2 installed successfully")
        except Exception as e:
            status_messages.append(f"✗ CTranslate2 installation failed: {e}")
    
    # 2. Check fast_align (optional, complex to build on HF Spaces)
    fast_align_path = shutil.which("fast_align")
    if fast_align_path:
        status_messages.append(f"✓ fast_align found at {fast_align_path}")
    else:
        status_messages.append("ℹ fast_align not available (optional - will use other aligners)")
        # Note: Building fast_align on HF Spaces is challenging due to build dependencies
        # We'll rely on the Python-based aligners instead
    
    # 3. Check for API keys (optional)
    if os.getenv("OPENAI_API_KEY"):
        status_messages.append("✓ OpenAI API key detected")
    if os.getenv("ANTHROPIC_API_KEY"):
        status_messages.append("✓ Anthropic API key detected")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        status_messages.append("ℹ No LLM API keys found (LLM modes will be unavailable)")
    
    return "\n".join(status_messages)

# Run setup on startup
SETUP_STATUS = check_and_setup_environment()
logger.info(f"Setup complete:\n{SETUP_STATUS}")

# ============================================================================
# TRANSLATION FUNCTION
# ============================================================================

async def translate_document_async(
    input_file,
    source_lang: str,
    target_lang: str,
    mode: str,
    nmt_backend: str,
    nllb_size: str,
    aligner: str,
    llm_provider: Optional[str],
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """
    Asynchronous document translation with progress tracking.
    
    Returns:
        Tuple of (output_file_path, log_messages)
    """
    
    if input_file is None:
        return None, "❌ Error: No file uploaded"
    
    # Create temp directory for processing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Setup paths
        input_path = Path(input_file.name)
        output_filename = f"{input_path.stem}_translated_{source_lang}_{target_lang}.docx"
        output_path = temp_dir / output_filename
        
        # Validate file type
        if not input_path.suffix.lower() == '.docx':
            return None, "❌ Error: Only .docx files are supported"
        
        # Map UI selections to enums
        mode_map = {
            'NMT Only': TranslationMode.NMT_ONLY,
            'LLM with Alignment': TranslationMode.LLM_WITH_ALIGN,
            'LLM without Alignment': TranslationMode.LLM_WITHOUT_ALIGN,
            'Hybrid (Recommended)': TranslationMode.HYBRID
        }
        
        # Setup logging capture
        log_messages = []
        
        class LogCapture(logging.Handler):
            def emit(self, record):
                log_messages.append(self.format(record))
        
        log_handler = LogCapture()
        log_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        
        progress(0.1, desc="Initializing translator...")
        
        # Initialize translator
        translator = UltimateDocumentTranslator(
            src_lang=source_lang,
            tgt_lang=target_lang,
            mode=mode_map[mode],
            nmt_backend=nmt_backend.lower() if nmt_backend != "Auto" else "auto",
            llm_provider=llm_provider.lower() if llm_provider and llm_provider != "None" else None,
            aligner=aligner.lower() if aligner != "Auto" else "auto",
            nllb_model_size=nllb_size
        )
        
        progress(0.2, desc="Processing document...")
        
        # Translate
        await translator.translate_document(input_path, output_path)
        
        progress(1.0, desc="Translation complete!")
        
        # Cleanup log handler
        logging.getLogger().removeHandler(log_handler)
        
        # Format log output
        log_output = "\n".join(log_messages[-50:])  # Last 50 messages
        success_msg = f"""
✅ Translation Complete!

📄 Input: {input_path.name}
📄 Output: {output_filename}
🌍 Direction: {source_lang.upper()} → {target_lang.upper()}
⚙️ Mode: {mode}
🔧 Backend: {nmt_backend}

Recent Logs:
{log_output}
"""
        
        return str(output_path), success_msg
        
    except Exception as e:
        error_msg = f"❌ Translation Error:\n{str(e)}\n\nPlease check your settings and try again."
        logger.error(f"Translation failed: {e}", exc_info=True)
        return None, error_msg

def translate_document_sync(*args, **kwargs):
    """Synchronous wrapper with explicit event loop management"""
    import asyncio
    import concurrent.futures
    from datetime import datetime
    
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 translate_document_sync called")
    
    # Define a helper to run the async function in a new thread's loop
    def run_in_new_loop(func, *args, **kwargs):
        """Create a fresh event loop in the new thread"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(func(*args, **kwargs))
        finally:
            new_loop.close()
    
    try:
        try:
            # Check if a loop is already running (Gradio/HF Spaces context)
            asyncio.get_running_loop()
            print(f"[DEBUG] ⚠️  Event loop running - Offloading to ThreadPool")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Pass the FUNCTION and ARGS separately
                future = executor.submit(
                    run_in_new_loop, 
                    translate_document_async, 
                    *args, 
                    **kwargs
                )
                result = future.result(timeout=600)
                print(f"[DEBUG] ✓ Thread execution completed")
                return result
                
        except RuntimeError:
            # No loop running (Standalone context)
            print(f"[DEBUG] ℹ️  No running loop - Using standard asyncio.run")
            result = asyncio.run(translate_document_async(*args, **kwargs))
            print(f"[DEBUG] ✓ asyncio.run() completed")
            return result
            
    except concurrent.futures.TimeoutError:
        error_msg = "❌ Translation timeout (>10 minutes)"
        print(f"[ERROR] {error_msg}")
        return None, error_msg
        
    except Exception as e:
        print(f"[ERROR] Critical failure: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"
        
    finally:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏁 Finished")
        print(f"{'='*60}\n")

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the Gradio interface"""
    
    # Language options (common pairs)
    languages = {
        "English": "en",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Hindi": "hi",
        "Dutch": "nl",
        "Polish": "pl",
        "Turkish": "tr",
        "Czech": "cs",
        "Ukrainian": "uk",
        "Vietnamese": "vi"
    }
    
    with gr.Blocks(title="Document Translator") as demo:  # REMOVED theme parameter
        gr.Markdown("""
        # 🌍 Document Translator
        
        Translate Word documents while preserving formatting, footnotes, and styling.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                input_file = gr.File(
                    label="Upload Document (.docx)",
                    file_types=[".docx"],
                    type="filepath"
                )
                
                with gr.Row():
                    source_lang = gr.Dropdown(
                        choices=list(languages.keys()),
                        value="English",
                        label="Source Language"
                    )
                    target_lang = gr.Dropdown(
                        choices=list(languages.keys()),
                        value="German",
                        label="Target Language"
                    )
                
                gr.Markdown("### ⚙️ Settings")
                
                mode = gr.Dropdown(
                    choices=[
                        "Hybrid (Recommended)",
                        "NMT Only",
                        "LLM with Alignment",
                        "LLM without Alignment"
                    ],
                    value="Hybrid (Recommended)",
                    label="Translation Mode",
                    info="Hybrid uses NMT with optional LLM enhancement"
                )
                
                nmt_backend = gr.Dropdown(
                    choices=["NLLB", "Madlad", "Opus", "CT2", "Auto"],
                    value="NLLB",
                    label="NMT Backend",
                    info="NLLB: Fast & balanced | Madlad: Academic | Opus: Specialized pairs"
                )
                
                nllb_size = gr.Dropdown(
                    choices=["600M", "1.3B", "3.3B"],
                    value="600M",
                    label="NLLB Model Size",
                    info="600M recommended for Hugging Face Spaces (limited RAM)"
                )
                
                aligner = gr.Dropdown(
                    choices=["Auto", "Awesome", "SimAlign", "Lindat", "Heuristic"],
                    value="Auto",
                    label="Word Aligner",
                    info="Auto will select best available aligner"
                )
                
                llm_provider = gr.Dropdown(
                    choices=["None", "OpenAI", "Anthropic", "Ollama"],
                    value="None",
                    label="LLM Provider (Optional)",
                    info="Requires API key in environment variables"
                )
                
                translate_btn = gr.Button("🚀 Translate Document", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Output")
                
                output_file = gr.File(
                    label="Translated Document",
                    interactive=False
                )
                
                log_output = gr.Textbox(
                    label="Translation Log",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
        
        gr.Markdown(f"### System Status\n```\n{SETUP_STATUS}\n```")
        
        gr.Markdown("""
        **Features:**
        - Multiple neural translation backends (NLLB, Madlad, Opus-MT, WMT21)
        - Word-level alignment for format preservation
        - Support for footnotes, tables, headers/footers
        - Optional LLM enhancement (OpenAI/Anthropic)
        
        **Recommended Settings:**
        - Mode: Hybrid (best quality)
        - Backend: NLLB (fastest, good quality)
        - Size: 600M (good balance)
                    
        ### 📖 Tips
        
        - **For best quality**: Use "Hybrid" mode with NLLB backend
        - **For speed**: Use "NMT Only" with NLLB 600M
        - **For academic texts**: Try Madlad backend
        - **For specific language pairs**: Opus-MT (if available)
        - **LLM modes**: Require API keys set as environment variables
        
        ### ⚠️ Limitations
        
        - Only .docx format supported (not .doc)
        - Large documents may take several minutes
        - Complex formatting may require manual review
        - LLM modes are slower and require API access
        """)
        
        def handle_translate(input_f, src_lang_name, tgt_lang_name, mode, nmt, nllb_sz, algn, llm):
            src_code = languages.get(src_lang_name, "en")
            tgt_code = languages.get(tgt_lang_name, "de")
            return translate_document_sync(input_f, src_code, tgt_code, mode, nmt, nllb_sz, algn, llm)
        
        translate_btn.click(
            fn=handle_translate,
            inputs=[
                input_file,
                source_lang,
                target_lang,
                mode,
                nmt_backend,
                nllb_size,
                aligner,
                llm_provider
            ],
            outputs=[output_file, log_output]
        )
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()  
    )