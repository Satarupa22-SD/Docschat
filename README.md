# Docs Chat

A command-line tool for analyzing research papers (PDFs) and answering questions with citations using either Google Gemini (API) or local open-source LLMs via Ollama.

## Features
- Load and index research papers (PDF)
- Semantic search for relevant content
- AI-powered answers with citations
- Choose between Gemini (API) or Ollama (local LLM)
- Interactive CLI or command-line mode
- Single-paper and multi-paper analysis modes

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure:**
   - Edit `config.yaml` to set your preferred backend:
     - `llm_backend: gemini` (default, requires API key)
     - `llm_backend: ollama` (run open-source LLMs locally, no API key needed)
   - For Gemini, set your API key in `gemini_api_key`.
   - For Ollama, ensure Ollama is running locally and set `ollama_model` as desired (e.g., `llama3`).

## Commands

### Paper Management
- `load <pdf_path>` - Load a research paper (PDF) for single-paper mode
- `load-folder <folder_path>` - Load all research papers (PDFs) from a folder for multi-paper mode
- `delete <pdf_path>` - Delete a specific loaded paper
- `delete all` - Delete all loaded papers from the database

### Analysis
- `research <query>` - Ask a question about the loaded paper (single-paper)
- `research --all <query>` - Ask a question across all loaded papers (multi-paper)
- `summarize` - Summarize the loaded paper (single-paper)
- `summarize --all` - Summarize all loaded papers (multi-paper)

### Citations and Export
- `reference` - Show detailed citations for the last query (single-paper)
- `reference --all` - Show detailed citations for the last query across all papers (multi-paper)
- `export <output_file.md>` - Export the last result (summary, research, or reference) to a Markdown file

### System
- `list-ollama-models` - List available Ollama models (if backend is 'ollama')
- `help` - Show this help message
- `exit` - Exit the program

## Usage

### Interactive Mode
```sh
python main.py --interactive
```

### Command-line Mode
- Load a paper:
  ```sh
  python main.py --load path/to/paper.pdf
  ```
- Ask a question:
  ```sh
  python main.py --research "What is the main contribution?"
  ```
- Show references for last query:
  ```sh
  python main.py --reference
  ```

## Configuration (`config.yaml`)
```
llm_backend: gemini  # or 'ollama'
gemini_api_key: your-gemini-api-key-here
ollama_base_url: http://localhost:11434
ollama_model: llama3
database_path: ./research_db
max_context_chunks: 5
```

## Requirements
- Python 3.8+
- For Ollama: [Ollama](https://ollama.com/) running locally
- For Gemini: Google Generative AI API key

## License
MIT