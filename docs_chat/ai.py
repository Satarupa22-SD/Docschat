from typing import List
from docs_chat.db import Reference
from docs_chat.config import Config
import requests

class AIAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.backend = config.get('llm_backend', 'gemini')
        if self.backend == 'gemini':
            try:
                import google.generativeai as genai
                self.genai = genai
                self.genai.configure(api_key=config.get('gemini_api_key'))
                self.model = self.genai.GenerativeModel('gemini-2.0-flash')
            except ImportError:
                raise ImportError("google-generativeai is required for Gemini backend.")
        elif self.backend == 'ollama':
            self.ollama_url = config.get('ollama_base_url', 'http://localhost:11434')
            self.ollama_model = config.get('ollama_model', 'llama3')
            if not self._model_exists(self.ollama_model):
                available = self._list_ollama_models()
                raise ValueError(
                    f"Ollama model '{self.ollama_model}' not found on server.\n"
                    f"Available models: {', '.join(available) if available else '[none found]'}\n"
                    f"Please download your desired model with: ollama pull <model-name> and update config.yaml."
                )
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def _model_exists(self, model_name: str) -> bool:
        """Check if a model exists on the Ollama server with flexible matching"""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get('models', [])
            available_models = [m['name'].strip() for m in models]
            
            model_name = model_name.strip()
            
            # Try exact match first
            if model_name in available_models:
                return True
            
            # Try case-insensitive match
            for available in available_models:
                if model_name.lower() == available.lower():
                    return True
            
            # Try partial matching (e.g., "gemma3" matches "gemma3:4b")
            for available in available_models:
                if (model_name.lower() in available.lower() or 
                    available.lower().startswith(model_name.lower() + ':')):
                    return True
            
            return False
            
        except Exception:
            return False

    def _list_ollama_models(self):
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get('models', [])
            return [m['name'] for m in models]
        except Exception:
            return []

    def answer_question(self, question: str, context_chunks: List[Reference]) -> str:
        context = "\n\n".join([
            f"[Page {ref.page_number}, Line {ref.line_number}, Section: {ref.section}]\n{ref.content}"
            for ref in context_chunks
        ])
        prompt = f"""
        Based on the following research paper excerpts, answer the user's question. 
        Be specific and cite information when possible.
        
        Research Paper Context:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive answer based on the provided context. If the context doesn't contain enough information to answer the question, state that clearly.
        """
        if self.backend == 'gemini':
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error generating answer (Gemini): {e}"
        elif self.backend == 'ollama':
            try:
                data = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
                resp = requests.post(f"{self.ollama_url}/api/generate", json=data, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                return result.get('response', '[No response from Ollama]')
            except Exception as e:
                return f"Error generating answer (Ollama): {e}"
        else:
            return "Invalid backend configuration."

    def summarize_paper(self, context_chunks: List[Reference]) -> str:
        context = "\n\n".join([
            f"[Page {ref.page_number}, Section: {ref.section}]\n{ref.content}"
            for ref in context_chunks
        ])
        prompt = f"""
        Summarize the following research paper in detail. Highlight the main contributions, methods, and findings. Include citations to page numbers and sections where appropriate.
        
        Research Paper Content:
        {context}
        
        Please provide a comprehensive summary with inline references to the source (e.g., [Page X, Section: Y]).
        """
        if self.backend == 'gemini':
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error generating summary (Gemini): {e}"
        elif self.backend == 'ollama':
            try:
                data = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
                resp = requests.post(f"{self.ollama_url}/api/generate", json=data, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                return result.get('response', '[No response from Ollama]')
            except Exception as e:
                return f"Error generating summary (Ollama): {e}"
        else:
            return "Invalid backend configuration."