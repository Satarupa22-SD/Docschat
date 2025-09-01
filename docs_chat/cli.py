import sys
from pathlib import Path
from docs_chat.config import Config
from docs_chat.db import ResearchDatabase, Reference
from docs_chat.ai import AIAssistant

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
except ImportError:
    Console = None

HELP_TEXT = """
[bold green]Docs Chat System - Help[/bold green]

Commands:
• [cyan]load <pdf_path>[/cyan] - Load a research paper (PDF) for single-paper mode
• [cyan]load-folder <folder_path>[/cyan] - Load all research papers (PDFs) from a folder for multi-paper mode
• [cyan]research <query>[/cyan] - Ask a question about the loaded paper (single-paper)
• [cyan]research --all <query>[/cyan] - Ask a question across all loaded papers (multi-paper)
• [cyan]summarize[/cyan] - Summarize the loaded paper (single-paper)
• [cyan]summarize --all[/cyan] - Summarize all loaded papers (multi-paper)
• [cyan]reference[/cyan] - Show detailed citations for the last query (single-paper)
• [cyan]reference --all[/cyan] - Show detailed citations for the last query across all papers (multi-paper)
• [cyan]export <output_file.md>[/cyan] - Export the last result (summary, research, or reference) to a Markdown file
• [cyan]delete <pdf_path>[/cyan] - Delete a specific loaded paper
• [cyan]delete all[/cyan] - Delete all loaded papers from the database
• [cyan]help[/cyan] - Show this help message
• [cyan]exit[/cyan] - Exit the program

[bold]Backend Management:[/bold]
• [cyan]change-backend <backend>[/cyan] - Switch between 'gemini' and 'ollama' backends
• [cyan]default-backend <backend>[/cyan] - Set default backend (gemini/ollama)
• [cyan]set-llm <model-name>[/cyan] - Set Ollama model and switch to Ollama backend
• [cyan]llm[/cyan] - Check available LLMs and system status
• [cyan]ollama[/cyan] - Ollama-specific commands (models, status)
• [cyan]backend[/cyan] - Show current backend

[bold]Backends:[/bold]
- [green]Gemini[/green]: Uses Google Generative AI (requires API key)
- [green]Ollama[/green]: Uses local open-source LLMs (no API key needed)
  - You can use any model you have downloaded with [italic]ollama pull <model-name>[/italic].
  - Set the model in config.yaml under [italic]ollama_model[/italic].

[dim]Use --all or -a to apply research, summarize, or reference to all loaded papers.[/dim]
"""

class ResearchCLI:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.console = Console() if Console else None
        self._ensure_gemini_api_key()
        self.database = ResearchDatabase(self.config.get('database_path', './research_db'))
        self.ai_assistant = AIAssistant(self.config)
        self.current_paper = None
        self.last_query_references = []
        self.last_output_text = None  # Store last output for export

    def _ensure_gemini_api_key(self):
        if self.config.get('llm_backend', 'gemini') == 'gemini':
            api_key = self.config.get('gemini_api_key', '')
            if not api_key or api_key == 'your-gemini-api-key-here':
                if self.console and Prompt:
                    api_key = Prompt.ask("[bold yellow]Enter your Gemini API key[/bold yellow]")
                else:
                    api_key = input("Enter your Gemini API key: ")
                self.config.set('gemini_api_key', api_key)
                if self.console:
                    self.console.print("[green]API key received, proceed with your queries.[/green]")
                else:
                    print("API key received, proceed with your queries.")

    def load_paper(self, paper_path: str) -> bool:
        if not Path(paper_path).exists():
            self._print(f"[red]Error: Paper file not found: {paper_path}[/red]")
            return False
        if not paper_path.lower().endswith('.pdf'):
            self._print(f"[red]Error: Only PDF files are supported[/red]")
            return False
        self._print("[bold green]Loading paper...[/bold green]", status=True)
        success = self.database.add_paper(paper_path)
        if success:
            self.current_paper = paper_path
            self._print(f"[green]✓ Successfully loaded paper: {Path(paper_path).name}[/green]")
            return True
        else:
            self._print(f"[red]✗ Failed to load paper[/red]")
            return False

    def research_command(self, query: str, all_papers: bool = False, export_path: str = None):
        output_lines = []
        if all_papers:
            self._print("[bold blue]Searching and analyzing across all loaded papers...", status=True)
            references = self.database.query_all_papers(query, n_results=self.config.get('max_context_chunks', 5))
            if not references:
                self._print("[yellow]No relevant information found in any loaded paper.[/yellow]")
                return
            self.last_query_references = references
            answer = self.ai_assistant.answer_question(query, references)
            output_lines.append(f"# Research Query (All Papers)\n**Question:** {query}\n\n**Answer:**\n{answer}\n")
            ref_table = ["| Paper | Page | Section | Relevance |", "|-------|------|---------|-----------|"]
            for ref in references:
                relevance = f"{ref.similarity_score:.2f}"
                ref_table.append(f"| {ref.paper_path if hasattr(ref, 'paper_path') else ''} | {ref.page_number} | {ref.section} | {relevance} |")
            output_lines.append('\n'.join(ref_table))
            self._print("\n" + "="*60)
            self._print(Panel(f"[bold cyan]Question (All Papers):[/bold cyan] {query}", title="Research Query (All Papers)"))
            self._print(Panel(Markdown(answer), title="Answer"))
            ref_table_rich = Table(title="Source Summary (All Papers)")
            ref_table_rich.add_column("Paper", style="magenta")
            ref_table_rich.add_column("Page", style="cyan")
            ref_table_rich.add_column("Section", style="green")
            ref_table_rich.add_column("Relevance", style="yellow")
            for ref in references:
                relevance = f"{ref.similarity_score:.2f}"
                ref_table_rich.add_row(ref.paper_path if hasattr(ref, 'paper_path') else '', str(ref.page_number), ref.section, relevance)
            self._print(ref_table_rich)
            self._print("\n[dim]Use 'reference --all' command to see detailed citations across all papers[/dim]")
            self.last_output_text = '\n\n'.join(output_lines)
            if export_path:
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(self.last_output_text)
            return
        if not self.current_paper:
            self._print("[red]No paper loaded. Please load a paper first.[/red]")
            return
        if not query.strip():
            query = self._prompt("Enter your research question")
        self._print("[bold blue]Searching and analyzing...", status=True)
        references = self.database.query(query, n_results=self.config.get('max_context_chunks', 5))
        if not references:
            self._print("[yellow]No relevant information found in the paper.[/yellow]")
            return
        self.last_query_references = references
        answer = self.ai_assistant.answer_question(query, references)
        self._print("\n" + "="*60)
        self._print(Panel(f"[bold cyan]Question:[/bold cyan] {query}", title="Research Query"))
        self._print(Panel(Markdown(answer), title="Answer"))
        ref_table = Table(title="Source Summary")
        ref_table.add_column("Page", style="cyan")
        ref_table.add_column("Section", style="green")
        ref_table.add_column("Relevance", style="yellow")
        for ref in references:
            relevance = f"{ref.similarity_score:.2f}"
            ref_table.add_row(str(ref.page_number), ref.section, relevance)
        self._print(ref_table)
        self._print("\n[dim]Use 'reference' command to see detailed citations[/dim]")
        self.last_output_text = '\n\n'.join(output_lines)
        if export_path:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(self.last_output_text)

    def reference_command(self, all_papers: bool = False, export_path: str = None):
        output_lines = []
        if all_papers:
            if not self.last_query_references:
                self._print("[yellow]No recent query to show references for.[/yellow]")
                return
            self._print("\n" + "="*60)
            self._print(Panel("[bold magenta]Detailed References (All Papers)", title="Citation Details (All Papers)"))
            paper_names = set()
            for i, ref in enumerate(self.last_query_references, 1):
                paper_name = getattr(ref, 'paper_path', '')
                if paper_name:
                    paper_names.add(Path(paper_name).name)
                ref_content = f"""
**Paper:** {Path(paper_name).name if paper_name else ''}
**Location:** Page {ref.page_number}, Line {ref.line_number}
**Section:** {ref.section}
**Relevance Score:** {ref.similarity_score:.3f}

**Content:**
{ref.content}
                """
                self._print(Panel(
                    Markdown(ref_content),
                    title=f"Reference {i}",
                    border_style="blue"
                ))
                output_lines.append(ref_content)
            if paper_names:
                citations = '\n'.join(f"- {name}" for name in sorted(paper_names))
                self._print(Panel(f"[bold]Citations:[/bold]\n{citations}", title="Citations"))
                output_lines.append(f"\nCitations:\n{citations}")
            self.last_output_text = '\n\n'.join(output_lines)
            if export_path:
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(self.last_output_text)
            return
        if not self.last_query_references:
            self._print("[yellow]No recent query to show references for.[/yellow]")
            return
        self._print("\n" + "="*60)
        self._print(Panel("[bold magenta]Detailed References", title="Citation Details"))
        for i, ref in enumerate(self.last_query_references, 1):
            ref_content = f"""
**Location:** Page {ref.page_number}, Line {ref.line_number}
**Section:** {ref.section}
**Relevance Score:** {ref.similarity_score:.3f}

**Content:**
{ref.content}
            """
            self._print(Panel(
                Markdown(ref_content),
                title=f"Reference {i}",
                border_style="blue"
            ))
        self.last_output_text = '\n\n'.join(output_lines)
        if export_path:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(self.last_output_text)

    def change_backend(self, backend: str):
        """Change the LLM backend on the fly"""
        backend = backend.lower().strip()
        if backend not in ['gemini', 'ollama']:
            self._print("[red]Invalid backend. Use 'gemini' or 'ollama'[/red]")
            return
        
        # Update config
        self.config.set('llm_backend', backend)
        
        # Reinitialize AI assistant with new backend
        try:
            self.ai_assistant = AIAssistant(self.config)
            self._print(f"[green]✓ Backend changed to {backend}[/green]")
            
            # Show current backend info
            if backend == 'gemini':
                api_key = self.config.get('gemini_api_key', '')
                if not api_key or api_key == 'your-gemini-api-key-here':
                    self._print("[yellow]Warning: Gemini API key not set. Use 'default-backend gemini' to configure.[/yellow]")
            elif backend == 'ollama':
                self._print(f"[blue]Ollama URL: {self.config.get('ollama_base_url', 'http://localhost:11434')}[/blue]")
                self._print(f"[blue]Model: {self.config.get('ollama_model', 'llama3')}[/blue]")
                
        except Exception as e:
            self._print(f"[red]Error changing backend: {e}[/red]")
            # Revert to previous backend
            self.config.set('llm_backend', 'gemini' if backend == 'ollama' else 'ollama')

    def default_backend(self, backend: str):
        """Set the default LLM backend and configure it"""
        backend = backend.lower().strip()
        if backend not in ['gemini', 'ollama']:
            self._print("[red]Invalid backend. Use 'gemini' or 'ollama'[/red]")
            return
        
        # Update config
        self.config.set('llm_backend', backend)
        
        # Configure the backend
        if backend == 'gemini':
            api_key = self.config.get('gemini_api_key', '')
            if not api_key or api_key == 'your-gemini-api-key-here':
                if self.console and Prompt:
                    api_key = Prompt.ask("[bold yellow]Enter your Gemini API key[/bold yellow]")
                else:
                    api_key = input("Enter your Gemini API key: ")
                self.config.set('gemini_api_key', api_key)
                self._print("[green]✓ Gemini API key configured[/green]")
            
            # Reinitialize AI assistant
            try:
                self.ai_assistant = AIAssistant(self.config)
                self._print(f"[green]✓ Default backend set to Gemini[/green]")
            except Exception as e:
                self._print(f"[red]Error configuring Gemini: {e}[/red]")
                
        elif backend == 'ollama':
            # Check if Ollama is running
            try:
                import requests
                url = self.config.get('ollama_base_url', 'http://localhost:11434')
                resp = requests.get(f"{url}/api/tags", timeout=5)
                resp.raise_for_status()
                self._print("[green]✓ Ollama server is running[/green]")
                
                # List available models
                models = resp.json().get('models', [])
                if models:
                    self._print(f"[blue]Available models: {', '.join([m['name'] for m in models])}[/blue]")
                else:
                    self._print("[yellow]No models found. Use 'ollama pull <model-name>' to download models.[/yellow]")
                
                # Reinitialize AI assistant
                self.ai_assistant = AIAssistant(self.config)
                self._print(f"[green]✓ Default backend set to Ollama[/green]")
                
            except Exception as e:
                self._print(f"[red]Error connecting to Ollama: {e}[/red]")
                self._print("[yellow]Make sure Ollama is running with 'ollama serve'[/yellow]")

    def set_llm_model(self, model_name: str):
        """Set the Ollama model and switch to Ollama backend"""
        if not model_name.strip():
            self._print("[red]Please specify a model name[/red]")
            return
        
        model_name = model_name.strip()
        self._print(f"[blue]Setting Ollama model to: {model_name}[/blue]")
        
        try:
            import requests
            url = self.config.get('ollama_base_url', 'http://localhost:11434')
            
            # Check if Ollama is running
            try:
                resp = requests.get(f"{url}/api/tags", timeout=5)
                resp.raise_for_status()
            except Exception as e:
                self._print(f"[red]Error connecting to Ollama server: {e}[/red]")
                self._print("[yellow]Make sure Ollama is running with 'ollama serve'[/yellow]")
                return
            
            # Get available models
            models = resp.json().get('models', [])
            available_models = [m['name'].strip() for m in models]
            
            # Debug: Print what we're comparing
            self._print(f"[dim]Looking for: '{model_name}'[/dim]")
            self._print(f"[dim]Available models: {available_models}[/dim]")
            
            # Check if the requested model exists (with flexible matching)
            model_found = False
            exact_match = None
            
            # First try exact match
            if model_name in available_models:
                model_found = True
                exact_match = model_name
            else:
                # Try case-insensitive match
                for available in available_models:
                    if model_name.lower() == available.lower():
                        model_found = True
                        exact_match = available
                        break
                
                # If still not found, try partial matching (e.g., "gemma3" matches "gemma3:latest")
                if not model_found:
                    for available in available_models:
                        if (model_name.lower() in available.lower() or 
                            available.lower().startswith(model_name.lower() + ':')):
                            model_found = True
                            exact_match = available
                            self._print(f"[yellow]Using closest match: {available}[/yellow]")
                            break
            
            if not model_found:
                self._print(f"[red]Model '{model_name}' not found on Ollama server[/red]")
                if available_models:
                    self._print(f"[blue]Available models: {', '.join(available_models)}[/blue]")
                    self._print(f"[yellow]To download the model, use: ollama pull {model_name}[/yellow]")
                else:
                    self._print("[yellow]No models found. Download a model with: ollama pull <model-name>[/yellow]")
                return
            
            # Use the exact match found
            final_model_name = exact_match or model_name
            
            # Update configuration
            self.config.set('ollama_model', final_model_name)
            self.config.set('llm_backend', 'ollama')
            
            # Reinitialize AI assistant with new model
            try:
                self.ai_assistant = AIAssistant(self.config)
                self._print(f"[green]✓ Successfully set Ollama model to: {final_model_name}[/green]")
                self._print(f"[green]✓ Backend switched to Ollama[/green]")
                self._print(f"[blue]Ollama URL: {url}[/blue]")
            except Exception as e:
                self._print(f"[red]Error initializing AI assistant with new model: {e}[/red]")
                
        except ImportError:
            self._print("[red]requests library not available. Cannot check Ollama server.[/red]")
        except Exception as e:
            self._print(f"[red]Error setting Ollama model: {e}[/red]")

    def check_llm_status(self):
        """Check which LLMs are available on the system"""
        self._print("[bold blue]LLM System Status Check[/bold blue]")
        
        # Check Gemini
        self._print("\n[bold cyan]Gemini Backend:[/bold cyan]")
        api_key = self.config.get('gemini_api_key', '')
        if api_key and api_key != 'your-gemini-api-key-here':
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                self._print("[green]✓ Gemini API key configured and valid[/green]")
            except ImportError:
                self._print("[red]✗ google-generativeai package not installed[/red]")
            except Exception as e:
                self._print(f"[red]✗ Gemini API error: {e}[/red]")
        else:
            self._print("[yellow]⚠ Gemini API key not configured[/yellow]")
        
        # Check Ollama
        self._print("\n[bold cyan]Ollama Backend:[/bold cyan]")
        try:
            import requests
            url = self.config.get('ollama_base_url', 'http://localhost:11434')
            resp = requests.get(f"{url}/api/tags", timeout=5)
            resp.raise_for_status()
            self._print("[green]✓ Ollama server is running[/green]")
            
            models = resp.json().get('models', [])
            if models:
                self._print(f"[green]✓ Available models: {', '.join([m['name'] for m in models])}[/green]")
            else:
                self._print("[yellow]⚠ Ollama running but no models installed[/yellow]")
                
        except Exception as e:
            self._print(f"[red]✗ Ollama server not accessible: {e}[/red]")
            self._print("[yellow]Start Ollama with 'ollama serve'[/yellow]")
        
        # Show current backend
        current_backend = self.config.get('llm_backend', 'gemini')
        current_model = self.config.get('ollama_model', 'llama3') if current_backend == 'ollama' else 'gemini-2.0-flash'
        self._print(f"\n[bold green]Current Backend: {current_backend.title()}[/bold green]")
        self._print(f"[bold green]Current Model: {current_model}[/bold green]")

    def _ollama_commands(self):
        """Handle Ollama-specific commands"""
        self._print("[blue]Ollama Commands:[/blue]")
        self._print("• [cyan]ollama models[/cyan] - List available models")
        self._print("• [cyan]ollama status[/cyan] - Check server status")
        self._print("• [cyan]ollama pull <model>[/cyan] - Download a model")
        self._print("\n[dim]Example: ollama models, ollama status, ollama pull llama3[/dim]")

    def _list_ollama_models(self):
        """List available Ollama models"""
        try:
            import requests
            url = self.config.get('ollama_base_url', 'http://localhost:11434')
            resp = requests.get(f"{url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get('models', [])
            if not models:
                self._print("[yellow]No models found on your Ollama server. Use 'ollama pull <model-name>' to download models.[/yellow]")
                return
            table = Table(title="Available Ollama Models")
            table.add_column("Model Name", style="cyan")
            for m in models:
                table.add_row(m['name'])
            self._print(table)
        except Exception as e:
            self._print(f"[red]Error listing Ollama models: {e}[/red]")

    def _check_ollama_status(self):
        """Check Ollama server status"""
        try:
            import requests
            url = self.config.get('ollama_base_url', 'http://localhost:11434')
            resp = requests.get(f"{url}/api/tags", timeout=5)
            resp.raise_for_status()
            self._print("[green]✓ Ollama server is running[/green]")
            
            models = resp.json().get('models', [])
            if models:
                self._print(f"[blue]Available models: {', '.join([m['name'] for m in models])}[/blue]")
            else:
                self._print("[yellow]No models installed[/yellow]")
                
        except Exception as e:
            self._print(f"[red]✗ Ollama server not accessible: {e}[/red]")
            self._print("[yellow]Start Ollama with 'ollama serve'[/yellow]")

    def _pull_ollama_model(self, model_name: str):
        """Download an Ollama model"""
        self._print(f"[blue]Downloading model: {model_name}...[/blue]")
        self._print("[yellow]This may take a while depending on model size...[/yellow]")
        
        try:
            import requests
            url = self.config.get('ollama_base_url', 'http://localhost:11434')
            data = {"name": model_name}
            resp = requests.post(f"{url}/api/pull", json=data, timeout=300)
            resp.raise_for_status()
            self._print(f"[green]✓ Model {model_name} downloaded successfully![/green]")
        except Exception as e:
            self._print(f"[red]Error downloading model: {e}[/red]")

    def summarize_command(self, paper_path: str = None, all_papers: bool = False, export_path: str = None):
        output_lines = []
        if all_papers:
            self._print("[bold blue]Summarizing all loaded papers...", status=True)
            all_references = self.database.get_all_references_for_all_papers()
            if not all_references:
                self._print("[yellow]No content found for any loaded paper.[/yellow]")
                return
            summary = self.ai_assistant.summarize_paper(all_references)
            output_lines.append(f"# Summary for All Loaded Papers\n{summary}")
            self._print("\n" + "="*60)
            self._print(Panel(f"[bold cyan]Summary for All Loaded Papers[/bold cyan]", title="Paper Summary (All Papers)"))
            self._print(Panel(Markdown(summary), title="Summary (All Papers)"))
            self.last_output_text = '\n\n'.join(output_lines)
            if export_path:
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(self.last_output_text)
            return
        if not paper_path:
            paper_path = self.current_paper
        if not paper_path:
            self._print("[red]No paper specified or loaded. Please load a paper or provide a path.[/red]")
            return
        self._print("[bold blue]Summarizing paper...", status=True)
        references = self.database.get_all_references_for_paper(paper_path)
        if not references:
            self._print("[yellow]No content found for the specified paper.[/yellow]")
            return
        summary = self.ai_assistant.summarize_paper(references)
        self._print("\n" + "="*60)
        self._print(Panel(f"[bold cyan]Summary for:[/bold cyan] {Path(paper_path).name}", title="Paper Summary"))
        self._print(Panel(Markdown(summary), title="Summary"))
        self.last_output_text = '\n\n'.join(output_lines)
        if export_path:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(self.last_output_text)

    def delete_command(self, paper_path: str = None):
        if not paper_path:
            self._print("[red]Please specify a paper to delete, or use 'delete all' to clear all documents.[/red]")
            return
        if paper_path.strip().lower() == 'all':
            try:
                # ChromaDB requires a valid where clause, so we'll delete all documents
                self.database.collection.delete(where={"paper_id": {"$exists": True}})
                self._print("[green]All loaded documents have been deleted from the database.[/green]")
            except Exception as e:
                self._print(f"[red]Error deleting all documents: {e}[/red]")
                # Fallback: try to get all documents and delete them individually
                try:
                    results = self.database.collection.get()
                    if results.get('ids'):
                        self.database.collection.delete(ids=results['ids'])
                        self._print("[green]All documents deleted using fallback method[/green]")
                except Exception as e2:
                    self._print(f"[red]Failed to delete all documents: {e2}[/red]")
            return
        success = self.database.delete_paper(paper_path)
        if success:
            self._print(f"[green]Deleted paper: {Path(paper_path).name} from the database.[/green]")
            if self.current_paper == paper_path:
                self.current_paper = None
        else:
            self._print(f"[red]Failed to delete paper: {Path(paper_path).name}[/red]")

    def load_folder_command(self, folder_path: str):
        from pathlib import Path
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            self._print(f"[red]Folder not found: {folder_path}[/red]")
            return
        pdf_files = list(folder.glob('*.pdf'))
        if not pdf_files:
            self._print(f"[yellow]No PDF files found in folder: {folder_path}[/yellow]")
            return
        
        # Clear existing database before loading new papers
        self._print("[blue]Clearing existing database...[/blue]")
        try:
            # ChromaDB requires a valid where clause, so we'll delete all documents
            self.database.collection.delete(where={"paper_id": {"$exists": True}})
            self._print("[green]✓ Database cleared[/green]")
        except Exception as e:
            self._print(f"[yellow]Warning: Could not clear database: {e}[/yellow]")
            # Fallback: try to get all documents and delete them individually
            try:
                results = self.database.collection.get()
                if results.get('ids'):
                    self.database.collection.delete(ids=results['ids'])
                    self._print("[green]✓ Database cleared using fallback method[/green]")
            except Exception as e2:
                self._print(f"[red]Failed to clear database: {e2}[/red]")
        
        loaded = 0
        failed = []
        for pdf in pdf_files:
            self._print(f"[blue]Loading: {pdf.name}...[/blue]")
            if self.database.add_paper(str(pdf)):
                loaded += 1
            else:
                failed.append(pdf.name)
        self._print(f"[green]Loaded {loaded} PDF(s) from {folder_path}.[/green]")
        if failed:
            self._print(f"[red]Failed to load: {', '.join(failed)}[/red]")

    def export_command(self, export_path: str):
        if not self.last_output_text:
            self._print("[yellow]No output to export. Run a research, summarize, or reference command first.[/yellow]")
            return
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(self.last_output_text)
            self._print(f"[green]Exported last output to {export_path}[/green]")
        except Exception as e:
            self._print(f"[red]Failed to export: {e}[/red]")

    def interactive_mode(self):
        self._print(Panel(HELP_TEXT, title="Welcome"))
        while True:
            try:
                command = self._prompt("\n[bold blue]docs-chat[/bold blue]").strip()
                if not command:
                    continue
                parts = command.split()
                cmd = parts[0].lower()
                args = parts[1:]
                all_flag = False
                if '--all' in args or '-a' in args:
                    all_flag = True
                    args = [a for a in args if a not in ('--all', '-a')]
                arg_str = ' '.join(args)
                
                if cmd == "exit":
                    self._print("[green]Goodbye![/green]")
                    break
                elif cmd == "help":
                    self._print(Panel(HELP_TEXT, title="Help"))
                elif cmd == "load":
                    if arg_str:
                        self.load_paper(arg_str)
                    else:
                        self._print("[red]Usage: load <pdf_path>[/red]")
                elif cmd == "load-folder":
                    if arg_str:
                        self.load_folder_command(arg_str)
                    else:
                        self._print("[red]Usage: load-folder <folder_path>[/red]")
                elif cmd == "research":
                    if arg_str:
                        self.research_command(arg_str, all_papers=all_flag)
                    else:
                        self.research_command(all_papers=all_flag)
                elif cmd == "reference":
                    self.reference_command(all_papers=all_flag)
                elif cmd == "summarize":
                    if arg_str:
                        self.summarize_command(arg_str, all_papers=all_flag)
                    else:
                        self.summarize_command(all_papers=all_flag)
                elif cmd == "delete":
                    if arg_str:
                        self.delete_command(arg_str)
                    else:
                        self.delete_command()
                elif cmd == "export":
                    if arg_str:
                        self.export_command(arg_str)
                    else:
                        self._print("[red]Usage: export <output_file.md>[/red]")
                elif cmd == "change-backend":
                    if arg_str:
                        self.change_backend(arg_str)
                    else:
                        self._print("[red]Usage: change-backend <gemini|ollama>[/red]")
                elif cmd == "default-backend":
                    if arg_str:
                        self.default_backend(arg_str)
                    else:
                        self._print("[red]Usage: default-backend <gemini|ollama>[/red]")
                elif cmd == "set-llm":
                    if arg_str:
                        self.set_llm_model(arg_str)
                    else:
                        self._print("[red]Usage: set-llm <model-name>[/red]")
                        self._print("[blue]Example: set-llm phi3, set-llm llama3.2:latest[/blue]")
                elif cmd == "llm":
                    self.check_llm_status()
                elif cmd == "backend":
                    current = self.config.get('llm_backend', 'gemini')
                    current_model = self.config.get('ollama_model', 'llama3') if current == 'ollama' else 'gemini-2.0-flash'
                    self._print(f"[bold green]Current backend: {current.title()}[/bold green]")
                    self._print(f"[bold green]Current model: {current_model}[/bold green]")
                elif cmd == "ollama":
                    if not arg_str:
                        self._ollama_commands()
                    else:
                        subcmd = arg_str.split()[0].lower()
                        if subcmd == "models":
                            self._list_ollama_models()
                        elif subcmd == "status":
                            self._check_ollama_status()
                        elif subcmd == "pull":
                            if len(arg_str.split()) < 2:
                                self._print("[red]Usage: ollama pull <model-name>[/red]")
                            else:
                                model_name = arg_str.split()[1]
                                self._pull_ollama_model(model_name)
                        else:
                            self._print(f"[red]Unknown Ollama command: {subcmd}[/red]")
                            self._print("[blue]Use 'ollama' to see available commands[/blue]")
                else:
                    self._print(f"[red]Unknown command: {cmd}[/red]")
                    
            except KeyboardInterrupt:
                self._print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                self._print(f"[red]Error: {e}[/red]")

    def _print(self, msg, status=False):
        if self.console:
            if status:
                with self.console.status(msg):
                    pass
            else:
                self.console.print(msg)
        else:
            print(msg)

    def _prompt(self, msg):
        if self.console and Prompt:
            return Prompt.ask(msg)
        else:
            return input(msg + ": ")