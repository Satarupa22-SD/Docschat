import argparse
from docs_chat.cli import ResearchCLI

def main():
    parser = argparse.ArgumentParser(description="Docs Chat - Research Paper CLI System")
    parser.add_argument("--load", help="Load a research paper PDF")
    parser.add_argument("--load-folder", help="Load all research paper PDFs from a folder")
    parser.add_argument("--research", help="Ask a research question")
    parser.add_argument("--reference", action="store_true", help="Show references for last query")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--summarize", nargs="?", const=True, help="Summarize a research paper (optionally provide PDF path)")
    parser.add_argument("--delete", nargs="?", const=True, help="Delete a research paper (optionally provide PDF path or 'all')")
    parser.add_argument("--all", "-a", action="store_true", help="Apply command to all loaded papers (multi-paper mode)")
    parser.add_argument("--export", help="Export results to a file (Markdown or text)")
    args = parser.parse_args()
    cli = ResearchCLI(config_path=args.config)
    if args.load:
        cli.load_paper(args.load)
    if args.research:
        if args.all:
            cli.research_command(args.research, all_papers=True, export_path=args.export)
        else:
            cli.research_command(args.research, export_path=args.export)
    if args.reference:
        if args.all:
            cli.reference_command(all_papers=True, export_path=args.export)
        else:
            cli.reference_command(export_path=args.export)
    if args.summarize:
        if args.all:
            cli.summarize_command(all_papers=True, export_path=args.export)
        elif isinstance(args.summarize, str):
            cli.summarize_command(args.summarize, export_path=args.export)
        else:
            cli.summarize_command(export_path=args.export)
    if args.delete:
        if isinstance(args.delete, str):
            cli.delete_command(args.delete)
        else:
            cli.delete_command()
    if args.load_folder:
        cli.load_folder_command(args.load_folder)
    if args.interactive or not any([args.load, args.research, args.reference]):
        cli.interactive_mode()

if __name__ == "__main__":
    main()
