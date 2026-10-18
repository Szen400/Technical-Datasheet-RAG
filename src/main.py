"""Interactive CLI and execution entrypoint for Technical Datasheet RAG."""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from src.rag_pipeline import TechnicalRAG

console = Console()


def display_rag_result(result: dict) -> None:
    """Pretty-print the RAG query result, citations, and execution latency."""
    console.print()
    console.print(Panel(f"[bold cyan]{result['query']}[/bold cyan]", title="Query", border_style="cyan"))

    console.print(Panel(result["answer"], title="Technical Assistant Answer", border_style="green"))

    # Source Attribution Table
    table = Table(title="Retrieved Citations & Context Chunks", border_style="dim")
    table.add_column("#", justify="center", style="bold", width=3)
    table.add_column("Document", style="yellow")
    table.add_column("Page", justify="center", width=6)
    table.add_column("Relevance", justify="right", style="green", width=11)
    table.add_column("Snippet Preview", style="dim white")

    for i, src in enumerate(result["sources"], 1):
        clean_snippet = src["content"].replace("\n", " ")
        if len(clean_snippet) > 100:
            clean_snippet = clean_snippet[:97] + "..."
        table.add_row(
            str(i),
            str(src["source"]),
            str(src["page"]),
            f"{src['similarity'] * 100:.1f}%",
            clean_snippet
        )

    console.print(table)

    latency = result["latency"]
    console.print(
        f"[dim]⚡ Retrieval: {latency['retrieval_seconds']}s | "
        f"Generation: {latency['generation_seconds']}s | "
        f"Total: {latency['total_seconds']}s[/dim]\n"
    )


def interactive_mode(rag: TechnicalRAG) -> None:
    """Run an interactive REPL loop in the terminal."""
    console.print(
        Panel.fit(
            "[bold green]Technical Datasheet RAG Assistant[/bold green]\n"
            "Ask questions about your technical datasheets (e.g. electrical specs, IP rating, operating temperature).\n"
            "Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.",
            border_style="blue"
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Datasheet Question[/bold cyan]")
            if user_input.strip().lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting Technical Datasheet RAG. Goodbye![/dim]")
                break
            if not user_input.strip():
                continue

            with console.status("[bold green]Searching technical documentation and generating answer...[/bold green]"):
                res = rag.query(user_input.strip())

            display_rag_result(res)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break


def main():
    parser = argparse.ArgumentParser(description="Query technical product datasheets with RAG")
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Single question to query against the technical documentation"
    )
    args = parser.parse_args()

    rag = TechnicalRAG()

    if args.question:
        with console.status("[bold green]Querying RAG pipeline...[/bold green]"):
            res = rag.query(args.question)
        display_rag_result(res)
    else:
        interactive_mode(rag)


if __name__ == "__main__":
    main()
