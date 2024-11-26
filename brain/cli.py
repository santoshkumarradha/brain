import subprocess

import click
from rich import print
from rich.console import Console

console = Console()


@click.group()
def cli():
    pass


@cli.command()
def start():
    """Start the FastAPI server."""
    console.print("[bold green]Starting FastAPI server...[/bold green]")
    try:
        subprocess.run(["uvicorn", "brain.server:app", "--reload"], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Failed to start server: {e}[/bold red]")


if __name__ == "__main__":
    cli()
