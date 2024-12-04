import os
import signal
import subprocess
from datetime import datetime

import click
import requests
from rich import box, print
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

# Define a Monokai-like theme
monokai_theme = Theme(
    {
        "success": "bold green",
        "error": "bold red",
        "info": "bold blue",
        "warning": "bold yellow",
        "title": "bold magenta",
        "dim": "dim white",
        "highlight": "bold cyan",
    }
)

console = Console(theme=monokai_theme)
PID_FILE = ".brain_server_pid"


@click.group()
def cli():
    pass


@cli.command()
@click.option("--debug", is_flag=True, help="Run Brain in debug mode.")
def start(debug):
    """Start Brain."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[error]OPENAI_API_KEY environment variable is not set. Please set it and try again.[/error]"
        )
        return

    if debug:
        console.print("[info]Brain is powering up in debug mode...[/info]")
        try:
            subprocess.run(["uvicorn", "brain.server:app", "--reload"], check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[error]Failed to start Brain: {e}[/error]")
    else:
        console.print("[info]Brain is coming online in the background...[/info]")
        try:
            process = subprocess.Popen(
                ["uvicorn", "brain.server:app"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with open(PID_FILE, "w") as f:
                f.write(str(process.pid))
            console.print(
                f"[success]Brain started successfully with PID {process.pid}.[/success]"
            )
        except Exception as e:
            console.print(f"[error]Failed to start Brain: {e}[/error]")


@cli.command()
def stop():
    """Stop Brain."""
    if not os.path.exists(PID_FILE):
        console.print("[error]No active Brain process found.[/error]")
        return

    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        os.remove(PID_FILE)
        console.print(
            f"[success]Brain process with PID {pid} has been shut down gracefully.[/success]"
        )
    except Exception as e:
        console.print(f"[error]Failed to stop Brain: {e}[/error]")


@cli.command()
@click.option("--multiagent", default=None, help="Filter runs by multiagent name.")
@click.option("--reasoner", default=None, help="Filter runs by reasoner name.")
@click.option("--project", default=None, help="Filter runs by project ID.")
@click.option(
    "--order-by",
    default="start_time",
    help="Order runs by a field (e.g., start_time, duration).",
)
@click.option(
    "--descending", default=True, is_flag=True, help="Order runs in descending order."
)
def runs(multiagent, reasoner, project, order_by, descending):
    """View multiagent runs."""
    console.print("[info]Retrieving Brain multiagent runs...[/info]")

    try:
        params = {
            "multiagent_name": multiagent,
            "reasoner_name": reasoner,
            "project_id": project,
        }
        response = requests.get("http://127.0.0.1:8000/list_runs", params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch runs: {response.text}")

        runs = response.json()["sessions"]

        if not runs:
            console.print("[error]No runs found with the given filters.[/error]")
            return

        # Sort runs based on the provided order_by field
        runs = sorted(runs, key=lambda x: x.get(order_by, ""), reverse=descending)

        # Create a table for displaying runs
        table = Table(
            title="[title]Filtered MultiAgent Runs[/title]",
            box=box.SIMPLE,
            show_lines=True,
        )
        table.add_column("Run ID", justify="left")
        table.add_column("MultiAgent", justify="left")
        table.add_column("Reasoners (Calls)", justify="left")
        table.add_column("Duration", justify="left")
        table.add_column("Start Time", justify="left")

        for run in runs:
            reasoner_calls_summary = " → ".join(
                [
                    f"{call['reasoner_name']} ([dim]{round(call['duration'], 2)}s[/dim])"
                    for call in run.get("reasoner_calls", [])
                ]
            )
            start_time = run.get("start_time", "N/A")
            if start_time != "N/A":
                start_time = datetime.fromisoformat(start_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            table.add_row(
                run.get("session_id", "N/A"),
                run.get("multiagent_name", "N/A"),
                reasoner_calls_summary if reasoner_calls_summary else "N/A",
                str(round(run.get("total_duration", 0), 2)),
                start_time,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[error]Failed to retrieve runs: {e}[/error]")


if __name__ == "__main__":
    cli()
