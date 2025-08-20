"""Command-line interface for GPT-OSS red-teaming framework."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_config, save_config
from .eval.converters import harmony_to_kaggle, kaggle_to_harmony
from .eval.reporting import generate_report, make_findings_json, summarize_results
from .eval.summaries import create_summary
from .logging_setup import setup_logging
from .model.backends import get_backend
from .model.sampling import set_seed
from .probes.base import ProbeResult
from .probes.registry import discover_probes, list_probes, run_category, run_probe

app = typer.Typer(
    name="redteam",
    help="Red-teaming framework for GPT-OSS-20B",
    add_completion=False,
)
console = Console()

# Global state
_config = None
_backend = None


def get_config():
    """Get or load configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_model_backend():
    """Get or initialize model backend."""
    global _backend
    if _backend is None:
        config = get_config()
        _backend = get_backend(
            config.backend.type,
            config.backend.model_dump(),
            config.providers,
        )
    return _backend


@app.command()
def init():
    """Initialize the red-teaming environment."""
    console.print("[bold green]Initializing red-team environment...[/bold green]")
    
    # Create directories
    dirs = ["runs", "findings_templates", "examples", "config"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        console.print(f"  ✓ Created {dir_name}/")
    
    # Check for config files
    if not Path("config/settings.yaml").exists():
        if Path("config/settings.example.yaml").exists():
            console.print("  ⚠ Copy config/settings.example.yaml to config/settings.yaml")
        else:
            console.print("  ⚠ No settings.yaml found")
    
    # Discover probes
    discovered = discover_probes()
    total_probes = sum(len(probes) for probes in discovered.values())
    console.print(f"\n[bold]Discovered {total_probes} probes in {len(discovered)} categories[/bold]")
    
    console.print("\n[bold green]Initialization complete![/bold green]")


@app.command()
def list(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """List available probes."""
    # Discover probes first
    discover_probes()
    
    probe_names = list_probes(category=category)
    
    if not probe_names:
        console.print(f"[yellow]No probes found{f' in category {category}' if category else ''}[/yellow]")
        return
    
    table = Table(title=f"Available Probes{f' - {category}' if category else ''}")
    table.add_column("Probe", style="cyan")
    table.add_column("Category", style="green")
    
    for name in sorted(probe_names):
        parts = name.split(".")
        if len(parts) == 2:
            table.add_row(parts[1], parts[0])
        else:
            table.add_row(name, "unknown")
    
    console.print(table)
    console.print(f"\n[bold]Total: {len(probe_names)} probes[/bold]")


@app.command()
def run(
    categories: str = typer.Option(None, "--categories", "-c", help="Comma-separated categories"),
    probes: str = typer.Option(None, "--probes", "-p", help="Comma-separated probe names"),
    backend: str = typer.Option(None, "--backend", "-b", help="Backend to use"),
    seeds: str = typer.Option(None, "--seeds", "-s", help="Comma-separated seeds"),
    out: str = typer.Option(None, "--out", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run probes against the model."""
    # Setup
    config = get_config()
    setup_logging(level="DEBUG" if verbose else config.logging.level)
    
    # Parse inputs
    category_list = categories.split(",") if categories else []
    probe_list = probes.split(",") if probes else []
    seed_list = [int(s) for s in seeds.split(",")] if seeds else config.seeds.default
    
    if not category_list and not probe_list:
        console.print("[red]Error: Specify --categories or --probes[/red]")
        raise typer.Exit(1)
    
    # Setup output directory
    if out:
        out_dir = Path(out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(config.output.base_dir) / timestamp
    
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Output directory: {out_dir}[/green]")
    
    # Get backend
    if backend:
        config.backend.type = backend
    model = get_model_backend()
    
    # Discover probes
    discover_probes()
    
    # Run probes
    results = []
    
    with console.status("[bold green]Running probes...") as status:
        # Run by category
        for category in category_list:
            status.update(f"Running category: {category}")
            cat_results = run_category(category, model, seeds=seed_list)
            results.extend(cat_results)
            console.print(f"  ✓ {category}: {len(cat_results)} results")
        
        # Run individual probes
        for probe_name in probe_list:
            status.update(f"Running probe: {probe_name}")
            for seed in seed_list:
                try:
                    result = run_probe(probe_name, model, seed=seed)
                    results.append(result)
                except Exception as e:
                    console.print(f"  ✗ {probe_name}: {e}", style="red")
    
    # Save results
    results_file = out_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    # Generate summary
    summary = create_summary(results, verbose=True)
    console.print("\n" + summary)
    
    # Save summary
    summary_file = out_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    
    console.print(f"\n[bold green]Results saved to {out_dir}[/bold green]")


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Run directory to report on"),
    format: str = typer.Option("markdown", "--format", "-f", help="Report format"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Generate a report from run results."""
    run_path = Path(run_dir)
    results_file = run_path / "results.json"
    
    if not results_file.exists():
        console.print(f"[red]Error: No results.json in {run_dir}[/red]")
        raise typer.Exit(1)
    
    # Load results
    with open(results_file) as f:
        results_data = json.load(f)
    
    # Convert back to ProbeResult objects
    results = []
    for data in results_data:
        result = ProbeResult(
            probe_name=data["probe_name"],
            passed=data["passed"],
            evidence=data.get("evidence", []),
            metrics=data.get("metrics", {}),
            seed=data.get("seed"),
            tags=data.get("tags", []),
            duration_ms=data.get("duration_ms", 0),
            error=data.get("error"),
        )
        results.append(result)
    
    # Generate report
    report_content = generate_report(results, format=format)
    
    # Save or print
    if output:
        with open(output, "w") as f:
            f.write(report_content)
        console.print(f"[green]Report saved to {output}[/green]")
    else:
        console.print(report_content)


@app.command()
def findings(
    issue_id: str = typer.Argument(..., help="Issue ID for findings"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", "-r", help="Run directory"),
    out: str = typer.Option(..., "--out", "-o", help="Output findings JSON path"),
):
    """Generate Kaggle findings JSON from run results."""
    if run_dir:
        run_path = Path(run_dir)
        results_file = run_path / "results.json"
        
        if not results_file.exists():
            console.print(f"[red]Error: No results.json in {run_dir}[/red]")
            raise typer.Exit(1)
        
        # Load results
        with open(results_file) as f:
            results_data = json.load(f)
        
        # Convert to ProbeResult objects
        results = []
        for data in results_data:
            result = ProbeResult(
                probe_name=data["probe_name"],
                passed=data["passed"],
                evidence=data.get("evidence", []),
                metrics=data.get("metrics", {}),
                repro_prompts=data.get("repro_prompts", []),
                seed=data.get("seed"),
                tags=data.get("tags", []),
                duration_ms=data.get("duration_ms", 0),
            )
            results.append(result)
    else:
        console.print("[yellow]No run directory specified, creating template[/yellow]")
        results = []
    
    # Generate findings
    meta = {
        "generator": "gpt-oss-redteam",
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    findings_data = make_findings_json(results, issue_id, meta, out)
    
    console.print(f"[green]Findings saved to {out}[/green]")
    console.print(f"  Issue ID: {issue_id}")
    console.print(f"  Findings: {len(findings_data['findings'])}")


@app.command("import-harmony")
def import_harmony(
    input: str = typer.Option(..., "--in", "-i", help="Input Harmony findings JSON"),
    output: str = typer.Option(..., "--out", "-o", help="Output Kaggle findings JSON"),
):
    """Import Harmony findings and convert to Kaggle format."""
    if not Path(input).exists():
        console.print(f"[red]Error: Input file {input} not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Converting {input} to Kaggle format...[/cyan]")
    
    try:
        result = harmony_to_kaggle(input, output)
        console.print(f"[green]✓ Converted {len(result['findings'])} findings[/green]")
        console.print(f"[green]✓ Saved to {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("export-harmony")
def export_harmony(
    input: str = typer.Option(..., "--in", "-i", help="Input Kaggle findings JSON"),
    output: str = typer.Option(..., "--out", "-o", help="Output Harmony findings JSON"),
):
    """Export Kaggle findings to Harmony format."""
    if not Path(input).exists():
        console.print(f"[red]Error: Input file {input} not found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Converting {input} to Harmony format...[/cyan]")
    
    try:
        result = kaggle_to_harmony(input, output)
        console.print(f"[green]✓ Converted {len(result['findings'])} findings[/green]")
        console.print(f"[green]✓ Saved to {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold]GPT-OSS Red Team Framework[/bold]")
    console.print(f"Version: {__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()