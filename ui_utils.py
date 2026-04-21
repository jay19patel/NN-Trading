# -*- coding: utf-8 -*-
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.table import Table
from rich.theme import Theme
from rich.text import Text
from typing import Optional, List, Dict, Any

# Custom theme for the Trading Engine
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "buy": "bold green",
    "sell": "bold red",
    "neutral": "bright_black",
    "highlight": "bold magenta"
})

# Singleton console instance
console = Console(theme=custom_theme)

def print_banner(title: str, subtitle: Optional[str] = None):
    """Prints a stylized banner for project phases"""
    content = Text(title, style="highlight")
    if subtitle:
        content.append(f"\n{subtitle}", style="info")
    
    console.print(Panel(content, expand=False, border_style="highlight"))

def get_progress():
    """Returns a pre-configured Progress instance"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True # Disappears after completion
    )

def create_metric_table(title: str, data: List[Dict[str, Any]]) -> Table:
    """Creates a styled table for metrics display"""
    table = Table(title=title, show_header=True, header_style="bold magenta", border_style="bright_black")
    
    if not data:
        return table
        
    # Add columns based on first item keys
    for key in data[0].keys():
        table.add_column(key.replace("_", " ").title(), justify="right")
        
    # Add rows
    for item in data:
        row_values = []
        for v in item.values():
            if isinstance(v, float):
                row_values.append(f"{v:.4f}")
            else:
                row_values.append(str(v))
        table.add_row(*row_values)
        
    return table
