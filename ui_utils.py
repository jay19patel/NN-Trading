# -*- coding: utf-8 -*-
from rich.console import Console
from rich.theme import Theme

# Custom theme for professional look
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
})

console = Console(theme=custom_theme)
