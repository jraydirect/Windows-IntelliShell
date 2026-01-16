"""Rich display utilities for Intent Shell output formatting."""

from typing import List, Dict, Any, Optional
from intent_shell.utils.terminal import TerminalColors


def format_message(message: str, success: bool = True, is_error: bool = False, is_warning: bool = False) -> str:
    """
    Format message with color coding.
    
    Args:
        message: Message text
        success: Whether operation succeeded
        is_error: Whether this is an error message
        is_warning: Whether this is a warning message
        
    Returns:
        Color-coded message string
    """
    if not TerminalColors.supports_color():
        return message
    
    # Error messages (red)
    if is_error or (not success and not is_warning):
        return TerminalColors.colorize(message, TerminalColors.BRIGHT_RED)
    
    # Warning messages (yellow)
    if is_warning:
        return TerminalColors.colorize(message, TerminalColors.BRIGHT_YELLOW)
    
    # Success messages (green) - only for positive success messages
    if success and any(word in message.lower() for word in ["success", "opening", "opened", "terminated", "completed", "done", "copied"]):
        return TerminalColors.colorize(message, TerminalColors.BRIGHT_GREEN)
    
    # Info messages (cyan)
    if any(word in message.lower() for word in ["info", "thinking", "debug"]):
        return TerminalColors.colorize(message, TerminalColors.BRIGHT_CYAN)
    
    return message


def format_table_with_rich(rows: List[List[str]], headers: List[str], title: Optional[str] = None) -> str:
    """
    Format data as a rich table if available, otherwise plain text.
    
    Args:
        rows: List of row data (each row is a list of strings)
        headers: Column headers
        title: Optional table title
        
    Returns:
        Formatted table string
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from io import StringIO
        
        # Create console that writes to string buffer
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=TerminalColors.supports_color(), width=None)
        table = Table(show_header=True, header_style="bold bright_blue", border_style="bright_blue", show_lines=False)
        
        # Add columns
        for header in headers:
            table.add_column(header, style="white")
        
        # Add rows
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        
        # Render table to string buffer
        console.print(table)
        result = string_buffer.getvalue()
        
        if title:
            title_colored = TerminalColors.colorize(title, TerminalColors.BRIGHT_BLUE)
            return f"{title_colored}\n{result}"
        
        return result
        
    except (ImportError, AttributeError, Exception):
        # Fallback to plain text table
        return format_table_plain(rows, headers, title)


def format_table_plain(rows: List[List[str]], headers: List[str], title: Optional[str] = None) -> str:
    """
    Format data as plain text table.
    
    Args:
        rows: List of row data
        headers: Column headers
        title: Optional table title
        
    Returns:
        Plain text table string
    """
    if not rows:
        return title + "\n(No data)" if title else "(No data)"
    
    # Calculate column widths
    col_widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Build table
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    
    # Header row
    header_row = " | ".join(str(header).ljust(col_widths[i]) for i, header in enumerate(headers))
    lines.append(header_row)
    lines.append("-" * len(header_row))
    
    # Data rows
    for row in rows:
        row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row) if i < len(col_widths))
        lines.append(row_str)
    
    return "\n".join(lines)


def format_process_table(processes: List[Dict[str, Any]], title: str = "Top Processes by Memory") -> str:
    """
    Format process list as a rich table.
    
    Args:
        processes: List of process dicts with 'name', 'pid', 'memory_mb'
        title: Table title
        
    Returns:
        Formatted table string
    """
    if not processes:
        return format_message(f"{title}\n(No processes found)", success=False)
    
    rows = []
    for i, proc in enumerate(processes[:10], 1):
        name = proc.get('name', 'N/A')
        pid = proc.get('pid', 'N/A')
        memory_mb = proc.get('memory_mb', 0)
        memory_str = f"{memory_mb:.1f} MB"
        
        rows.append([str(i), str(name), memory_str, str(pid)])
    
    headers = ["#", "Process Name", "Memory", "PID"]
    return format_table_with_rich(rows, headers, title)


def format_file_table(files: List[Dict[str, Any]], title: str = "Recent Files") -> str:
    """
    Format file list as a rich table.
    
    Args:
        files: List of file dicts with 'name', 'size_mb', 'modified'
        title: Table title
        
    Returns:
        Formatted table string
    """
    if not files:
        return format_message(f"{title}\n(No files found)", success=True)
    
    from datetime import datetime
    
    rows = []
    for i, file_info in enumerate(files[:10], 1):
        name = file_info.get('name', 'N/A')
        size_mb = file_info.get('size_mb', 0)
        
        # Format size
        if size_mb >= 0.01:
            size_str = f"{size_mb:.2f} MB"
        else:
            size_str = f"{size_mb * 1024:.0f} KB"
        
        # Format modified time
        modified = file_info.get('modified', 0)
        if modified:
            mod_time = datetime.fromtimestamp(modified)
            time_str = mod_time.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = "N/A"
        
        rows.append([str(i), str(name), size_str, time_str])
    
    headers = ["#", "File Name", "Size", "Modified"]
    return format_table_with_rich(rows, headers, title)
