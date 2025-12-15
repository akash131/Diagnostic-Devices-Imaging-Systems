"""Display utilities for CLI output."""

from dataclasses import dataclass, field
from typing import Optional, Any
import sys
import time


class Colors:
    """ANSI color codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class Display:
    """Display utilities."""

    def __init__(self, use_colors: bool = True, width: int = 80):
        self.use_colors = use_colors and sys.stdout.isatty()
        self.width = width

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def print(self, message: str, color: str = None, end: str = "\n"):
        """Print a message with optional color."""
        if color and self.use_colors:
            message = self._colorize(message, color)
        print(message, end=end)

    def success(self, message: str):
        """Print success message."""
        self.print(f"[OK] {message}", Colors.GREEN)

    def error(self, message: str):
        """Print error message."""
        self.print(f"[ERROR] {message}", Colors.RED)

    def warning(self, message: str):
        """Print warning message."""
        self.print(f"[WARN] {message}", Colors.YELLOW)

    def info(self, message: str):
        """Print info message."""
        self.print(f"[INFO] {message}", Colors.CYAN)

    def header(self, text: str, char: str = "="):
        """Print a header."""
        border = char * self.width
        self.print(border, Colors.BOLD)
        self.print(f" {text}", Colors.BOLD)
        self.print(border, Colors.BOLD)

    def subheader(self, text: str, char: str = "-"):
        """Print a subheader."""
        self.print(f"{text}", Colors.BOLD)
        self.print(char * len(text))

    def divider(self, char: str = "-"):
        """Print a divider line."""
        self.print(char * self.width, Colors.DIM)

    def blank(self, count: int = 1):
        """Print blank lines."""
        for _ in range(count):
            print()

    def key_value(self, key: str, value: Any, key_width: int = 20):
        """Print a key-value pair."""
        key_str = f"{key}:".ljust(key_width)
        self.print(f"{self._colorize(key_str, Colors.DIM)} {value}")

    def status(self, label: str, status: str, is_ok: bool = True):
        """Print a status line."""
        color = Colors.GREEN if is_ok else Colors.RED
        self.print(f"{label.ljust(30)} [{self._colorize(status, color)}]")

    def bullet(self, text: str, indent: int = 0):
        """Print a bullet point."""
        prefix = " " * indent + "• "
        self.print(f"{prefix}{text}")


@dataclass
class Column:
    """Table column definition."""

    name: str
    width: int = 0
    align: str = "left"  # left, right, center
    color: str = ""

    def format_value(self, value: Any) -> str:
        """Format a value for this column."""
        text = str(value)
        if self.width:
            if self.align == "right":
                text = text.rjust(self.width)
            elif self.align == "center":
                text = text.center(self.width)
            else:
                text = text.ljust(self.width)
        return text


class TableDisplay:
    """Display tabular data."""

    def __init__(
        self,
        columns: list[Column] = None,
        use_colors: bool = True,
        border: str = "|",
        header_border: str = "-",
    ):
        self.columns = columns or []
        self.use_colors = use_colors and sys.stdout.isatty()
        self.border = border
        self.header_border = header_border
        self._display = Display(use_colors)

    def add_column(
        self,
        name: str,
        width: int = 0,
        align: str = "left",
        color: str = "",
    ):
        """Add a column."""
        self.columns.append(Column(name, width, align, color))

    def auto_width(self, data: list[dict]):
        """Auto-calculate column widths from data."""
        for col in self.columns:
            col.width = max(col.width, len(col.name))
            for row in data:
                value = row.get(col.name, "")
                col.width = max(col.width, len(str(value)))

    def print_header(self):
        """Print table header."""
        header = self.border + " "
        for col in self.columns:
            header += col.format_value(col.name) + f" {self.border} "
        self._display.print(header.rstrip(), Colors.BOLD)

        # Print border line
        total_width = sum(c.width for c in self.columns) + len(self.columns) * 3 + 1
        self._display.print(self.header_border * total_width)

    def print_row(self, row: dict):
        """Print a table row."""
        line = self.border + " "
        for col in self.columns:
            value = row.get(col.name, "")
            formatted = col.format_value(value)
            if col.color and self.use_colors:
                formatted = f"{col.color}{formatted}{Colors.RESET}"
            line += formatted + f" {self.border} "
        print(line.rstrip())

    def print_footer(self):
        """Print table footer."""
        total_width = sum(c.width for c in self.columns) + len(self.columns) * 3 + 1
        self._display.print(self.header_border * total_width)

    def print_table(self, data: list[dict]):
        """Print complete table."""
        if not self.columns:
            if data:
                # Auto-create columns from first row
                for key in data[0].keys():
                    self.add_column(key)

        self.auto_width(data)
        self.print_header()
        for row in data:
            self.print_row(row)
        self.print_footer()
        self._display.print(f"Total: {len(data)} rows", Colors.DIM)


class ProgressDisplay:
    """Display progress bars and spinners."""

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    BAR_FILL = "█"
    BAR_EMPTY = "░"

    def __init__(self, use_colors: bool = True, width: int = 40):
        self.use_colors = use_colors and sys.stdout.isatty()
        self.width = width
        self._spinner_idx = 0

    def _colorize(self, text: str, color: str) -> str:
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def progress_bar(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        show_percent: bool = True,
    ) -> str:
        """Generate a progress bar string."""
        if total <= 0:
            percent = 0
            filled = 0
        else:
            percent = min(100, int(current / total * 100))
            filled = int(self.width * current / total)

        bar = self.BAR_FILL * filled + self.BAR_EMPTY * (self.width - filled)

        if show_percent:
            percent_str = f" {percent:3d}%"
        else:
            percent_str = ""

        line = f"{prefix}[{bar}]{percent_str} {suffix}"
        return line

    def print_progress(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        clear_line: bool = True,
    ):
        """Print a progress bar."""
        bar = self.progress_bar(current, total, prefix, suffix)
        if clear_line:
            print(f"\r{bar}", end="", flush=True)
        else:
            print(bar)

        if current >= total:
            print()  # Newline when complete

    def spinner(self, message: str = "") -> str:
        """Get next spinner character with message."""
        char = self.SPINNER_CHARS[self._spinner_idx]
        self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_CHARS)
        return f"{char} {message}"

    def print_spinner(self, message: str = ""):
        """Print spinner animation."""
        print(f"\r{self.spinner(message)}", end="", flush=True)


class StatusDisplay:
    """Display status information."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stdout.isatty()
        self._display = Display(use_colors)

    def device_status(
        self,
        device_id: str,
        device_type: str,
        status: str,
        details: dict = None,
    ):
        """Display device status."""
        is_online = status.lower() in ("online", "ready", "active", "ok")
        status_color = Colors.GREEN if is_online else Colors.RED

        self._display.subheader(f"Device: {device_id}")
        self._display.key_value("Type", device_type)
        self._display.key_value("Status", f"{status_color if self.use_colors else ''}{status}{Colors.RESET if self.use_colors else ''}")

        if details:
            self._display.blank()
            for key, value in details.items():
                self._display.key_value(key, value)

    def calibration_status(self, calibrations: list[dict]):
        """Display calibration status."""
        self._display.header("Calibration Status")

        table = TableDisplay()
        table.add_column("Device", width=15)
        table.add_column("Last Cal", width=12)
        table.add_column("Next Due", width=12)
        table.add_column("Status", width=10)

        for cal in calibrations:
            status = cal.get("status", "").lower()
            if status == "current":
                cal["Status"] = f"{Colors.GREEN}Current{Colors.RESET}" if self.use_colors else "Current"
            elif status == "due_soon":
                cal["Status"] = f"{Colors.YELLOW}Due Soon{Colors.RESET}" if self.use_colors else "Due Soon"
            elif status == "overdue":
                cal["Status"] = f"{Colors.RED}OVERDUE{Colors.RESET}" if self.use_colors else "OVERDUE"

        table.print_table(calibrations)

    def study_summary(
        self,
        study_uid: str,
        patient_name: str,
        modality: str,
        series_count: int,
        image_count: int,
        status: str,
    ):
        """Display study summary."""
        self._display.subheader(f"Study: {study_uid}")
        self._display.key_value("Patient", patient_name)
        self._display.key_value("Modality", modality)
        self._display.key_value("Series", series_count)
        self._display.key_value("Images", image_count)
        self._display.key_value("Status", status)
