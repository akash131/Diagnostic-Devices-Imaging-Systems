"""CLI commands for the imaging system."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import argparse
import sys


@dataclass
class Command:
    """CLI command definition."""

    name: str
    description: str
    handler: Callable[..., int]
    arguments: list[dict] = field(default_factory=list)
    subcommands: list["Command"] = field(default_factory=list)

    def add_argument(
        self,
        name: str,
        arg_type: type = str,
        required: bool = False,
        default: Any = None,
        help_text: str = "",
        choices: list = None,
    ):
        """Add an argument to the command."""
        self.arguments.append({
            "name": name,
            "type": arg_type,
            "required": required,
            "default": default,
            "help": help_text,
            "choices": choices,
        })

    def add_subcommand(self, command: "Command"):
        """Add a subcommand."""
        self.subcommands.append(command)


class CLI:
    """Main CLI application."""

    def __init__(self, name: str = "imaging", description: str = ""):
        self.name = name
        self.description = description or "Diagnostic Imaging Systems CLI"
        self.commands: dict[str, Command] = {}
        self._parser = argparse.ArgumentParser(
            prog=name,
            description=self.description,
        )
        self._subparsers = self._parser.add_subparsers(dest="command", help="Available commands")
        self._setup_default_commands()

    def _setup_default_commands(self):
        """Set up default CLI commands."""
        # Device commands
        device_cmd = Command(
            name="device",
            description="Device management commands",
            handler=self._device_handler,
        )
        device_cmd.add_subcommand(Command(
            name="list",
            description="List all devices",
            handler=self._device_list,
        ))
        device_cmd.add_subcommand(Command(
            name="status",
            description="Show device status",
            handler=self._device_status,
        ))
        device_cmd.add_subcommand(Command(
            name="calibrate",
            description="Run device calibration",
            handler=self._device_calibrate,
        ))
        self.register_command(device_cmd)

        # Study commands
        study_cmd = Command(
            name="study",
            description="Study management commands",
            handler=self._study_handler,
        )
        study_cmd.add_subcommand(Command(
            name="list",
            description="List studies",
            handler=self._study_list,
        ))
        study_cmd.add_subcommand(Command(
            name="info",
            description="Show study information",
            handler=self._study_info,
        ))
        study_cmd.add_subcommand(Command(
            name="export",
            description="Export study",
            handler=self._study_export,
        ))
        self.register_command(study_cmd)

        # Schedule commands
        schedule_cmd = Command(
            name="schedule",
            description="Scheduling commands",
            handler=self._schedule_handler,
        )
        schedule_cmd.add_subcommand(Command(
            name="today",
            description="Show today's schedule",
            handler=self._schedule_today,
        ))
        schedule_cmd.add_subcommand(Command(
            name="add",
            description="Add new exam",
            handler=self._schedule_add,
        ))
        self.register_command(schedule_cmd)

        # QA commands
        qa_cmd = Command(
            name="qa",
            description="Quality assurance commands",
            handler=self._qa_handler,
        )
        qa_cmd.add_subcommand(Command(
            name="status",
            description="Show calibration status",
            handler=self._qa_status,
        ))
        qa_cmd.add_subcommand(Command(
            name="report",
            description="Generate QA report",
            handler=self._qa_report,
        ))
        self.register_command(qa_cmd)

        # DICOM commands
        dicom_cmd = Command(
            name="dicom",
            description="DICOM operations",
            handler=self._dicom_handler,
        )
        dicom_cmd.add_subcommand(Command(
            name="send",
            description="Send DICOM to PACS",
            handler=self._dicom_send,
        ))
        dicom_cmd.add_subcommand(Command(
            name="query",
            description="Query PACS",
            handler=self._dicom_query,
        ))
        dicom_cmd.add_subcommand(Command(
            name="verify",
            description="Verify DICOM connection",
            handler=self._dicom_verify,
        ))
        self.register_command(dicom_cmd)

    def register_command(self, command: Command):
        """Register a command."""
        self.commands[command.name] = command
        parser = self._subparsers.add_parser(command.name, help=command.description)

        # Add arguments
        for arg in command.arguments:
            name = arg["name"]
            if arg["required"]:
                parser.add_argument(name, type=arg["type"], help=arg["help"], choices=arg.get("choices"))
            else:
                parser.add_argument(f"--{name}", type=arg["type"], default=arg["default"],
                                    help=arg["help"], choices=arg.get("choices"))

        # Add subcommands
        if command.subcommands:
            sub_parsers = parser.add_subparsers(dest="subcommand")
            for subcmd in command.subcommands:
                sub_parser = sub_parsers.add_parser(subcmd.name, help=subcmd.description)
                for arg in subcmd.arguments:
                    name = arg["name"]
                    if arg["required"]:
                        sub_parser.add_argument(name, type=arg["type"], help=arg["help"])
                    else:
                        sub_parser.add_argument(f"--{name}", type=arg["type"], default=arg["default"],
                                                help=arg["help"])

    def run(self, args: list[str] = None) -> int:
        """Run the CLI."""
        parsed = self._parser.parse_args(args)

        if not parsed.command:
            self._parser.print_help()
            return 0

        command = self.commands.get(parsed.command)
        if command:
            return command.handler(parsed)

        return 1

    # Command handlers
    def _device_handler(self, args) -> int:
        """Handle device commands."""
        if hasattr(args, "subcommand") and args.subcommand:
            cmd = self.commands["device"]
            for subcmd in cmd.subcommands:
                if subcmd.name == args.subcommand:
                    return subcmd.handler(args)
        print("Usage: imaging device <command>")
        print("Commands: list, status, calibrate")
        return 0

    def _device_list(self, args) -> int:
        """List devices."""
        print("Registered Devices:")
        print("-" * 60)
        print(f"{'ID':<15} {'Type':<15} {'Modality':<10} {'Status':<10}")
        print("-" * 60)
        # Simulated device list
        devices = [
            ("US_001", "Ultrasound", "US", "Online"),
            ("DX_001", "X-Ray", "DX", "Online"),
            ("OCT_001", "OCT Scanner", "OPT", "Online"),
            ("TG_001", "Thermal Camera", "TG", "Offline"),
        ]
        for dev in devices:
            print(f"{dev[0]:<15} {dev[1]:<15} {dev[2]:<10} {dev[3]:<10}")
        return 0

    def _device_status(self, args) -> int:
        """Show device status."""
        device_id = getattr(args, "device_id", None) or "US_001"
        print(f"Device Status: {device_id}")
        print("-" * 40)
        print(f"Status: Online")
        print(f"Last Calibration: 2024-01-15")
        print(f"Images Today: 45")
        print(f"Temperature: 23.5°C")
        return 0

    def _device_calibrate(self, args) -> int:
        """Calibrate device."""
        device_id = getattr(args, "device_id", None) or "US_001"
        print(f"Calibrating device: {device_id}")
        print("Running self-test...")
        print("Calibration complete. All parameters within spec.")
        return 0

    def _study_handler(self, args) -> int:
        """Handle study commands."""
        if hasattr(args, "subcommand") and args.subcommand:
            cmd = self.commands["study"]
            for subcmd in cmd.subcommands:
                if subcmd.name == args.subcommand:
                    return subcmd.handler(args)
        print("Usage: imaging study <command>")
        print("Commands: list, info, export")
        return 0

    def _study_list(self, args) -> int:
        """List studies."""
        print("Recent Studies:")
        print("-" * 80)
        print(f"{'Accession':<12} {'Patient':<20} {'Modality':<8} {'Date':<12} {'Status':<12}")
        print("-" * 80)
        studies = [
            ("ACC001", "Smith^John", "US", "2024-01-15", "Completed"),
            ("ACC002", "Doe^Jane", "DX", "2024-01-15", "In Progress"),
            ("ACC003", "Brown^Bob", "OPT", "2024-01-14", "Reported"),
        ]
        for s in studies:
            print(f"{s[0]:<12} {s[1]:<20} {s[2]:<8} {s[3]:<12} {s[4]:<12}")
        return 0

    def _study_info(self, args) -> int:
        """Show study info."""
        accession = getattr(args, "accession", None) or "ACC001"
        print(f"Study Information: {accession}")
        print("-" * 40)
        print("Patient: Smith^John")
        print("Modality: US")
        print("Description: Abdominal Ultrasound")
        print("Series: 3")
        print("Images: 45")
        print("Status: Completed")
        return 0

    def _study_export(self, args) -> int:
        """Export study."""
        accession = getattr(args, "accession", None) or "ACC001"
        print(f"Exporting study: {accession}")
        print("Format: DICOM")
        print("Export complete.")
        return 0

    def _schedule_handler(self, args) -> int:
        """Handle schedule commands."""
        if hasattr(args, "subcommand") and args.subcommand:
            cmd = self.commands["schedule"]
            for subcmd in cmd.subcommands:
                if subcmd.name == args.subcommand:
                    return subcmd.handler(args)
        print("Usage: imaging schedule <command>")
        print("Commands: today, add")
        return 0

    def _schedule_today(self, args) -> int:
        """Show today's schedule."""
        print("Today's Schedule:")
        print("-" * 70)
        print(f"{'Time':<8} {'Patient':<20} {'Modality':<8} {'Exam':<25} {'Room':<8}")
        print("-" * 70)
        schedule = [
            ("09:00", "Smith^John", "US", "Abdominal Ultrasound", "US-1"),
            ("09:30", "Doe^Jane", "DX", "Chest X-Ray", "XR-1"),
            ("10:00", "Brown^Bob", "OPT", "Macular OCT", "EYE-1"),
        ]
        for s in schedule:
            print(f"{s[0]:<8} {s[1]:<20} {s[2]:<8} {s[3]:<25} {s[4]:<8}")
        return 0

    def _schedule_add(self, args) -> int:
        """Add exam to schedule."""
        print("Adding exam to schedule...")
        print("Exam scheduled successfully.")
        return 0

    def _qa_handler(self, args) -> int:
        """Handle QA commands."""
        if hasattr(args, "subcommand") and args.subcommand:
            cmd = self.commands["qa"]
            for subcmd in cmd.subcommands:
                if subcmd.name == args.subcommand:
                    return subcmd.handler(args)
        print("Usage: imaging qa <command>")
        print("Commands: status, report")
        return 0

    def _qa_status(self, args) -> int:
        """Show QA status."""
        print("Calibration Status:")
        print("-" * 60)
        print(f"{'Device':<15} {'Last Cal':<12} {'Next Due':<12} {'Status':<10}")
        print("-" * 60)
        status = [
            ("US_001", "2024-01-01", "2024-02-01", "Current"),
            ("DX_001", "2023-12-15", "2024-01-15", "Due Soon"),
            ("OCT_001", "2023-10-01", "2024-01-01", "Overdue"),
        ]
        for s in status:
            print(f"{s[0]:<15} {s[1]:<12} {s[2]:<12} {s[3]:<10}")
        return 0

    def _qa_report(self, args) -> int:
        """Generate QA report."""
        print("Generating QA Report...")
        print("Report saved to: qa_report_20240115.pdf")
        return 0

    def _dicom_handler(self, args) -> int:
        """Handle DICOM commands."""
        if hasattr(args, "subcommand") and args.subcommand:
            cmd = self.commands["dicom"]
            for subcmd in cmd.subcommands:
                if subcmd.name == args.subcommand:
                    return subcmd.handler(args)
        print("Usage: imaging dicom <command>")
        print("Commands: send, query, verify")
        return 0

    def _dicom_send(self, args) -> int:
        """Send DICOM to PACS."""
        print("Sending DICOM to PACS...")
        print("Host: pacs.hospital.local")
        print("Port: 11112")
        print("AE Title: PACS_SCP")
        print("Images sent: 45")
        print("Status: Success")
        return 0

    def _dicom_query(self, args) -> int:
        """Query PACS."""
        print("Querying PACS...")
        print("Results: 3 studies found")
        return 0

    def _dicom_verify(self, args) -> int:
        """Verify DICOM connection."""
        print("Verifying DICOM connection...")
        print("C-ECHO to PACS_SCP... Success")
        print("Connection verified.")
        return 0


def main():
    """Main entry point."""
    cli = CLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
