"""Report generation utilities."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from .report import Report, ReportStatus


class ReportGenerator(ABC):
    """Abstract report generator."""

    @abstractmethod
    def generate(self, report: Report) -> str:
        """Generate report content."""
        pass

    @abstractmethod
    def save(self, report: Report, output_path: str):
        """Save report to file."""
        pass


class TextReportGenerator(ReportGenerator):
    """Plain text report generator."""

    def generate(self, report: Report) -> str:
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(f"IMAGING REPORT - {report.modality}")
        lines.append("=" * 80)
        lines.append("")

        # Patient info
        lines.append(f"Patient: {report.patient_name}")
        lines.append(f"Patient ID: {report.patient_id}")
        lines.append(f"Study Date: {report.study_date}")
        lines.append(f"Study: {report.study_description}")
        lines.append("")

        # Clinical history
        if report.clinical_history:
            lines.append("CLINICAL HISTORY:")
            lines.append(report.clinical_history)
            lines.append("")

        # Technique
        if report.technique:
            lines.append("TECHNIQUE:")
            lines.append(report.technique)
            lines.append("")

        # Comparison
        if report.comparison:
            lines.append("COMPARISON:")
            lines.append(report.comparison)
            lines.append("")

        # Findings
        lines.append("FINDINGS:")
        lines.append("-" * 40)
        for section in report.sections:
            lines.append(f"\n{section.title}:")
            if section.content:
                lines.append(section.content)
            for finding in section.findings:
                severity = f" [{finding.severity.value.upper()}]" if finding.severity.value != "normal" else ""
                lines.append(f"  • {finding.description}{severity}")
                if finding.location:
                    lines.append(f"    Location: {finding.location}")
                for measurement in finding.measurements:
                    lines.append(f"    {measurement.get('name', '')}: {measurement.get('value', '')} {measurement.get('unit', '')}")

        # Impression
        lines.append("")
        lines.append("IMPRESSION:")
        lines.append("-" * 40)
        lines.append(report.impression if report.impression else "See findings above.")

        # Recommendation
        if report.recommendation:
            lines.append("")
            lines.append("RECOMMENDATION:")
            lines.append(report.recommendation)

        # Footer
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"Reporting Physician: {report.reporting_physician}")
        lines.append(f"Report Status: {report.status.value.upper()}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def save(self, report: Report, output_path: str):
        content = self.generate(report)
        with open(output_path, "w") as f:
            f.write(content)


class HTMLReportGenerator(ReportGenerator):
    """HTML report generator."""

    def generate(self, report: Report) -> str:
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Imaging Report - {report.patient_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; }}
        .section-title {{ font-weight: bold; color: #2c3e50; border-bottom: 1px solid #ddd; }}
        .finding {{ margin: 10px 0 10px 20px; }}
        .severity-mild {{ color: #f39c12; }}
        .severity-moderate {{ color: #e67e22; }}
        .severity-severe {{ color: #e74c3c; }}
        .severity-critical {{ color: #c0392b; font-weight: bold; }}
        .impression {{ background: #ecf0f1; padding: 15px; margin: 20px 0; }}
        .footer {{ border-top: 1px solid #333; padding-top: 10px; font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Imaging Report</h1>
        <table>
            <tr><th>Patient Name</th><td>{report.patient_name}</td><th>Patient ID</th><td>{report.patient_id}</td></tr>
            <tr><th>Study Date</th><td>{report.study_date}</td><th>Modality</th><td>{report.modality}</td></tr>
            <tr><th>Study</th><td colspan="3">{report.study_description}</td></tr>
        </table>
    </div>
"""

        if report.clinical_history:
            html += f"""
    <div class="section">
        <h2 class="section-title">Clinical History</h2>
        <p>{report.clinical_history}</p>
    </div>
"""

        if report.technique:
            html += f"""
    <div class="section">
        <h2 class="section-title">Technique</h2>
        <p>{report.technique}</p>
    </div>
"""

        html += """
    <div class="section">
        <h2 class="section-title">Findings</h2>
"""

        for section in report.sections:
            html += f"        <h3>{section.title}</h3>\n"
            if section.content:
                html += f"        <p>{section.content}</p>\n"

            for finding in section.findings:
                severity_class = f"severity-{finding.severity.value}" if finding.severity.value != "normal" else ""
                html += f'        <div class="finding {severity_class}">\n'
                html += f"            <strong>{finding.description}</strong>\n"
                if finding.location:
                    html += f"            <br>Location: {finding.location}\n"
                if finding.measurements:
                    html += "            <ul>\n"
                    for m in finding.measurements:
                        html += f"                <li>{m.get('name', '')}: {m.get('value', '')} {m.get('unit', '')}</li>\n"
                    html += "            </ul>\n"
                html += "        </div>\n"

        html += "    </div>\n"

        html += f"""
    <div class="impression">
        <h2>Impression</h2>
        <p>{report.impression if report.impression else 'See findings above.'}</p>
    </div>
"""

        if report.recommendation:
            html += f"""
    <div class="section">
        <h2 class="section-title">Recommendation</h2>
        <p>{report.recommendation}</p>
    </div>
"""

        html += f"""
    <div class="footer">
        <p>Reporting Physician: {report.reporting_physician}</p>
        <p>Report Status: {report.status.value.upper()}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html

    def save(self, report: Report, output_path: str):
        content = self.generate(report)
        with open(output_path, "w") as f:
            f.write(content)


class PDFReportGenerator(ReportGenerator):
    """PDF report generator (requires reportlab)."""

    def generate(self, report: Report) -> bytes:
        """Generate PDF content as bytes."""
        # Simplified implementation without reportlab
        # In production, use reportlab or weasyprint
        text_gen = TextReportGenerator()
        text_content = text_gen.generate(report)
        return text_content.encode("utf-8")

    def save(self, report: Report, output_path: str):
        content = self.generate(report)
        # For actual PDF, would use reportlab here
        # Fallback to text file with .pdf extension note
        with open(output_path, "wb") as f:
            f.write(content)


class JSONReportGenerator(ReportGenerator):
    """JSON report generator."""

    def generate(self, report: Report) -> str:
        return json.dumps(report.to_dict(), indent=2, default=str)

    def save(self, report: Report, output_path: str):
        content = self.generate(report)
        with open(output_path, "w") as f:
            f.write(content)


class ReportExporter:
    """Export reports to various formats."""

    GENERATORS = {
        "text": TextReportGenerator,
        "txt": TextReportGenerator,
        "html": HTMLReportGenerator,
        "pdf": PDFReportGenerator,
        "json": JSONReportGenerator,
    }

    @classmethod
    def export(
        cls,
        report: Report,
        output_path: str,
        format: str = "text",
    ) -> bool:
        """Export report to specified format."""
        generator_class = cls.GENERATORS.get(format.lower())
        if not generator_class:
            raise ValueError(f"Unknown format: {format}")

        generator = generator_class()
        generator.save(report, output_path)
        return True

    @classmethod
    def export_all_formats(
        cls,
        report: Report,
        output_dir: str,
        base_name: str = "report",
    ) -> dict[str, str]:
        """Export report to all supported formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for format_name, generator_class in cls.GENERATORS.items():
            if format_name in ("txt",):  # Skip aliases
                continue

            output_path = output_dir / f"{base_name}.{format_name}"
            generator = generator_class()
            generator.save(report, str(output_path))
            paths[format_name] = str(output_path)

        return paths
