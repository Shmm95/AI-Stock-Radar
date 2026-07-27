"""Command-line runner for the OpenAI Radar Analyst."""

from src.agents.analyst_agent import (
    generate_ai_report,
    get_ai_report_path,
)


def main() -> None:
    """Generate and print the latest AI radar report."""

    print("=" * 80)
    print("OPENAI RADAR ANALYST")
    print("=" * 80)
    print("Reading data/reports/radar_latest.csv...")
    print()

    try:
        report = generate_ai_report(
            top_n=5,
        )
    except Exception as error:
        print(f"AI report generation failed: {error}")
        raise SystemExit(1) from error

    print(report)

    print()
    print("=" * 80)
    print(f"Report saved to: {get_ai_report_path()}")
    print("=" * 80)


if __name__ == "__main__":
    main()