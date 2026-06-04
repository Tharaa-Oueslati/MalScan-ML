#!/usr/bin/env python3
"""
scan.py
=======
MalScan-ML — Static Malware Detector
CLI entry point.

Usage:
    python scan.py suspicious.exe
    python scan.py suspicious.exe --model models/xgboost_model.pkl
    python scan.py --batch ./folder_of_files/
    python scan.py suspicious.exe --json
"""

import sys
import json
import time
from pathlib import Path

import click
from colorama import init, Fore, Back, Style

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import MalwarePredictor

init(autoreset=True)  # colorama


BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════╗
║  {Fore.WHITE}███╗   ███╗ █████╗ ██╗     {Fore.CYAN}              ║
║  {Fore.WHITE}████╗ ████║██╔══██╗██║     {Fore.CYAN}              ║
║  {Fore.WHITE}██╔████╔██║███████║██║     {Fore.CYAN}   SCAN       ║
║  {Fore.WHITE}██║╚██╔╝██║██╔══██║██║     {Fore.CYAN}    ML        ║
║  {Fore.WHITE}██║ ╚═╝ ██║██║  ██║███████╗{Fore.CYAN}              ║
║  {Fore.WHITE}╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝{Fore.CYAN}              ║
╠══════════════════════════════════════════╣
║  {Fore.YELLOW}Static Malware Detector  v1.0{Fore.CYAN}            ║
║  {Fore.WHITE}PE Analysis · ML Classification{Fore.CYAN}           ║
╚══════════════════════════════════════════╝{Style.RESET_ALL}
"""


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print prediction result to console."""
    verdict = result["verdict"]
    confidence = result["confidence"]
    filepath = Path(result["filepath"]).name

    print()

    if verdict == "ERROR":
        print(f"{Fore.RED}[ERROR] {result['error']}{Style.RESET_ALL}")
        return

    # Verdict banner
    if verdict == "MALWARE":
        color = Fore.RED
        icon = "⚠️ "
        bg = Back.RED + Fore.WHITE
    else:
        color = Fore.GREEN
        icon = "✅"
        bg = Back.GREEN + Fore.WHITE

    # Confidence bar (20 chars wide)
    filled = int(confidence / 5)
    bar = "█" * filled + "░" * (20 - filled)

    print(f"{color}┌{'─'*44}┐")
    print(f"│  File:  {Style.RESET_ALL}{filepath:<35}{color}│")
    print(f"│  {bg} VERDICT: {verdict} {Style.RESET_ALL}{color}  Confidence: {confidence:.1f}%  │")
    print(f"│  [{bar}]  {color}│")
    print(f"│  Model: {Style.RESET_ALL}{result.get('model_name', 'unknown'):<35}{color}│")
    print(f"├{'─'*44}┤")

    # Risk indicators
    indicators = result.get("risk_indicators", [])
    print(f"│  {Fore.YELLOW}Risk Indicators:{Style.RESET_ALL}{color}                           │")
    for ind in indicators:
        truncated = ind[:40] + "…" if len(ind) > 40 else ind
        print(f"│  {Fore.WHITE}• {truncated:<42}{color}│")

    print(f"└{'─'*44}┘{Style.RESET_ALL}")

    # Verbose: print all features
    if verbose and result.get("features"):
        print(f"\n{Fore.CYAN}Feature Values:{Style.RESET_ALL}")
        features = result["features"]
        for key, val in sorted(features.items()):
            if key not in ("filename", "label"):
                if isinstance(val, float):
                    print(f"  {key:<40} {val:.4f}")
                else:
                    print(f"  {key:<40} {val}")


@click.command()
@click.argument("target", required=False, default=None)
@click.option("--model", "-m", default=None,
              help="Path to model .pkl file (default: models/best_model.pkl)")
@click.option("--batch", "-b", default=None,
              help="Scan all PE files in a directory")
@click.option("--json-output", "-j", "json_output", is_flag=True, default=False,
              help="Output results as JSON")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show all extracted features")
@click.option("--threshold", "-t", default=50.0,
              help="Malware confidence threshold % (default: 50.0)")
@click.option("--no-banner", is_flag=True, default=False,
              help="Suppress banner")
def main(target, model, batch, json_output, verbose, threshold, no_banner):
    """
    MalScan-ML: Static PE malware detection using Machine Learning.

    \b
    Examples:
        python scan.py suspicious.exe
        python scan.py suspicious.exe --verbose
        python scan.py --batch ./pe_files/ --json-output
        python scan.py suspicious.exe --threshold 70
    """
    if not no_banner and not json_output:
        print(BANNER)

    # Load model
    try:
        predictor = MalwarePredictor(model_path=model)
    except FileNotFoundError as e:
        print(f"{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Run training first:{Style.RESET_ALL}")
        print("  python src/train.py --data data/processed/features.csv")
        sys.exit(1)

    # Determine files to scan
    if batch:
        batch_dir = Path(batch)
        if not batch_dir.is_dir():
            print(f"{Fore.RED}[ERROR] Not a directory: {batch}{Style.RESET_ALL}")
            sys.exit(1)
        files = list(batch_dir.glob("*.exe")) + list(batch_dir.glob("*.dll"))
        if not files:
            files = [f for f in batch_dir.iterdir() if f.is_file()]
        print(f"{Fore.CYAN}[*] Scanning {len(files)} files in {batch_dir}{Style.RESET_ALL}\n")
    elif target:
        files = [Path(target)]
    else:
        print(f"{Fore.RED}[ERROR] Provide a file or --batch directory.{Style.RESET_ALL}")
        ctx = click.get_current_context()
        print(ctx.get_help())
        sys.exit(1)

    # Run predictions
    all_results = []
    malware_count = 0

    for filepath in files:
        if not json_output:
            print(f"{Fore.CYAN}[*] Analyzing: {filepath.name}...{Style.RESET_ALL}")

        t0 = time.time()
        result = predictor.predict(str(filepath))
        elapsed = time.time() - t0
        result["elapsed_ms"] = round(elapsed * 1000, 1)

        # Apply custom threshold
        if result["verdict"] != "ERROR":
            if result["confidence"] >= threshold and result["label"] == 1:
                result["verdict"] = "MALWARE"
            elif result["confidence"] < threshold and result["label"] == 1:
                result["verdict"] = "SUSPICIOUS"

        if result["verdict"] in ("MALWARE", "SUSPICIOUS"):
            malware_count += 1

        all_results.append(result)

        if not json_output:
            print_result(result, verbose=verbose)
            print(f"  {Fore.WHITE}⏱  Scan time: {elapsed*1000:.0f}ms{Style.RESET_ALL}")

    # JSON output mode
    if json_output:
        print(json.dumps(all_results, indent=2, default=str))
        return

    # Batch summary
    if batch and len(files) > 1:
        clean = len(files) - malware_count
        print(f"\n{Fore.CYAN}{'═'*46}")
        print(f"  SCAN SUMMARY")
        print(f"{'═'*46}{Style.RESET_ALL}")
        print(f"  Files scanned:  {len(files)}")
        print(f"  {Fore.RED}Malware found:  {malware_count}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Clean files:    {clean}{Style.RESET_ALL}")
        rate = malware_count / len(files) * 100
        print(f"  Detection rate: {rate:.1f}%")
        print(f"{Fore.CYAN}{'═'*46}{Style.RESET_ALL}\n")

    # Exit code: 1 if any malware found (useful in CI/scripts)
    sys.exit(1 if malware_count > 0 else 0)


if __name__ == "__main__":
    main()
