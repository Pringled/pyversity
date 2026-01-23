"""
CLI entry point for benchmarks.

Usage:
    python -m benchmarks download   # Download datasets
    python -m benchmarks run        # Run benchmarks on all datasets
    python -m benchmarks report     # Generate markdown report + plots
    python -m benchmarks            # Run all (default)
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from benchmarks.core import (
    DATASET_REGISTRY,
    BenchmarkConfig,
    generate_report,
    run_benchmark,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmarks/results")
DATA_DIR = Path("local/data")


def cmd_download() -> None:
    """Download datasets from original sources."""
    logger.info("Downloading benchmark datasets...")

    for name, info in DATASET_REGISTRY.items():
        if info.download_url is None:
            continue  # HuggingFace datasets auto-download

        dest_dir = Path(info.path)

        if dest_dir.exists() and any(dest_dir.iterdir()):
            logger.debug(f"{name} already exists, skipping")
            continue

        logger.info(f"Downloading {name}...")
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            zip_path = DATA_DIR / f"{name}.zip"

            # Download
            response = requests.get(info.download_url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)
            zip_path.unlink()

            # Rename extracted directory (zip may extract to different name)
            for extracted_dir in DATA_DIR.iterdir():
                if extracted_dir.is_dir() and extracted_dir != dest_dir:
                    # Match by dataset name or known aliases
                    dir_name_lower = extracted_dir.name.lower()
                    if name in dir_name_lower or (name == "lastfm" and "hetrec" in dir_name_lower):
                        extracted_dir.rename(dest_dir)
                        break

            logger.info(f"Saved {name} to {dest_dir}")

        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

    logger.info("Download complete")


def cmd_run() -> None:
    """Run benchmarks on all datasets."""
    logger.info("Running benchmark suite...")

    for name, info in DATASET_REGISTRY.items():
        logger.info(f"Benchmarking {name}...")

        config = BenchmarkConfig(
            dataset=name,
            rating_threshold=info.rating_threshold,
        )

        try:
            run_benchmark(config)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {name}: {e} (run: python -m benchmarks download)")
        except Exception as e:
            logger.error(f"Error on {name}: {e}")

    logger.info("Benchmark complete")


def cmd_report() -> None:
    """Generate markdown report and plots."""
    generate_report(RESULTS_DIR)


def cmd_latency() -> None:
    """Run latency benchmark and generate plot."""
    from benchmarks.core.report import generate_latency_plot

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_latency_plot(RESULTS_DIR / "latency.png")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pyversity Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["download", "run", "report", "latency"],
        default=None,
        help="Command to run (default: run)",
    )

    args = parser.parse_args()

    if args.command == "download":
        cmd_download()
    elif args.command == "run":
        cmd_run()
    elif args.command == "report":
        cmd_report()
    elif args.command == "latency":
        cmd_latency()
    else:
        # Default: run
        cmd_run()


if __name__ == "__main__":
    main()
