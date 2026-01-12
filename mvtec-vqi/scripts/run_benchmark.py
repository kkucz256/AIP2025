import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
# Assuming script is run from project root
current_dir = Path(os.getcwd())
if (current_dir / "src").exists():
    sys.path.append(str(current_dir))
else:
    # Fallback if run from scripts/
    sys.path.append(str(current_dir.parent))

try:
    from src.utils.common import load_config
    from src.benchmark.runner import BenchmarkRunner
    from src.benchmark.report import ReportGenerator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please run this script from the project root directory: python scripts/run_benchmark.py")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Benchmark and Generate Report")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--backends", nargs="+", help="List of backends to benchmark")
    parser.add_argument("--categories", nargs="+", help="List of categories to benchmark (default: all)")
    parser.add_argument("--output-dir", default="reports/benchmark", help="Directory for the report")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Default backends if not specified
    if not args.backends:
        # Check what is likely available or default to known ones
        # We try to include what we saw in the file structure
        args.backends = ["padim_resnet50", "padim_wide_resnet50_2", "cae"]
        print(f"No backends specified, using defaults: {args.backends}")

    runner = BenchmarkRunner(
        config=config, 
        backends=args.backends, 
        categories=args.categories
    )
    
    print("Starting benchmark...")
    print(f"Backends: {args.backends}")
    print(f"Categories: {runner.categories}")
    
    results = runner.run()
    
    print("Generating report...")
    reporter = ReportGenerator(args.output_dir)
    reporter.generate(results)
    
    print("Done.")

if __name__ == "__main__":
    main()
