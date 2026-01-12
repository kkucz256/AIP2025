import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def generate(self, results):
        self.results = [r for r in results if "error" not in r]
        self.errors = [r for r in results if "error" in r]
        
        if not self.results:
            print("No successful results to report.")
            # Generate error report at least
            if self.errors:
                 self._generate_html()
            return

        self._plot_auroc_comparison()
        self._plot_advanced_metrics()
        self._plot_inference_time()
        
        html_content = self._generate_html()
        report_path = self.output_dir / "benchmark_report.html"
        report_path.write_text(html_content, encoding="utf-8")
        print(f"Report generated at {report_path}")

    def _plot_auroc_comparison(self):
        # Extract data
        backends = sorted(list(set(r["backend"] for r in self.results)))
        categories = sorted(list(set(r["category"] for r in self.results)))
        
        x = np.arange(len(categories))
        width = 0.8 / len(backends)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, backend in enumerate(backends):
            scores = []
            for cat in categories:
                res = next((r for r in self.results if r["backend"] == backend and r["category"] == cat), None)
                scores.append(res["image_auroc"] if res else 0)
            
            ax.bar(x + i * width, scores, width, label=backend)

        ax.set_ylabel('Image AUROC')
        ax.set_title('Image AUROC by Category and Model')
        ax.set_xticks(x + width * (len(backends) - 1) / 2)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "auroc_comparison.png")
        plt.close()

    def _plot_advanced_metrics(self):
        backends = sorted(list(set(r["backend"] for r in self.results)))
        
        # Average metrics across all categories for each backend
        avg_f1 = []
        avg_pro = []
        
        for backend in backends:
            backend_results = [r for r in self.results if r["backend"] == backend]
            if backend_results:
                avg_f1.append(np.mean([r["f1_max"] for r in backend_results]))
                avg_pro.append(np.mean([r["pro_score"] for r in backend_results]))
            else:
                avg_f1.append(0)
                avg_pro.append(0)

        x = np.arange(len(backends))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, avg_f1, width, label='Avg F1-Max')
        ax.bar(x + width/2, avg_pro, width, label='Avg PRO Score')
        
        ax.set_ylabel('Score')
        ax.set_title('Average F1-Max and PRO Score by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(backends)
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "advanced_metrics.png")
        plt.close()

    def _plot_inference_time(self):
        backends = sorted(list(set(r["backend"] for r in self.results)))
        avg_times = []
        
        for backend in backends:
            times = [r["inference_time_ms"] for r in self.results if r["backend"] == backend]
            avg_times.append(np.mean(times) if times else 0)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(backends, avg_times, color='orange')
        ax.set_ylabel('Average Inference Time (ms)')
        ax.set_title('Inference Speed Comparison')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "inference_time.png")
        plt.close()

    def _generate_html(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build Table Rows
        table_rows = ""
        for r in self.results:
            table_rows += f"""
            <tr>
                <td>{r['backend']}</td>
                <td>{r['category']}</td>
                <td>{r['image_auroc']:.4f}</td>
                <td>{r['pixel_auroc']:.4f}</td>
                <td>{r['f1_max']:.4f}</td>
                <td>{r['pro_score']:.4f}</td>
                <td>{r['inference_time_ms']:.2f}</td>
            </tr>
            """
            
        error_section = ""
        if self.errors:
            error_rows = ""
            for e in self.errors:
                error_rows += f"<li><strong>{e['backend']} / {e['category']}:</strong> {e['error']}</li>"
            error_section = f"""
            <div class="error-section">
                <h3>Errors encountered:</h3>
                <ul>{error_rows}</ul>
            </div>
            """

        plots_section = ""
        if self.results:
             plots_section = """
            <h2>Performance Summary</h2>
            <div class="plots">
                <div class="plot-container">
                    <h3>Model Accuracy (AUROC)</h3>
                    <img src="plots/auroc_comparison.png" alt="AUROC Comparison">
                </div>
                <div class="plot-container">
                    <h3>Advanced Metrics (F1 & PRO)</h3>
                    <img src="plots/advanced_metrics.png" alt="Advanced Metrics">
                </div>
                <div class="plot-container">
                    <h3>Inference Speed</h3>
                    <img src="plots/inference_time.png" alt="Inference Time">
                </div>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plots {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                .plot-container {{ flex: 1; min-width: 400px; border: 1px solid #eee; padding: 10px; }}
                img {{ max-width: 100%; height: auto; }}
                .error-section {{ color: red; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Automated Benchmark Report</h1>
            <p>Generated on: {timestamp}</p>
            
            {plots_section}

            <h2>Detailed Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Backend</th>
                        <th>Category</th>
                        <th>Image AUROC</th>
                        <th>Pixel AUROC</th>
                        <th>F1-Max</th>
                        <th>PRO Score</th>
                        <th>Inference Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            
            {error_section}
        </body>
        </html>
        """
        return html
