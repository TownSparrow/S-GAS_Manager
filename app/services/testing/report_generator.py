import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json


class ReportGenerator:
    """ Report generator """
    
    def __init__(self, results_dir: str = "logs/benchmarks"):
        self.results_dir = Path(results_dir)
    
    def generate_plots(self, csv_file: str, output_dir: str = None):
        """ Generating the schedules from CSV """
        df = pd.read_csv(csv_file)
        
        if output_dir is None:
            output_dir = self.results_dir / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Coverage schedule
        plt.figure(figsize=(12, 6))
        plt.plot(df['iteration'], df['coverage_ratio'] * 100, marker='o', linewidth=2, markersize=8)
        plt.title('Document coverage by iterations', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Coverage (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Target (nearby): 80%')
        plt.legend()
        plt.savefig(output_dir / f"{Path(csv_file).stem}_coverage.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # VRAM resources schedule
        plt.figure(figsize=(12, 6))
        plt.plot(df['iteration'], df['vram_allocated_gb'], marker='s', color='red', linewidth=2, markersize=8)
        plt.title('VRAM resources by iterations', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('VRAM (GB)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=7.5, color='r', linestyle='--', alpha=0.5, label='Target (nearby): 7.5 GB')
        plt.legend()
        plt.savefig(output_dir / f"{Path(csv_file).stem}_vram.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Latency schedule
        plt.figure(figsize=(12, 6))
        plt.plot(df['iteration'], df['latency_ms'], marker='^', color='green', linewidth=2, markersize=8)
        plt.title('Latency by iterations', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Target (nearby): 200 ms')
        plt.legend()
        plt.savefig(output_dir / f"{Path(csv_file).stem}_latency.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Active chunks schedule
        plt.figure(figsize=(12, 6))
        plt.bar(df['iteration'], df['active_chunks'], alpha=0.7, color='blue')
        plt.title('Active chunks in VRAM by iterations', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Chunks amount', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(output_dir / f"{Path(csv_file).stem}_chunks.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Schedule exported in {output_dir}")
        return output_dir
    
    def generate_html_report(self, json_file: str, plots_dir: str):
        """ Generating the HTML report """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = data['summary']
        detailed = data['detailed_metrics']
        
        html = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <title>S-GAS Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 28px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                .target {{ color: #27ae60; font-size: 12px; }}
                .warning {{ color: #e74c3c; font-size: 12px; }}
                img {{ max-width: 100%; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #3498db; color: white; }}
                tr:hover {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>S-GAS Benchmark Report</h1>
                
                <h2>Summary metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{summary['total_turns']}</div>
                        <div class="metric-label">All iterations</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary.get('generation_metrics', {}).get('multi_turn_accuracy', 'N/A')}%</div>
                        <div class="metric-label">Multi-turn Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['final_coverage']*100:.1f}%</div>
                        <div class="metric-label">Final Coverage</div>
                        <div class="target">Target (nearby): 80%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['peak_vram_gb']:.2f} GB</div>
                        <div class="metric-label">Peak VRAM Resources</div>
                        <div class="target">Target (nearby): ≤7.5 GB</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['avg_latency_ms']:.1f} ms</div>
                        <div class="metric-label">Avarage Latency</div>
                        <div class="target">Target (nearby): ≤200 ms</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['avg_cache_hit_rate']*100:.1f}%</div>
                        <div class="metric-label">Cache Hit Rate</div>
                    </div>
                </div>
                
                <h2>Comparison with targets from hypothesis</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Target</th>
                        <th>Reached</th>
                        <th>Efficiency</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>VRAM</td>
                        <td>≤7.5 GB</td>
                        <td>{summary['target_comparison']['vram_achieved']:.2f} GB</td>
                        <td>{summary['target_comparison']['vram_efficiency']:.1f}%</td>
                        <td>{'✅ Success' if summary['target_comparison']['vram_efficiency'] >= 100 else '⚠️ Warning'}</td>
                    </tr>
                    <tr>
                        <td>Latency</td>
                        <td>≤200 ms</td>
                        <td>{summary['target_comparison']['latency_achieved']:.1f} ms</td>
                        <td>{summary['target_comparison']['latency_efficiency']:.1f}%</td>
                        <td>{'✅ Success' if summary['target_comparison']['latency_efficiency'] >= 100 else '⚠️ Warning'}</td>
                    </tr>
                    <tr>
                        <td>Coverage</td>
                        <td>≥80%</td>
                        <td>{summary['target_comparison']['coverage_achieved']*100:.1f}%</td>
                        <td>{summary['target_comparison']['coverage_efficiency']:.1f}%</td>
                        <td>{'✅ Success' if summary['target_comparison']['coverage_efficiency'] >= 100 else '⚠️ Warning'}</td>
                    </tr>
                </table>
                
                <h2>Schedules</h2>
                <img src="{plots_dir}/{Path(json_file).stem}_coverage.png" alt="Coverage">
                <img src="{plots_dir}/{Path(json_file).stem}_vram.png" alt="VRAM">
                <img src="{plots_dir}/{Path(json_file).stem}_latency.png" alt="Latency">
                <img src="{plots_dir}/{Path(json_file).stem}_chunks.png" alt="Chunks">
                
                <h2>Detailed statistics by iteration</h2>
                <table>
                    <tr>
                        <th>Iteration</th>
                        <th>Coverage</th>
                        <th>VRAM (GB)</th>
                        <th>Latency (ms)</th>
                        <th>Active chunks</th>
                        <th>Cache Hit Rate</th>
                    </tr>
        """
        
        for metric in detailed:
            html += f"""
                    <tr>
                        <td>{metric['iteration']}</td>
                        <td>{metric['coverage_ratio']*100:.1f}%</td>
                        <td>{metric['vram_allocated_gb']:.2f}</td>
                        <td>{metric['latency_ms']:.1f}</td>
                        <td>{metric['active_chunks']}</td>
                        <td>{metric['cache_hit_rate']*100:.1f}%</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        html_file = Path(json_file).with_suffix('.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML report is exported: {html_file}")
        return str(html_file)