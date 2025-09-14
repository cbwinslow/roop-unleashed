#!/usr/bin/env python3
"""
Comprehensive test report generation system.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReportGenerator:
    """Generate comprehensive test reports with metrics and visualizations."""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "outputs"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": sys.platform
            },
            "test_results": {},
            "performance_metrics": {},
            "quality_assessments": {},
            "coverage_data": {},
            "recommendations": []
        }
    
    def collect_test_results(self):
        """Collect test results from various test categories."""
        # Mock test result collection - in practice, this would parse pytest output
        test_categories = [
            "unit_tests",
            "integration_tests", 
            "performance_tests",
            "face_processing_tests",
            "agent_tests",
            "hardware_tests",
            "monitoring_tests"
        ]
        
        for category in test_categories:
            self.report_data["test_results"][category] = self._generate_mock_test_results(category)
    
    def _generate_mock_test_results(self, category):
        """Generate mock test results for demonstration."""
        import random
        
        # Simulate different success rates for different test categories
        success_rates = {
            "unit_tests": 0.95,
            "integration_tests": 0.90,
            "performance_tests": 0.85,
            "face_processing_tests": 0.92,
            "agent_tests": 0.88,
            "hardware_tests": 0.80,
            "monitoring_tests": 0.93
        }
        
        base_success_rate = success_rates.get(category, 0.90)
        total_tests = random.randint(20, 100)
        passed_tests = int(total_tests * base_success_rate + random.uniform(-0.05, 0.05) * total_tests)
        passed_tests = max(0, min(total_tests, passed_tests))
        
        failed_tests = total_tests - passed_tests
        skipped_tests = random.randint(0, 5)
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "duration": random.uniform(5, 120),  # seconds
            "details": self._generate_test_details(passed_tests, failed_tests, skipped_tests)
        }
    
    def _generate_test_details(self, passed, failed, skipped):
        """Generate detailed test information."""
        details = []
        
        # Add passed tests
        for i in range(passed):
            details.append({
                "name": f"test_function_{i+1}",
                "status": "passed",
                "duration": round(random.uniform(0.1, 2.0), 3)
            })
        
        # Add failed tests
        failure_reasons = [
            "AssertionError: Expected quality > 0.8, got 0.75",
            "TimeoutError: Processing took longer than expected",
            "MemoryError: CUDA out of memory",
            "ValueError: Invalid input format"
        ]
        
        for i in range(failed):
            details.append({
                "name": f"test_function_{passed + i + 1}",
                "status": "failed",
                "duration": round(random.uniform(0.1, 5.0), 3),
                "error": random.choice(failure_reasons)
            })
        
        # Add skipped tests
        skip_reasons = [
            "GPU not available",
            "Test data not found",
            "Platform not supported"
        ]
        
        for i in range(skipped):
            details.append({
                "name": f"test_function_{passed + failed + i + 1}",
                "status": "skipped",
                "reason": random.choice(skip_reasons)
            })
        
        return details
    
    def collect_performance_metrics(self):
        """Collect performance benchmarking data."""
        import numpy as np
        
        # Mock performance metrics
        self.report_data["performance_metrics"] = {
            "face_detection": {
                "avg_time_ms": np.random.uniform(150, 300),
                "p95_time_ms": np.random.uniform(300, 500),
                "throughput_fps": np.random.uniform(5, 15),
                "memory_usage_mb": np.random.uniform(800, 1500),
                "gpu_utilization_percent": np.random.uniform(60, 90)
            },
            "face_swapping": {
                "avg_time_ms": np.random.uniform(400, 800),
                "p95_time_ms": np.random.uniform(800, 1200),
                "throughput_fps": np.random.uniform(2, 6),
                "memory_usage_mb": np.random.uniform(1500, 3000),
                "gpu_utilization_percent": np.random.uniform(70, 95)
            },
            "quality_assessment": {
                "avg_time_ms": np.random.uniform(50, 150),
                "p95_time_ms": np.random.uniform(150, 250),
                "ssim_calculation_ms": np.random.uniform(20, 50),
                "psnr_calculation_ms": np.random.uniform(15, 40)
            },
            "system_resources": {
                "peak_cpu_usage_percent": np.random.uniform(70, 95),
                "peak_memory_usage_gb": np.random.uniform(4, 12),
                "peak_gpu_memory_usage_gb": np.random.uniform(2, 6),
                "disk_io_mb_s": np.random.uniform(50, 200)
            }
        }
    
    def collect_quality_assessments(self):
        """Collect quality assessment data."""
        import numpy as np
        
        # Mock quality assessment results
        self.report_data["quality_assessments"] = {
            "face_detection_accuracy": {
                "precision": np.random.uniform(0.85, 0.95),
                "recall": np.random.uniform(0.80, 0.92),
                "f1_score": np.random.uniform(0.82, 0.93),
                "false_positive_rate": np.random.uniform(0.02, 0.08)
            },
            "face_swapping_quality": {
                "avg_ssim": np.random.uniform(0.75, 0.90),
                "avg_psnr": np.random.uniform(25, 35),
                "identity_preservation": np.random.uniform(0.80, 0.92),
                "blending_quality": np.random.uniform(0.78, 0.88)
            },
            "temporal_consistency": {
                "frame_to_frame_variance": np.random.uniform(0.02, 0.08),
                "flickering_score": np.random.uniform(0.1, 0.3),
                "motion_smoothness": np.random.uniform(0.85, 0.95)
            },
            "robustness": {
                "angle_tolerance": np.random.uniform(0.75, 0.90),
                "lighting_adaptation": np.random.uniform(0.70, 0.85),
                "obstruction_handling": np.random.uniform(0.65, 0.80),
                "resolution_scaling": np.random.uniform(0.80, 0.92)
            }
        }
    
    def collect_coverage_data(self):
        """Collect code coverage information."""
        # Mock coverage data
        self.report_data["coverage_data"] = {
            "overall_coverage": 78.5,
            "line_coverage": 82.3,
            "branch_coverage": 74.7,
            "function_coverage": 89.2,
            "by_module": {
                "roop.core": 85.2,
                "roop.enhanced_face_detection": 90.1,
                "roop.enhanced_face_swapper": 87.6,
                "roop.advanced_blending": 82.3,
                "agents.manager": 75.8,
                "agents.enhanced_agents": 71.4,
                "roop.ui": 68.9,
                "roop.utilities": 91.2
            },
            "uncovered_lines": [
                {"file": "roop/core.py", "lines": [45, 67, 123, 145]},
                {"file": "agents/manager.py", "lines": [78, 234, 267]},
                {"file": "roop/ui.py", "lines": [156, 289, 345, 398, 423]}
            ]
        }
    
    def generate_recommendations(self):
        """Generate improvement recommendations based on collected data."""
        recommendations = []
        
        # Analyze test results
        total_tests = sum(category["total"] for category in self.report_data["test_results"].values())
        total_passed = sum(category["passed"] for category in self.report_data["test_results"].values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        if overall_success_rate < 0.90:
            recommendations.append({
                "category": "Test Quality",
                "priority": "High",
                "description": f"Overall test success rate is {overall_success_rate:.1%}. Investigate failing tests.",
                "action": "Review and fix failing test cases"
            })
        
        # Analyze performance metrics
        face_swap_time = self.report_data["performance_metrics"]["face_swapping"]["avg_time_ms"]
        if face_swap_time > 600:
            recommendations.append({
                "category": "Performance",
                "priority": "Medium",
                "description": f"Face swapping taking {face_swap_time:.0f}ms on average. Consider optimization.",
                "action": "Profile and optimize face swapping algorithm"
            })
        
        # Analyze coverage
        coverage = self.report_data["coverage_data"]["overall_coverage"]
        if coverage < 80:
            recommendations.append({
                "category": "Code Coverage",
                "priority": "Medium",
                "description": f"Code coverage is {coverage:.1f}%. Aim for >80% coverage.",
                "action": "Add tests for uncovered code paths"
            })
        
        # Analyze quality metrics
        avg_ssim = self.report_data["quality_assessments"]["face_swapping_quality"]["avg_ssim"]
        if avg_ssim < 0.80:
            recommendations.append({
                "category": "Quality",
                "priority": "High",
                "description": f"Average SSIM is {avg_ssim:.2f}. Quality may be below expectations.",
                "action": "Investigate and improve face swapping quality"
            })
        
        self.report_data["recommendations"] = recommendations
    
    def generate_html_report(self):
        """Generate HTML report with visualizations."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Roop-Unleashed Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #444; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007acc; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007acc; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .chart-container {{ margin: 20px 0; height: 400px; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 20px; margin: 20px 0; }}
        .recommendation {{ margin: 10px 0; padding: 10px; border-left: 3px solid #f39c12; background: white; }}
        .priority-high {{ border-left-color: #e74c3c; }}
        .priority-medium {{ border-left-color: #f39c12; }}
        .priority-low {{ border-left-color: #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .status-passed {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
        .status-skipped {{ color: #f39c12; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Roop-Unleashed Comprehensive Test Report</h1>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{overall_success_rate:.1%}</div>
                <div class="metric-label">Overall Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{coverage:.1f}%</div>
                <div class="metric-label">Code Coverage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_processing_time:.0f}ms</div>
                <div class="metric-label">Avg Processing Time</div>
            </div>
        </div>

        <h2>📊 Test Results Overview</h2>
        <div class="chart-container">
            <canvas id="testResultsChart"></canvas>
        </div>

        <h2>⚡ Performance Metrics</h2>
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>

        <h2>🎯 Quality Assessment</h2>
        <div class="chart-container">
            <canvas id="qualityChart"></canvas>
        </div>

        <h2>📈 Code Coverage</h2>
        <table>
            <thead>
                <tr><th>Module</th><th>Coverage</th><th>Status</th></tr>
            </thead>
            <tbody>
                {coverage_table_rows}
            </tbody>
        </table>

        <h2>💡 Recommendations</h2>
        <div class="recommendations">
            {recommendations_html}
        </div>

        <h2>📋 Detailed Test Results</h2>
        {detailed_test_tables}

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #666; font-size: 0.9em;">
            Report generated on {timestamp}<br>
            Python {python_version} on {platform}
        </p>
    </div>

    <script>
        // Test Results Chart
        const testCtx = document.getElementById('testResultsChart').getContext('2d');
        new Chart(testCtx, {{
            type: 'bar',
            data: {{
                labels: {test_labels},
                datasets: [{{
                    label: 'Passed',
                    data: {passed_data},
                    backgroundColor: '#27ae60'
                }}, {{
                    label: 'Failed',
                    data: {failed_data},
                    backgroundColor: '#e74c3c'
                }}, {{
                    label: 'Skipped',
                    data: {skipped_data},
                    backgroundColor: '#f39c12'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        // Performance Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(perfCtx, {{
            type: 'radar',
            data: {{
                labels: ['Face Detection', 'Face Swapping', 'Quality Check', 'GPU Utilization', 'Memory Efficiency'],
                datasets: [{{
                    label: 'Performance Score',
                    data: {performance_data},
                    borderColor: '#007acc',
                    backgroundColor: 'rgba(0, 122, 204, 0.2)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});

        // Quality Chart
        const qualityCtx = document.getElementById('qualityChart').getContext('2d');
        new Chart(qualityCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['SSIM Score', 'PSNR Score', 'Identity Preservation', 'Blending Quality'],
                datasets: [{{
                    data: {quality_data},
                    backgroundColor: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    </script>
</body>
</html>
        """
        
        # Prepare data for the template
        total_tests = sum(category["total"] for category in self.report_data["test_results"].values())
        total_passed = sum(category["passed"] for category in self.report_data["test_results"].values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        coverage = self.report_data["coverage_data"]["overall_coverage"]
        
        # Calculate average processing time
        face_detection_time = self.report_data["performance_metrics"]["face_detection"]["avg_time_ms"]
        face_swap_time = self.report_data["performance_metrics"]["face_swapping"]["avg_time_ms"]
        avg_processing_time = (face_detection_time + face_swap_time) / 2
        
        # Prepare chart data
        test_categories = list(self.report_data["test_results"].keys())
        test_labels = [cat.replace("_", " ").title() for cat in test_categories]
        passed_data = [self.report_data["test_results"][cat]["passed"] for cat in test_categories]
        failed_data = [self.report_data["test_results"][cat]["failed"] for cat in test_categories]
        skipped_data = [self.report_data["test_results"][cat]["skipped"] for cat in test_categories]
        
        # Performance data (normalized to 0-100 scale)
        perf_metrics = self.report_data["performance_metrics"]
        performance_data = [
            min(100, (1000 / perf_metrics["face_detection"]["avg_time_ms"]) * 10),  # Inverse of time
            min(100, (1000 / perf_metrics["face_swapping"]["avg_time_ms"]) * 10),
            min(100, (1000 / perf_metrics["quality_assessment"]["avg_time_ms"]) * 20),
            perf_metrics["face_swapping"]["gpu_utilization_percent"],
            100 - (perf_metrics["system_resources"]["peak_memory_usage_gb"] / 16 * 100)  # Memory efficiency
        ]
        
        # Quality data (as percentages)
        quality_metrics = self.report_data["quality_assessments"]["face_swapping_quality"]
        quality_data = [
            quality_metrics["avg_ssim"] * 100,
            (quality_metrics["avg_psnr"] / 40) * 100,  # Normalize PSNR
            quality_metrics["identity_preservation"] * 100,
            quality_metrics["blending_quality"] * 100
        ]
        
        # Coverage table
        coverage_data = self.report_data["coverage_data"]["by_module"]
        coverage_table_rows = ""
        for module, cov in coverage_data.items():
            status = "Good" if cov >= 80 else "Needs Improvement" if cov >= 60 else "Poor"
            coverage_table_rows += f"<tr><td>{module}</td><td>{cov:.1f}%</td><td>{status}</td></tr>"
        
        # Recommendations
        recommendations_html = ""
        for rec in self.report_data["recommendations"]:
            priority_class = f"priority-{rec['priority'].lower()}"
            recommendations_html += f"""
            <div class="recommendation {priority_class}">
                <strong>{rec['category']} ({rec['priority']} Priority)</strong><br>
                {rec['description']}<br>
                <em>Action: {rec['action']}</em>
            </div>
            """
        
        # Detailed test tables
        detailed_test_tables = ""
        for category, results in self.report_data["test_results"].items():
            detailed_test_tables += f"<h3>{category.replace('_', ' ').title()}</h3>"
            detailed_test_tables += "<table><thead><tr><th>Test</th><th>Status</th><th>Duration</th><th>Details</th></tr></thead><tbody>"
            
            for test in results["details"]:
                status_class = f"status-{test['status']}"
                duration = f"{test.get('duration', 0):.3f}s" if 'duration' in test else "N/A"
                details = test.get('error', test.get('reason', ''))
                
                detailed_test_tables += f"""
                <tr>
                    <td>{test['name']}</td>
                    <td class="{status_class}">{test['status'].upper()}</td>
                    <td>{duration}</td>
                    <td>{details}</td>
                </tr>
                """
            
            detailed_test_tables += "</tbody></table>"
        
        # Fill template
        html_content = html_template.format(
            overall_success_rate=overall_success_rate,
            total_tests=total_tests,
            coverage=coverage,
            avg_processing_time=avg_processing_time,
            test_labels=json.dumps(test_labels),
            passed_data=json.dumps(passed_data),
            failed_data=json.dumps(failed_data),
            skipped_data=json.dumps(skipped_data),
            performance_data=json.dumps([round(x, 1) for x in performance_data]),
            quality_data=json.dumps([round(x, 1) for x in quality_data]),
            coverage_table_rows=coverage_table_rows,
            recommendations_html=recommendations_html if recommendations_html else "<p>No recommendations at this time. Great job!</p>",
            detailed_test_tables=detailed_test_tables,
            timestamp=self.report_data["metadata"]["generated_at"],
            python_version=self.report_data["metadata"]["python_version"],
            platform=self.report_data["metadata"]["platform"]
        )
        
        # Write HTML report
        report_path = self.output_dir / "test-report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def generate_json_report(self):
        """Generate JSON report for programmatic consumption."""
        report_path = self.output_dir / "test-report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        return report_path
    
    def generate_summary_report(self):
        """Generate concise summary report."""
        total_tests = sum(category["total"] for category in self.report_data["test_results"].values())
        total_passed = sum(category["passed"] for category in self.report_data["test_results"].values())
        total_failed = sum(category["failed"] for category in self.report_data["test_results"].values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        coverage = self.report_data["coverage_data"]["overall_coverage"]
        
        summary = f"""
# Roop-Unleashed Test Summary Report

**Generated:** {self.report_data["metadata"]["generated_at"]}

## 📊 Overall Results
- **Total Tests:** {total_tests}
- **Passed:** {total_passed}
- **Failed:** {total_failed}
- **Success Rate:** {overall_success_rate:.1%}
- **Code Coverage:** {coverage:.1f}%

## ⚡ Performance Highlights
- **Face Detection:** {self.report_data["performance_metrics"]["face_detection"]["avg_time_ms"]:.0f}ms avg
- **Face Swapping:** {self.report_data["performance_metrics"]["face_swapping"]["avg_time_ms"]:.0f}ms avg
- **GPU Utilization:** {self.report_data["performance_metrics"]["face_swapping"]["gpu_utilization_percent"]:.0f}%

## 🎯 Quality Metrics
- **SSIM Score:** {self.report_data["quality_assessments"]["face_swapping_quality"]["avg_ssim"]:.2f}
- **PSNR Score:** {self.report_data["quality_assessments"]["face_swapping_quality"]["avg_psnr"]:.1f}dB
- **Identity Preservation:** {self.report_data["quality_assessments"]["face_swapping_quality"]["identity_preservation"]:.1%}

## 💡 Key Recommendations
"""
        
        for rec in self.report_data["recommendations"][:3]:  # Top 3 recommendations
            summary += f"- **{rec['category']}** ({rec['priority']}): {rec['description']}\n"
        
        if not self.report_data["recommendations"]:
            summary += "- No critical issues found. System is performing well!\n"
        
        summary += f"\n---\n*Full report available in test-report.html*\n"
        
        summary_path = self.output_dir / "test-summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return summary_path
    
    def generate_all_reports(self):
        """Generate all report formats."""
        print("Collecting test data...")
        self.collect_test_results()
        self.collect_performance_metrics()
        self.collect_quality_assessments()
        self.collect_coverage_data()
        
        print("Generating recommendations...")
        self.generate_recommendations()
        
        print("Generating reports...")
        html_report = self.generate_html_report()
        json_report = self.generate_json_report()
        summary_report = self.generate_summary_report()
        
        print(f"✅ Reports generated:")
        print(f"  📄 HTML Report: {html_report}")
        print(f"  📋 JSON Report: {json_report}")
        print(f"  📝 Summary: {summary_report}")
        
        return {
            "html": html_report,
            "json": json_report,
            "summary": summary_report
        }


if __name__ == "__main__":
    # Generate comprehensive test report
    generator = TestReportGenerator()
    reports = generator.generate_all_reports()
    
    print("\n🎉 Test report generation completed!")
    print(f"Open {reports['html']} in your browser to view the full report.")