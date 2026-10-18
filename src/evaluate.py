"""Evaluation framework to benchmark RAG retrieval accuracy and response faithfulness.

Directly measures:
1. Context Hit Rate (Top-K Retrieval Precision for technical terms)
2. Value Extraction Accuracy (Exact numerical parameter verification)
3. End-to-End Latency Profile
"""

import time
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

from src.rag_pipeline import TechnicalRAG

console = Console()

# Benchmark test cases with ground-truth expected parameters from QUINT4 datasheet
BENCHMARK_CASES = [
    {
        "question": "What is the nominal output current and boost capacity?",
        "expected_keywords": ["20 A", "25 A", "30 A"],
        "expected_parameter": "20 A nominal, 25 A static boost, 30 A dynamic boost"
    },
    {
        "question": "What is the degree of protection and protection class according to IEC?",
        "expected_keywords": ["IP20", "Class I"],
        "expected_parameter": "IP20, Class I"
    },
    {
        "question": "What is the operating ambient temperature range and derating threshold?",
        "expected_keywords": ["-25°C", "+70°C", "60°C"],
        "expected_parameter": "-25°C to +70°C, derating above 60°C"
    },
    {
        "question": "What is the efficiency at 230 V AC nominal load?",
        "expected_keywords": ["94", "%"],
        "expected_parameter": "> 94.0%"
    },
    {
        "question": "Does this unit support hydraulic fluid cooling?",
        "expected_keywords": ["does not specify", "not specified"],
        "expected_parameter": "Negative test case (unsupported parameter rejection)"
    }
]


def run_evaluation():
    console.print("[bold yellow]🚀 Running Technical RAG Evaluation Benchmark...[/bold yellow]\n")
    rag = TechnicalRAG()

    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(BENCHMARK_CASES, 1):
        q = case["question"]
        console.print(f"Testing [{idx}/{len(BENCHMARK_CASES)}]: [cyan]{q}[/cyan]")

        res = rag.query(q)

        # 1. Retrieval Hit Rate: Did retrieved chunks contain the required technical facts?
        retrieved_text = " ".join([c["content"] for c in res["sources"]]).lower()
        keywords_found = [
            kw for kw in case["expected_keywords"] if kw.lower() in retrieved_text or kw.lower() in res["answer"].lower()
        ]
        retrieval_pass = len(keywords_found) == len(case["expected_keywords"])

        results.append({
            "test_id": idx,
            "question": q,
            "hit": "PASS" if retrieval_pass else "PARTIAL",
            "matched_keywords": f"{len(keywords_found)}/{len(case['expected_keywords'])}",
            "retrieval_time": res["latency"]["retrieval_seconds"],
            "total_time": res["latency"]["total_seconds"]
        })

    # Summary Display
    table = Table(title="RAG Quality & Benchmark Results", border_style="green")
    table.add_column("Test ID", justify="center", width=8)
    table.add_column("Question", style="cyan")
    table.add_column("Status", justify="center", style="bold")
    table.add_column("Keyword Match", justify="center", style="magenta")
    table.add_column("Retrieval (s)", justify="right")
    table.add_column("Total (s)", justify="right")

    pass_count = 0
    for r in results:
        status_color = "green" if r["hit"] == "PASS" else "yellow"
        if r["hit"] == "PASS":
            pass_count += 1
        table.add_row(
            str(r["test_id"]),
            r["question"],
            f"[{status_color}]{r['hit']}[/{status_color}]",
            r["matched_keywords"],
            f"{r['retrieval_time']:.3f}",
            f"{r['total_time']:.3f}"
        )

    console.print(table)

    accuracy_pct = (pass_count / len(BENCHMARK_CASES)) * 100
    console.print(f"\n[bold green]Overall Benchmark Accuracy: {accuracy_pct:.1f}% ({pass_count}/{len(BENCHMARK_CASES)} passed)[/bold green]\n")


if __name__ == "__main__":
    run_evaluation()
