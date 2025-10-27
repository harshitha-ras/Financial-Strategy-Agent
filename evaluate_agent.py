#!/usr/bin/env python3
"""
Financial Strategy Agent Evaluation Script

This script evaluates the Financial Strategy Agent by:
1. Running test prompts through the agent
2. Comparing agent outputs with expected outputs
3. Computing similarity scores using multiple metrics
4. Highlighting matching words between outputs
5. Generating detailed HTML reports
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Set
from collections import Counter
import difflib

# Text similarity metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. ROUGE metrics will be unavailable.")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed. BLEU scores will be unavailable.")

# Import the agent (only if needed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# These imports are only needed when actually running the agent
AGENT_AVAILABLE = True
try:
    from app import run_financial_strategist_agent, setup_calcbench
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.node_parser import SentenceSplitter
except ImportError as e:
    AGENT_AVAILABLE = False
    print(f"Warning: Agent dependencies not available: {e}")
    print("You can still use --skip-agent to test the evaluation logic.")


class AgentEvaluator:
    """Evaluates the Financial Strategy Agent using various metrics."""

    def __init__(self, test_cases_path: str, upload_dir: str = "uploaded_docs"):
        """
        Initialize the evaluator.

        Args:
            test_cases_path: Path to JSON file with test cases
            upload_dir: Directory for uploaded documents
        """
        self.test_cases_path = test_cases_path
        self.upload_dir = upload_dir
        self.results = []

        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)

    def load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON file."""
        with open(self.test_cases_path, 'r') as f:
            return json.load(f)

    def setup_environment(self):
        """Set up the environment for running the agent."""
        if not AGENT_AVAILABLE:
            raise RuntimeError("Agent dependencies not available. Install requirements or use --skip-agent.")

        # Check for required API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY environment variable is required")

        # Configure LLM and embeddings
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        # Setup Calcbench if credentials are available
        calcbench_email = os.getenv("CALCBENCH_EMAIL")
        calcbench_password = os.getenv("CALCBENCH_PASSWORD")
        if calcbench_email and calcbench_password:
            setup_calcbench(calcbench_email, calcbench_password)
        else:
            print("Warning: CALCBENCH credentials not found. Some features may be limited.")

    def run_agent_on_prompt(self, prompt: str) -> str:
        """
        Run the agent on a single prompt and capture output.

        Args:
            prompt: The user prompt to test

        Returns:
            The agent's response as a string
        """
        if not AGENT_AVAILABLE:
            raise RuntimeError("Agent dependencies not available. Cannot run agent.")

        # Capture stdout to get agent output
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            run_financial_strategist_agent(prompt, self.upload_dir)
            output = captured_output.getvalue()
        except Exception as e:
            output = f"Error running agent: {str(e)}"
        finally:
            sys.stdout = old_stdout

        # Extract strategic recommendations from output
        # Look for "Strategic Recommendations:" section
        strategy_match = re.search(
            r'Strategic Recommendations:(.+?)(?=\n\n|\Z)',
            output,
            re.DOTALL | re.IGNORECASE
        )

        if strategy_match:
            return strategy_match.group(1).strip()

        # If not found, return full output
        return output.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation for better matching
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            return word_tokenize(text.lower())
        else:
            # Simple whitespace tokenization
            return self.normalize_text(text).split()

    def calculate_word_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate word overlap between two texts (Jaccard similarity).

        Returns:
            Float between 0 and 1
        """
        words1 = set(self.tokenize(text1))
        words2 = set(self.tokenize(text2))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not ROUGE_AVAILABLE:
            return {}

        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }

    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        if not NLTK_AVAILABLE:
            return 0.0

        reference_tokens = self.tokenize(reference)
        hypothesis_tokens = self.tokenize(hypothesis)

        # Use smoothing to avoid zero scores
        smoothing = SmoothingFunction().method1

        try:
            return sentence_bleu(
                [reference_tokens],
                hypothesis_tokens,
                smoothing_function=smoothing
            )
        except:
            return 0.0

    def find_matching_words(self, text1: str, text2: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Find matching and non-matching words between two texts.

        Returns:
            Tuple of (matching_words, only_in_text1, only_in_text2)
        """
        words1 = set(self.tokenize(text1))
        words2 = set(self.tokenize(text2))

        matching = words1.intersection(words2)
        only_in_1 = words1 - words2
        only_in_2 = words2 - words1

        return matching, only_in_1, only_in_2

    def highlight_matching_words_html(self, text1: str, text2: str) -> Tuple[str, str]:
        """
        Generate HTML with highlighted matching words.

        Returns:
            Tuple of (highlighted_text1_html, highlighted_text2_html)
        """
        matching_words, _, _ = self.find_matching_words(text1, text2)

        def highlight_text(text: str) -> str:
            """Highlight matching words in text."""
            words = text.split()
            highlighted = []

            for word in words:
                # Clean word for matching
                clean_word = re.sub(r'[^\w\s]', '', word.lower())

                if clean_word in matching_words:
                    highlighted.append(f'<span class="match">{word}</span>')
                else:
                    highlighted.append(word)

            return ' '.join(highlighted)

        return highlight_text(text1), highlight_text(text2)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using sequence matching.
        This is a simple alternative to embeddings-based similarity.
        """
        return difflib.SequenceMatcher(None,
                                       self.normalize_text(text1),
                                       self.normalize_text(text2)).ratio()

    def evaluate_single_case(self, test_case: Dict, run_agent: bool = True) -> Dict:
        """
        Evaluate a single test case.

        Args:
            test_case: Dictionary with 'id', 'prompt', 'expected_output', 'context'
            run_agent: Whether to actually run the agent (set False for testing)

        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating Test Case #{test_case['id']}...")
        print(f"Prompt: {test_case['prompt'][:100]}...")

        # Run the agent
        if run_agent:
            agent_output = self.run_agent_on_prompt(test_case['prompt'])
        else:
            # For testing without running agent
            agent_output = "This is a test output for evaluation purposes."

        expected_output = test_case['expected_output']

        # Calculate metrics
        word_overlap = self.calculate_word_overlap(expected_output, agent_output)
        rouge_scores = self.calculate_rouge_scores(expected_output, agent_output)
        bleu_score = self.calculate_bleu_score(expected_output, agent_output)
        semantic_sim = self.calculate_semantic_similarity(expected_output, agent_output)

        # Find matching words
        matching, only_expected, only_agent = self.find_matching_words(
            expected_output, agent_output
        )

        # Generate highlighted HTML
        expected_html, agent_html = self.highlight_matching_words_html(
            expected_output, agent_output
        )

        # Calculate composite score (weighted average)
        composite_score = (
            word_overlap * 0.3 +
            rouge_scores.get('rougeL_f', 0) * 0.3 +
            bleu_score * 0.2 +
            semantic_sim * 0.2
        )

        result = {
            'test_case_id': test_case['id'],
            'prompt': test_case['prompt'],
            'context': test_case.get('context', 'N/A'),
            'expected_output': expected_output,
            'agent_output': agent_output,
            'expected_output_html': expected_html,
            'agent_output_html': agent_html,
            'metrics': {
                'word_overlap': round(word_overlap, 4),
                'semantic_similarity': round(semantic_sim, 4),
                'bleu_score': round(bleu_score, 4),
                **{k: round(v, 4) for k, v in rouge_scores.items()},
                'composite_score': round(composite_score, 4)
            },
            'matching_words_count': len(matching),
            'total_expected_words': len(self.tokenize(expected_output)),
            'total_agent_words': len(self.tokenize(agent_output)),
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Composite Score: {result['metrics']['composite_score']:.2%}")

        return result

    def evaluate_all(self, run_agent: bool = True) -> List[Dict]:
        """
        Evaluate all test cases.

        Args:
            run_agent: Whether to actually run the agent

        Returns:
            List of evaluation results
        """
        print("=" * 80)
        print("FINANCIAL STRATEGY AGENT EVALUATION")
        print("=" * 80)

        # Setup environment
        if run_agent:
            print("\nSetting up environment...")
            self.setup_environment()

        # Load test cases
        print(f"\nLoading test cases from {self.test_cases_path}...")
        test_cases = self.load_test_cases()
        print(f"Found {len(test_cases)} test cases")

        # Evaluate each case
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] ", end="")
            result = self.evaluate_single_case(test_case, run_agent=run_agent)
            results.append(result)

        self.results = results
        return results

    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics across all evaluations."""
        if not self.results:
            return {}

        metrics = [r['metrics'] for r in self.results]

        def avg(key):
            values = [m[key] for m in metrics if key in m]
            return sum(values) / len(values) if values else 0

        return {
            'total_test_cases': len(self.results),
            'average_composite_score': round(avg('composite_score'), 4),
            'average_word_overlap': round(avg('word_overlap'), 4),
            'average_semantic_similarity': round(avg('semantic_similarity'), 4),
            'average_bleu_score': round(avg('bleu_score'), 4),
            'average_rouge1_f': round(avg('rouge1_f'), 4),
            'average_rouge2_f': round(avg('rouge2_f'), 4),
            'average_rougeL_f': round(avg('rougeL_f'), 4),
            'evaluation_timestamp': datetime.now().isoformat()
        }

    def save_results_json(self, output_path: str):
        """Save evaluation results to JSON file."""
        output = {
            'summary': self.generate_summary_stats(),
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    def generate_html_report(self, output_path: str):
        """Generate comprehensive HTML report with highlighted text."""
        summary = self.generate_summary_stats()

        # Use f-strings to avoid issues with CSS curly braces
        timestamp = summary['evaluation_timestamp']
        total_cases = summary['total_test_cases']
        avg_composite = summary['average_composite_score']
        avg_overlap = summary['average_word_overlap']
        avg_rougeL = summary.get('average_rougeL_f', 0)
        test_cases_html = self._generate_test_cases_html()

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Strategy Agent Evaluation Report</title>
    <style>
        * {{{{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}}}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}

        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}

        .summary {{
            background: #ecf0f1;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 40px;
        }}

        .summary h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}

        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-value {{
            color: #2c3e50;
            font-size: 2em;
            font-weight: bold;
            margin-top: 5px;
        }}

        .test-case {{
            margin-bottom: 50px;
            padding: 30px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}

        .test-case h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.5em;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}

        .score-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            margin-left: 15px;
        }}

        .score-excellent {{
            background: #27ae60;
            color: white;
        }}

        .score-good {{
            background: #f39c12;
            color: white;
        }}

        .score-fair {{
            background: #e67e22;
            color: white;
        }}

        .score-poor {{
            background: #e74c3c;
            color: white;
        }}

        .prompt-box {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
            border-radius: 4px;
        }}

        .context-tag {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            margin: 5px 5px 5px 0;
        }}

        .output-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 25px 0;
        }}

        .output-panel {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}

        .output-panel h4 {{
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #dee2e6;
        }}

        .output-text {{
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .match {{
            background: #fff59d;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .metrics-table th,
        .metrics-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}

        .metrics-table th {{
            background: #34495e;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}

        .metrics-table tr:hover {{
            background: #f8f9fa;
        }}

        .metric-bar {{
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}

        .metric-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 10px;
            transition: width 0.3s ease;
        }}

        @media (max-width: 768px) {{
            .output-comparison {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial Strategy Agent Evaluation Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Test Cases</div>
                    <div class="stat-value">{total_cases}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Average Composite Score</div>
                    <div class="stat-value">{avg_composite:.1%}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Average Word Overlap</div>
                    <div class="stat-value">{avg_overlap:.1%}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Average ROUGE-L</div>
                    <div class="stat-value">{avg_rougeL:.1%}</div>
                </div>
            </div>
        </div>

        <h2 style="margin: 40px 0 20px 0; color: #2c3e50;">Detailed Results</h2>

        {test_cases_html}
    </div>
</body>
</html>
        """

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✓ HTML report saved to {output_path}")

    def _generate_test_cases_html(self) -> str:
        """Generate HTML for individual test cases."""
        html_parts = []

        for result in self.results:
            score = result['metrics']['composite_score']

            # Determine score class
            if score >= 0.75:
                score_class = "score-excellent"
                score_label = "Excellent"
            elif score >= 0.50:
                score_class = "score-good"
                score_label = "Good"
            elif score >= 0.25:
                score_class = "score-fair"
                score_label = "Fair"
            else:
                score_class = "score-poor"
                score_label = "Needs Improvement"

            # Generate metrics rows
            metrics_rows = ""
            for metric_name, metric_value in result['metrics'].items():
                if metric_name != 'composite_score':
                    bar_width = int(metric_value * 100)
                    metrics_rows += f"""
                    <tr>
                        <td style="font-weight: 500;">{metric_name.replace('_', ' ').title()}</td>
                        <td>{metric_value:.2%}</td>
                        <td style="width: 200px;">
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: {bar_width}%"></div>
                            </div>
                        </td>
                    </tr>
                    """

            # Generate context tags
            context_tags = ""
            if result['context'] != 'N/A':
                for tag in result['context'].split(','):
                    context_tags += f'<span class="context-tag">{tag.strip()}</span>'

            case_html = f"""
        <div class="test-case">
            <h3>
                Test Case #{result['test_case_id']}
                <span class="score-badge {score_class}">{score:.1%} - {score_label}</span>
            </h3>

            <div class="prompt-box">
                <strong>Prompt:</strong><br>
                {result['prompt']}
            </div>

            <div style="margin: 15px 0;">
                {context_tags}
            </div>

            <h4 style="margin: 25px 0 15px 0; color: #2c3e50;">Output Comparison</h4>
            <p style="color: #7f8c8d; margin-bottom: 15px;">
                <span class="match">Highlighted words</span> appear in both outputs
            </p>

            <div class="output-comparison">
                <div class="output-panel">
                    <h4>Expected Output</h4>
                    <div class="output-text">{result['expected_output_html']}</div>
                    <p style="color: #7f8c8d; margin-top: 15px; font-size: 0.9em;">
                        Total words: {result['total_expected_words']}
                    </p>
                </div>

                <div class="output-panel">
                    <h4>Agent Output</h4>
                    <div class="output-text">{result['agent_output_html']}</div>
                    <p style="color: #7f8c8d; margin-top: 15px; font-size: 0.9em;">
                        Total words: {result['total_agent_words']}
                    </p>
                </div>
            </div>

            <h4 style="margin: 25px 0 15px 0; color: #2c3e50;">Evaluation Metrics</h4>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Score</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics_rows}
                </tbody>
            </table>

            <p style="color: #7f8c8d; margin-top: 15px; font-size: 0.9em;">
                Matching words: {result['matching_words_count']} |
                Evaluated: {result['timestamp']}
            </p>
        </div>
            """

            html_parts.append(case_html)

        return '\n'.join(html_parts)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate the Financial Strategy Agent'
    )
    parser.add_argument(
        '--test-cases',
        default='evaluation_test_cases.json',
        help='Path to test cases JSON file'
    )
    parser.add_argument(
        '--output-json',
        default='evaluation_results.json',
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--output-html',
        default='evaluation_report.html',
        help='Path to save HTML report'
    )
    parser.add_argument(
        '--upload-dir',
        default='uploaded_docs',
        help='Directory for uploaded documents'
    )
    parser.add_argument(
        '--skip-agent',
        action='store_true',
        help='Skip running the agent (for testing the evaluation script only)'
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = AgentEvaluator(
        test_cases_path=args.test_cases,
        upload_dir=args.upload_dir
    )

    # Run evaluation
    evaluator.evaluate_all(run_agent=not args.skip_agent)

    # Generate outputs
    evaluator.save_results_json(args.output_json)
    evaluator.generate_html_report(args.output_html)

    # Print summary
    summary = evaluator.generate_summary_stats()
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total Test Cases: {summary['total_test_cases']}")
    print(f"Average Composite Score: {summary['average_composite_score']:.2%}")
    print(f"Average Word Overlap: {summary['average_word_overlap']:.2%}")
    print(f"Average Semantic Similarity: {summary['average_semantic_similarity']:.2%}")
    if summary.get('average_rougeL_f'):
        print(f"Average ROUGE-L: {summary['average_rougeL_f']:.2%}")
    print("\n✓ Evaluation complete! Check the HTML report for detailed results.")
    print(f"  HTML Report: {args.output_html}")
    print(f"  JSON Results: {args.output_json}")


if __name__ == "__main__":
    main()
