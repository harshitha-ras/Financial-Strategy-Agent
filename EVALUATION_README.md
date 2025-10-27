# Financial Strategy Agent - Evaluation System

This evaluation system provides comprehensive testing and scoring for the Financial Strategy Agent. It compares agent outputs against expected responses using multiple metrics and generates detailed HTML reports with visual highlighting of matching content.

## Overview

The evaluation system consists of:

1. **Test Cases** (`evaluation_test_cases.json`) - Prompts with expected outputs
2. **Evaluation Script** (`evaluate_agent.py`) - Main evaluation engine
3. **HTML Reports** - Visual comparison with highlighted matching words
4. **JSON Results** - Detailed metrics and scores

## Features

### Evaluation Metrics

The system calculates multiple metrics for comprehensive evaluation:

- **Word Overlap (Jaccard Similarity)** - Measures shared vocabulary between outputs
- **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L) - Standard NLP evaluation metrics
- **BLEU Score** - Machine translation quality metric adapted for text similarity
- **Semantic Similarity** - Sequence-based text matching
- **Composite Score** - Weighted average of all metrics (30% word overlap, 30% ROUGE-L, 20% BLEU, 20% semantic)

### Visual Highlighting

The HTML report highlights matching words between expected and agent outputs, making it easy to:
- Identify what the agent got right
- Spot missing key terms
- Compare response structure

### Detailed Reports

Generated reports include:
- Summary statistics across all test cases
- Individual test case scores with color-coded badges
- Side-by-side output comparison
- Visual metric bars
- Timestamp tracking

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The evaluation system requires:
- `rouge-score` - For ROUGE metrics
- `nltk` - For BLEU scores and tokenization

### 2. Set Environment Variables

The evaluation script needs the same API keys as the main agent:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export CALCBENCH_EMAIL="your-calcbench-email"
export CALCBENCH_PASSWORD="your-calcbench-password"
```

## Usage

### Basic Usage

Run the evaluation with default settings:

```bash
python evaluate_agent.py
```

This will:
1. Load test cases from `evaluation_test_cases.json`
2. Run each prompt through the agent
3. Calculate similarity metrics
4. Generate `evaluation_results.json` (detailed JSON results)
5. Generate `evaluation_report.html` (visual report)

### Custom Parameters

```bash
python evaluate_agent.py \
  --test-cases custom_test_cases.json \
  --output-json my_results.json \
  --output-html my_report.html \
  --upload-dir my_docs
```

**Parameters:**
- `--test-cases` - Path to test cases JSON file (default: `evaluation_test_cases.json`)
- `--output-json` - Path for JSON results (default: `evaluation_results.json`)
- `--output-html` - Path for HTML report (default: `evaluation_report.html`)
- `--upload-dir` - Directory for uploaded documents (default: `uploaded_docs`)
- `--skip-agent` - Skip running agent (for testing script only)

### Test Without Running Agent

To test the evaluation script without actually running the agent:

```bash
python evaluate_agent.py --skip-agent
```

This is useful for:
- Testing the evaluation logic
- Verifying report generation
- Checking metrics calculation

## Test Cases Format

Test cases are stored in JSON format:

```json
[
  {
    "id": 1,
    "prompt": "Your test prompt with company tickers (AAPL) (MSFT)",
    "expected_output": "The expected strategic recommendations...",
    "context": "Industry sector, specific challenge"
  }
]
```

**Fields:**
- `id` (required) - Unique identifier for the test case
- `prompt` (required) - The user query to test
- `expected_output` (required) - The ideal response you expect
- `context` (optional) - Tags describing the test scenario

### Creating Test Cases

1. **Write realistic prompts** that include:
   - Clear business problem description
   - Company tickers in parentheses (e.g., "Ford (F)")
   - Specific strategic questions

2. **Define expected outputs** that include:
   - 2-3 numbered strategic recommendations
   - Data-driven rationale
   - Actionable next steps
   - Similar structure to actual agent outputs

3. **Add context tags** for categorization:
   - Industry (e.g., "automotive", "retail", "technology")
   - Challenge type (e.g., "margin compression", "competition")
   - Data sources (e.g., "SEC filings", "market trends")

### Example Test Case

```json
{
  "id": 1,
  "prompt": "Our automotive company is facing margin compression due to rising battery costs. Analyze how Tesla (TSLA) and Ford (F) are managing their supply chain costs and suggest strategies.",
  "expected_output": "Based on the analysis of Tesla and Ford's strategies:\n\n1. Vertical Integration: Tesla's in-house battery production at Gigafactories reduces supplier dependency. Ford partnered with SK Innovation for joint battery plants. Consider establishing strategic partnerships or investing in production capabilities.\n\n2. Supplier Diversification: Ford has diversified battery suppliers across regions. Implement multi-sourcing for critical components.\n\n3. Technology Investment: Tesla's 4680 battery cells promise 50% cost reduction. Allocate R&D budget toward next-generation technologies.",
  "context": "Automotive industry, margin compression, battery supply chain"
}
```

## Understanding the Results

### Composite Score Interpretation

The composite score (0-1) indicates overall quality:

- **0.75 - 1.00**: Excellent - Agent response closely matches expected output
- **0.50 - 0.74**: Good - Strong alignment with most key points
- **0.25 - 0.49**: Fair - Some alignment but missing key elements
- **0.00 - 0.24**: Needs Improvement - Significant divergence from expected output

### Metric Breakdown

1. **Word Overlap** (Jaccard Similarity)
   - Measures vocabulary overlap
   - Good for checking if key terms are present
   - Range: 0-1 (higher is better)

2. **ROUGE-L** (Longest Common Subsequence)
   - Measures longest matching sequence
   - Good for structural similarity
   - Range: 0-1 (higher is better)

3. **BLEU Score**
   - Precision-focused metric
   - Good for checking exact phrase matches
   - Range: 0-1 (higher is better)

4. **Semantic Similarity**
   - Sequence-based text matching
   - Good for overall content alignment
   - Range: 0-1 (higher is better)

### Reading the HTML Report

The HTML report includes:

1. **Summary Section**
   - Total test cases evaluated
   - Average scores across all metrics
   - Timestamp of evaluation

2. **Test Case Details**
   - Score badge (color-coded)
   - Original prompt and context tags
   - Side-by-side comparison with highlighted matches
   - Detailed metrics table with visual bars
   - Word counts and timestamps

3. **Highlighted Text**
   - Yellow highlighting shows words that appear in both outputs
   - Helps identify coverage of key concepts
   - Makes it easy to spot missing important terms

## Programmatic Usage

You can also use the evaluator in your own Python scripts:

```python
from evaluate_agent import AgentEvaluator

# Create evaluator
evaluator = AgentEvaluator(
    test_cases_path='evaluation_test_cases.json',
    upload_dir='uploaded_docs'
)

# Run evaluation
evaluator.setup_environment()
results = evaluator.evaluate_all(run_agent=True)

# Generate reports
evaluator.save_results_json('results.json')
evaluator.generate_html_report('report.html')

# Get summary statistics
summary = evaluator.generate_summary_stats()
print(f"Average Score: {summary['average_composite_score']:.2%}")
```

### Evaluating a Single Test Case

```python
from evaluate_agent import AgentEvaluator

evaluator = AgentEvaluator('evaluation_test_cases.json')
evaluator.setup_environment()

test_case = {
    'id': 999,
    'prompt': 'Your test prompt here (AAPL)',
    'expected_output': 'Expected strategic recommendations...',
    'context': 'Test scenario'
}

result = evaluator.evaluate_single_case(test_case, run_agent=True)
print(f"Score: {result['metrics']['composite_score']:.2%}")
```

## Customizing Evaluation

### Adding New Metrics

To add custom evaluation metrics, extend the `AgentEvaluator` class:

```python
class CustomEvaluator(AgentEvaluator):
    def calculate_custom_metric(self, text1, text2):
        # Your custom metric logic
        return score

    def evaluate_single_case(self, test_case, run_agent=True):
        result = super().evaluate_single_case(test_case, run_agent)

        # Add custom metric
        custom_score = self.calculate_custom_metric(
            test_case['expected_output'],
            result['agent_output']
        )
        result['metrics']['custom_metric'] = custom_score

        return result
```

### Adjusting Composite Score Weights

Edit `evaluate_agent.py` line ~268 to change metric weights:

```python
# Current weights: 30% word overlap, 30% ROUGE-L, 20% BLEU, 20% semantic
composite_score = (
    word_overlap * 0.3 +
    rouge_scores.get('rougeL_f', 0) * 0.3 +
    bleu_score * 0.2 +
    semantic_sim * 0.2
)
```

### Custom HTML Styling

Edit the CSS in `generate_html_report()` method to customize:
- Colors and fonts
- Layout and spacing
- Highlight colors
- Badge styles

## Best Practices

### 1. Test Case Quality

- **Be specific**: Include exact company tickers and clear problems
- **Be realistic**: Write prompts that real users would ask
- **Be comprehensive**: Cover different industries and scenarios
- **Keep expectations realistic**: Don't expect word-for-word matches

### 2. Evaluation Frequency

- Run after significant agent changes
- Create regression tests for bug fixes
- Track scores over time to measure improvements
- Use version control for test cases

### 3. Interpreting Results

- Focus on composite score trends, not absolute values
- Review low-scoring cases to identify weaknesses
- Check if mismatches are semantic or factual
- Use highlighted text to verify key concepts are covered

### 4. Iteration

- Start with 5-10 high-quality test cases
- Add cases when bugs are found
- Update expected outputs as agent improves
- Remove outdated or irrelevant cases

## Troubleshooting

### ROUGE/NLTK Not Available

If you see warnings about missing packages:

```bash
pip install rouge-score nltk
```

### Agent Errors During Evaluation

- Check that all API keys are set correctly
- Verify network connectivity
- Check API rate limits
- Review error messages in console output

### Low Scores Despite Good Output

- Expected outputs may be too specific
- Metrics favor exact matches over semantic equivalence
- Consider if the agent's response is actually valid
- Adjust expected outputs to be more general

### HTML Report Not Generating

- Check file permissions in output directory
- Verify results were generated successfully
- Check for errors in console output
- Ensure sufficient disk space

## Example Workflow

1. **Create test cases** based on real user queries
2. **Run initial evaluation** to establish baseline
3. **Review HTML report** to identify weak areas
4. **Improve agent** based on findings
5. **Re-run evaluation** to measure improvement
6. **Track scores over time** using version control

```bash
# Initial evaluation
python evaluate_agent.py --output-html baseline_report.html

# After improvements
python evaluate_agent.py --output-html improved_report.html

# Compare results
diff evaluation_results_baseline.json evaluation_results_improved.json
```

## Contributing

To add new test cases:

1. Edit `evaluation_test_cases.json`
2. Follow the JSON format
3. Include diverse scenarios
4. Test with `--skip-agent` first to verify format
5. Run full evaluation to establish scores

## License

This evaluation system is part of the Financial Strategy Agent project and follows the same license.

## Support

For issues or questions about the evaluation system:
1. Check this README
2. Review example test cases
3. Run with `--skip-agent` to test script functionality
4. Check API credentials and environment setup
