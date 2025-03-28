# DeepResearchResultProcessor

A tool for processing technical roadmaps and generating structured learning materials with automated scheduling.

## Description

DeepResearchResultProcessor is a Python application that uses AI to analyze technical roadmaps, extract key concepts, and generate comprehensive learning materials. It automatically schedules learning tasks based on estimated time requirements, creating a structured learning path with deadlines.

The tool processes markdown roadmap files and generates:
- Detailed markdown files for each technical concept
- Front matter with tags and scheduled dates
- An index file linking all concepts
- A JSON output with all extracted information

## Features

- **AI-Powered Concept Extraction**: Automatically identifies technical concepts from roadmap content
- **Detailed Concept Analysis**: Extracts prerequisites, learning resources, explanations, and more
- **Automated Learning Schedule**: Calculates start and deadline dates based on concept complexity
- **Obsidian-Compatible Output**: Generates markdown files with front matter for use in Obsidian or similar knowledge management tools
- **Configurable Processing**: Control the number of concepts processed with command-line arguments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepResearchResultProcessor.git
cd DeepResearchResultProcessor

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python DeepResearchResultProcessor.py --input your_roadmap.md --model gpt-4o-mini --api-key your_openai_api_key
```

### Command-line Arguments

- `--input`, `-i`: Input markdown file path (default: jina_result_.md)
- `--model`, `-m`: OpenAI model to use (default: gpt-4o-mini)
- `--api-key`, `-k`: OpenAI API key (overrides environment variable)
- `--recursion-limit`, `-r`: Maximum number of concepts to process (default: 25)

### Example

```bash
python DeepResearchResultProcessor.py --input machine_learning_roadmap.md --recursion-limit 10
```

## Output

The tool generates:

1. A directory named after the input file containing:
   - Markdown files for each concept with front matter, explanations, resources, and practice questions
   - An index.md file linking to all concept files
2. A JSON file with all extracted concept information

### Example Concept Markdown

```markdown
---
start-date: "2025-03-25"
deadline: "2025-03-29"
---

## Category
Machine Learning Algorithm

## Time Needed
3-5 days

## Prerequisites
- Linear Algebra
- Probability Theory

## Explanation
Detailed explanation of the concept...

## Learning Resources
- Resource 1
- Resource 2

## Practice Questions
- Question 1
- Question 2

---
#concept #machine-learning-algorithm
```

## Dependencies

- Python 3.8+
- langchain_openai
- pydantic
- langgraph

## License

MIT License
