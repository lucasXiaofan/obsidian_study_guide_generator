import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from HelperFunctions import create_markdown_file, update_knowledge_index

def extract_json_from_markdown(markdown_path: str) -> str:
    """
    Extract JSON content from a markdown file.
    
    Args:
        markdown_path: Path to the markdown file containing JSON
        
    Returns:
        str: Extracted JSON content
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to extract JSON content from the markdown file
    # This assumes the entire file is JSON or the JSON is enclosed in code blocks
    if content.strip().startswith('{') and content.strip().endswith('}'):
        # The entire file is JSON
        return content
    
    # Try to extract JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        return json_match.group(1)
    
    # If no JSON is found, return the content as is
    return content

def process_json_to_obsidian_folder(json_content: str, existing_concepts: List[str] = None) -> str:
    """
    Process a JSON string in markdown format and create an Obsidian folder structure
    with dedicated pages for the main concept and prerequisite concepts.
    
    Args:
        json_content: JSON string containing main_topic and prerequisites
        
    Returns:
        str: Path to the created folder
    """
    # Parse the JSON content
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON content provided")
    
    # Extract main topic information
    if "main_topic" not in data:
        raise ValueError("JSON must contain a 'main_topic' field")
    
    main_topic = data["main_topic"]
    main_concept_name = main_topic["concept_name"]
    
    # Create a clean folder name
    folder_name = main_concept_name.replace('/', '-').replace('\\', '-').replace(' ', '_')
    folder_name = ''.join(c for c in folder_name if c.isalnum() or c in ('_', '-'))
    
    # Get existing concepts from knowledge_index.md if not provided
    if existing_concepts is None:
        existing_concepts = []
        knowledge_index_path = Path('./knowledge_index.md')
        if knowledge_index_path.exists():
            with open(knowledge_index_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    existing_concepts = [concept.strip() for concept in content.split(',')]
    
    # Create the folder
    output_dir = Path(f'./{folder_name}')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert main topic to the format expected by create_markdown_file
    main_concept = {
        "concept_name": main_topic["concept_name"],
        "category": main_topic.get("category", ""),
        "explanation": main_topic.get("explanation_motivation", ""),
        "prerequisites": main_topic.get("prerequisites", []),
        "learning_resources": main_topic.get("learning_resources", []),
        "questions": main_topic.get("study_questions", []),
        "time_needed": main_topic.get("approximate_study_time", "3-5 days")  # Use approximate_study_time if available
    }
    
    # Check if main concept already exists in knowledge_index
    clean_main_name = main_concept_name.replace('/', '-').replace('\\', '-').replace(' ', '_')
    clean_main_name = ''.join(c for c in clean_main_name if c.isalnum() or c in ('_', '-'))
    
    # Create the main concept markdown file if it doesn't exist in knowledge_index
    if clean_main_name not in existing_concepts:
        main_file_path = create_markdown_file(main_concept, output_dir)
    
    # Process prerequisites if they exist
    processed_concepts = [main_concept]
    if "prerequisites" in data:
        for prereq_key, prereq_data in data["prerequisites"].items():
            # Convert prerequisite to the format expected by create_markdown_file
            prereq_concept = {
                "concept_name": prereq_data["concept_name"],
                "category": prereq_data.get("category", ""),
                "explanation": prereq_data.get("explanation_motivation", ""),
                "prerequisites": prereq_data.get("prerequisites", []),
                "learning_resources": prereq_data.get("learning_resources", []),
                "questions": prereq_data.get("study_questions", []),
                "time_needed": prereq_data.get("approximate_study_time", "2-3 days")  # Use approximate_study_time if available
            }
            
            # Clean prerequisite name for checking against knowledge_index
            clean_prereq_name = prereq_data["concept_name"].replace('/', '-').replace('\\', '-').replace(' ', '_')
            clean_prereq_name = ''.join(c for c in clean_prereq_name if c.isalnum() or c in ('_', '-'))
            
            # Create the prerequisite concept markdown file if it doesn't exist in knowledge_index
            if clean_prereq_name not in existing_concepts:
                prereq_file_path = create_markdown_file(prereq_concept, output_dir)
                processed_concepts.append(prereq_concept)
            else:
                print(f"Skipping creation of {clean_prereq_name}.md as it already exists in knowledge_index")
    
    # Create an index file named after the main concept
    index_content = [f"# {main_concept_name} Study Guide\n"]
    index_content.append(f"## Main Concept\n")
    index_content.append(f"- [[{folder_name}/{main_concept_name.replace('/', '-').replace('\\', '-').replace(' ', '_')}|{main_concept_name}]]")

    
    # Add benefits section if it exists
    if "benefits" in data:
        benefits = data["benefits"]
        index_content.append(f"\n## Benefits\n")
        
        if "career_advancement" in benefits:
            index_content.append(f"### Career Advancement\n{benefits['career_advancement']}\n\n")
        
        if "monetization_opportunities" in benefits:
            index_content.append(f"### Monetization Opportunities\n{benefits['monetization_opportunities']}\n\n")
        
        if "research_prospects" in benefits:
            index_content.append(f"### Research Prospects\n{benefits['research_prospects']}\n\n")
        
        if "limitations_and_ethical_considerations" in benefits:
            index_content.append(f"### Limitations and Ethical Considerations\n{benefits['limitations_and_ethical_considerations']}\n\n")
    
    # Add dataview queries
    index_content.append(f"""
## Concept List
```dataview
TABLE start-date as "start-date"
FROM "{folder_name}"
SORT file.name ASC
```

## TODOs
```dataview
TASK
FROM "{folder_name}"
WHERE !completed
SORT file.name ASC
```
""")
    
    # Write the index file named after the main concept
    with open(output_dir / f'{main_concept_name.replace("/", "-").replace("\\", "-").replace(" ", "_")}_index.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(index_content))
    
    # Update the knowledge index with the newly processed concepts
    update_knowledge_index(processed_concepts, main_concept_name)
    
    return str(output_dir)

def markdown_to_obsidian_folder(markdown_path: str) -> str:
    """
    Read a markdown file containing JSON and process it to create an Obsidian folder structure.
    
    Args:
        markdown_path: Path to the markdown file containing JSON
        
    Returns:
        str: Path to the created folder
    """
    # Get existing concepts from knowledge_index.md
    existing_concepts = []
    knowledge_index_path = Path('./knowledge_index.md')
    if knowledge_index_path.exists():
        with open(knowledge_index_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                existing_concepts = [concept.strip() for concept in content.split(',')]
    
    # Extract JSON content from markdown file
    json_content = extract_json_from_markdown(markdown_path)
    
    # Process the JSON content
    return process_json_to_obsidian_folder(json_content, existing_concepts)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert markdown with JSON to Obsidian folder structure")
    parser.add_argument("--input", "-i", type=str, default="deepresearch json.md", 
                        help="Input markdown file path containing JSON (default: deepresearch json.md)")
    
    args = parser.parse_args()
    
    output_dir = markdown_to_obsidian_folder(args.input)
    print(f"Created Obsidian folder structure at: {output_dir}")
