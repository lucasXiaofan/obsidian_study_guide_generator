import datetime
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional, Tuple

def parse_time_needed(time_str: str) -> int:
    """Parse the time_needed string and convert to approximate days.
    
    Args:
        time_str: String describing estimated learning time (e.g., "2-3 days", "1 week")
        
    Returns:
        int: Estimated number of days
    """
    time_str = time_str.lower()
    
    # Handle ranges like "2-3 days"
    if "-" in time_str and "day" in time_str:
        parts = time_str.split("-")
        if len(parts) == 2 and parts[1].strip().startswith("day"):
            try:
                # Take the higher end of the range
                days = int(parts[1].split()[0].strip())
                return days
            except ValueError:
                pass
    
    # Check for days
    if "day" in time_str:
        try:
            days = int(''.join(c for c in time_str if c.isdigit()))
            return days
        except ValueError:
            pass
    
    # Check for weeks
    if "week" in time_str:
        try:
            weeks = int(''.join(c for c in time_str if c.isdigit()))
            return weeks * 7
        except ValueError:
            pass
    
    # Check for hours
    if "hour" in time_str:
        try:
            hours = int(''.join(c for c in time_str if c.isdigit()))
            return max(1, hours // 8)  # Convert hours to days, minimum 1 day
        except ValueError:
            pass
    
    # Default to 3 days if parsing fails
    return 3

def calculate_dates(concepts: List[Dict[str, Any]]) -> List[Tuple[datetime.date, datetime.date]]:
    """Calculate start and deadline dates for each concept based on their order and estimated time.
    
    Args:
        concepts: List of concept dictionaries with time_needed field
        
    Returns:
        List of (start_date, deadline) tuples for each concept
    """
    today = datetime.date.today()
    current_date = today
    dates = []
    total_days = 0
    
    for concept in concepts:
        # Parse the time needed string to get number of days
        days_needed = parse_time_needed(concept['time_needed'])
        
        # Calculate deadline (start_date + days_needed)
        start_date = current_date
        total_days += days_needed
        deadline = start_date + datetime.timedelta(days=days_needed)
        
        dates.append((start_date, deadline))
        
        # Set the next start date to the day after the deadline
        current_date = deadline + datetime.timedelta(days=1)
    
    return dates, total_days

def create_markdown_file(concept: dict, output_dir: Path, start_date: datetime.date = None, deadline: datetime.date = None, concept_mapping: Dict[str, str] = None, existing_concepts: List[str] = None):
    """Create a markdown file for a single concept."""
    # Clean the concept name to be used as filename
    filename = concept['concept_name'].replace('/', '-').replace('\\', '-').replace(' ', '_')
    filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-'))
    
    file_path = output_dir / f"{filename}.md"
    
    # Format prerequisites as Obsidian links if they exist in knowledge_index or as plain text if not
    prerequisite_lines = []
    for prerequisite in concept['prerequisites']:
        # Clean prerequisite name for filename
        clean_prereq = prerequisite.replace('/', '-').replace('\\', '-').replace(' ', '_')
        clean_prereq = ''.join(c for c in clean_prereq if c.isalnum() or c in ('_', '-'))
        
        # Check if prerequisite exists in knowledge_index or is in current batch of concepts
        
        prerequisite_lines.append(f"- [[{clean_prereq}|{prerequisite}]]")
        
    
    prerequisites = '\n'.join(prerequisite_lines)
    
    # Format learning resources as a bulleted list
    resources = '\n'.join(f"- {resource}" for resource in concept['learning_resources'])
    
    # Format questions as a bulleted list
    questions = '\n'.join(f"- [ ] {question}" for question in concept['questions'])
    
    # Format dates for front matter
    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else datetime.date.today().strftime("%Y-%m-%d")
    deadline_str = deadline.strftime("%Y-%m-%d") if deadline else (datetime.date.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
    
    # Create the markdown content with front matter and Obsidian formatting
    content = f"""---
start-date: "{start_date_str}"
deadline: "{deadline_str}"
---

## Category
{concept['category']}

## Time Needed
{concept['time_needed']}

## Prerequisites
{prerequisites}

## Explanation
{concept['explanation']}

## Learning Resources
{resources}

## Practice Questions
{questions}

## Notes

---
#concept #{concept['category'].lower().replace(' ', '-').replace(',', '')}
"""
    
    # Write the content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return file_path

def extract_title_from_filename(filename: str) -> str:
    """Extract a title from the input filename."""
    # Remove file extension and path
    base_name = Path(filename).stem
    # Replace hyphens with spaces, then replace spaces with underscores
    title = base_name.replace('-', ' ').replace(' ', '_')
    return title


def update_knowledge_index(processed_concepts: List[Dict[str, Any]], target_concept_name,filename_mapping: Dict[str, str] = None):
    """
    Update the knowledge_index.md file with newly processed concepts.
    Ensures that names stored in knowledge_index.md match the filenames used in Obsidian.
    
    Args:
        processed_concepts: List of processed concept dictionaries
        filename_mapping: Optional mapping of concept names to their filename-friendly versions
    """
    knowledge_index_path = Path('./knowledge_index.md')
    existing_concepts = []
    
    # Read existing concepts if the file exists
    if knowledge_index_path.exists():
        with open(knowledge_index_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                existing_concepts = [concept.strip() for concept in content.split(',')]
    
    # Extract concept names from the processed concepts
    # Use the same filename cleaning logic as in create_markdown_file for consistency
    new_concept_names = []
    for concept in processed_concepts:
        # If we have a filename mapping and this concept is in it, use that
        if filename_mapping and concept['concept_name'] in filename_mapping:
            clean_name = filename_mapping[concept['concept_name']]
        else:
            # Otherwise apply the same cleaning logic as in create_markdown_file
            clean_name = concept['concept_name'].replace('/', '-').replace('\\', '-').replace(' ', '_')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c in ('_', '-'))
        target_concept_name = target_concept_name.replace('/', '-').replace('\\', '-').replace(' ', '_')
        target_concept_name = ''.join(c for c in target_concept_name if c.isalnum() or c in ('_', '-'))
        new_concept_names.append(clean_name)
        new_concept_names.append(target_concept_name)
    
    # Combine existing and new concepts, removing duplicates
    all_concepts = list(set(existing_concepts + new_concept_names))
    
    # Write back to knowledge_index.md
    with open(knowledge_index_path, 'w', encoding='utf-8') as f:
        f.write(', '.join(all_concepts))
    
    print(f"Updated knowledge_index.md with {len(new_concept_names)} new concepts")
    print(f"Total concepts in knowledge_index.md: {len(all_concepts)}")