import os
from pathlib import Path
from typing import List, Set

def update_knowledge_index_from_folder(folder_path: str) -> None:
    """
    Go through all .md files in the given folder, extract their names,
    and add all unique names to ./knowledge_index.md
    
    Args:
        folder_path: Path to the folder containing .md files
    """
    # Convert to Path object
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Get all .md files in the folder
    md_files = list(folder.glob("*.md"))
    
    # Extract filenames without extension
    file_names = [file.stem for file in md_files]
    
    # Read existing knowledge index if it exists
    knowledge_index_path = Path('./knowledge_index.md')
    existing_concepts: Set[str] = set()
    
    if knowledge_index_path.exists():
        with open(knowledge_index_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                existing_concepts = {concept.strip() for concept in content.split(',')}
    
    # Add new file names to existing concepts
    all_concepts = existing_concepts.union(set(file_names))
    
    # Write back to knowledge_index.md
    with open(knowledge_index_path, 'w', encoding='utf-8') as f:
        f.write(', '.join(sorted(all_concepts)))
    
    # Print summary
    new_concepts = set(file_names) - existing_concepts
    print(f"Added {len(new_concepts)} new concepts to knowledge_index.md")
    print(f"Total concepts in knowledge_index.md: {len(all_concepts)}")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("Usage: python UpdateKnowledgeIndex.py <folder_path>")
            print("Description: Updates knowledge_index.md with names of all .md files in the specified folder")
            print("Example: python UpdateKnowledgeIndex.py ./notes")
        else:
            folder_path = sys.argv[1]
            update_knowledge_index_from_folder(folder_path)
    else:
        print("Usage: python UpdateKnowledgeIndex.py <folder_path>")
        print("Use --help for more information")
