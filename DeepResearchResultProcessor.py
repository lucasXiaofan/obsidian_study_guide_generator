import os

import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional, Tuple
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.managed.is_last_step import RemainingSteps

# Load environment variables


class LLMConfig:
    """Configuration class for LLM models"""
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        print(f"Using model: {model_name}, api_key: {api_key}")
        self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)

class ConceptInfo(BaseModel):
    """Pydantic model for concept information"""
    concept_name: str = Field(description="Name of the technical concept")
    prerequisites: List[str] = Field(description="List of prerequisite concepts or knowledge needed")
    learning_resources: List[str] = Field(description="List of learning resources and links")
    explanation: str = Field(description="Comprehensive explanation of the concept")
    questions: List[str] = Field(description="Questions to help understanding")
    time_needed: str = Field(description="Estimated time needed to understand the concept")
    category: str = Field(description="Category of the concept (e.g., calculus, NLP, dataset, Machine learning algorithm)")

class ConceptList(BaseModel):
    """Pydantic model for list of concepts"""
    concepts: List[str] 

class ConceptMapping(BaseModel):
    """Pydantic model for concept name mapping"""
    mapping: Dict[str, str] = Field(description="Mapping of original concept names to standardized names")

class WorkflowState(TypedDict):
    roadmap_content: str
    extracted_concepts: List[str]
    current_concept_index: int
    processed_concepts: List[Dict[str, Any]]
    final_output: Dict[str, Any]
    llm: Optional[ChatOpenAI]
    remainStep: RemainingSteps

def read_roadmap_file(file_path: str) -> str:
    """Read the roadmap markdown file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_concepts(state: WorkflowState) -> WorkflowState:
    print(f"current remain step: {state['remainStep']}")
    """Extract concept names from the roadmap content using OpenAI"""
    
    # Create structured LLM for concept extraction
    structured_llm = state['llm'].with_structured_output(ConceptList)
    
    # Extract concepts using structured output
    concepts_list = structured_llm.invoke(
        f"""
        You are an expert at identifying specific technical concepts from educational content.
        
        Please analyze the following learning roadmap and extract all distinct technical concepts.
        Focus on specific algorithms, techniques, architectures, and methodologies rather than broad categories.
        
        For example:
        - Instead of "Machine Learning", extract specific concepts like "Gradient Descent", "Support Vector Machines"
        - Instead of "Neural Networks", extract specific architectures like "Transformer", "LSTM", "ResNet"
        - Instead of "Computer Vision", extract specific techniques like "SigLIP", "YOLO", "ViT"
        
        If content is machine learning relevant, also extract relevant benchmarks like DPG-Bench, MT-bench, etc.
        
        Roadmap content:
        {state['roadmap_content']}
        output format: json
        {ConceptList.model_json_schema()}
        """
    )
    
    state["extracted_concepts"] = concepts_list.concepts
    state["current_concept_index"] = 0
    state["processed_concepts"] = []
    print(f"Extracted {len(concepts_list.concepts)} concepts from the roadmap, concepts: {concepts_list.concepts}")
    
    return state

def process_concept(state: WorkflowState) -> WorkflowState:
    print(f"current remain step: {state['remainStep']}")
    """Process a single concept using OpenAI with structured output"""
    if state["current_concept_index"] >= len(state["extracted_concepts"]):
        return state
        
    concept = state["extracted_concepts"][state["current_concept_index"]]
    
    # Create structured LLM for concept processing
    structured_llm = state['llm'].with_structured_output(ConceptInfo)
    
    # Process concept using structured output
    concept_info = structured_llm.invoke(
        f"""
        You are an expert at extracting detailed information about technical concepts from educational content.
        Use the roadmap content below as context:
        {state['roadmap_content']}
        Please analyze the following concept: "{concept}" and extract detailed information from the roadmap.
        
        For this concept, provide:
        1. Concept Name: A precise and well-defined term.
        2. Prerequisite Concepts: Names of concepts that should be understood beforehand. name the Prerequisite in the same style as the concept name.
        3. Learning Resources: working website Links to materials (articles, tutorials, videos).
        4. High-Level Explanation: A brief but insightful overview of the concept, why understand target concept in roadmap need this concept, and what are the application of this concept .
        5. Estimated Learning Time: An approximate duration required to grasp the concept (normally take days to fully understand).   
        6. Category: The domain to which the concept belongs (e.g., optimization algorithm, transformer architecture, reinforcement learning, dataset processing).
        
        
        output format: json
        {ConceptInfo.model_json_schema()}
        """
    )
    
    # Convert to dictionary for easier manipulation
    concept_info_dict = concept_info.model_dump()
    
    # Get all available concept names (from filtered concepts and knowledge index if it exists)
    available_concepts = state["extracted_concepts"].copy()
    if "existing_concepts" in state and state["existing_concepts"]:
        available_concepts.extend(state["existing_concepts"])
    
    # If we have prerequisites and available concepts, standardize the prerequisites
    if concept_info_dict["prerequisites"] and available_concepts:
        # Use LLM to check all prerequisites against all available concepts
        standardization_result = state['llm'].with_structured_output(ConceptList).invoke(
            f"""
            I have a list of prerequisite concepts and a list of available concepts. 
            I need to standardize the prerequisite names to match the available concept names if they refer to the same concept.
            
            Prerequisites: {concept_info_dict["prerequisites"]}
            
            Available Concepts: {available_concepts}
            
            For each prerequisite, check if it refers to the same concept as any of the available concepts but is written differently.
            Examples:
            - "MLP" should be updated to "Multiple Layer Perceptron" if available
            - "CNN" should be updated to "Convolutional Neural Network" if available
            - "RNN" should be updated to "Recurrent Neural Network" if available
            - "SVM" should be updated to "Support Vector Machine" if available
            
            where each prerequisite is either:
            1. The original prerequisite name if no match is found
            2. The matching available concept name if a match is found
            output format: json
            {ConceptList.model_json_schema()}
            
            """
        )
        
        try:
            # Parse the JSON response
            
            # Update the prerequisites in the concept info
            concept_info_dict["prerequisites"] = standardization_result.concepts
            print(f"Updated prerequisites: {concept_info_dict['prerequisites']}")
        except :
            print(f"Failed to parse LLM response as JSON: {standardization_result}")
            # Keep original prerequisites if parsing fails
    
    # Add the processed concept to the list
    state["processed_concepts"].append(concept_info_dict)
    state["current_concept_index"] += 1
    print(f"Processed concept: {concept_info}")
    return state

def filter_redundant_concepts(state: WorkflowState) -> WorkflowState:
    """
    Filter out redundant concepts by checking against existing concepts in knowledge_index.md
    and using LLM to identify and remove redundant concepts.
    """
    print("Filtering redundant concepts based on knowledge_index.md...")
    
    # Look for knowledge_index.md in the root directory
    knowledge_index_path = Path('./knowledge_index.md')
    existing_concepts = []
    
    # Check if knowledge_index.md exists
    if knowledge_index_path.exists():
        print(f"Found existing knowledge index at {knowledge_index_path}")
        # Read the knowledge index file
        with open(knowledge_index_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Parse comma-separated concepts
        if content:
            existing_concepts = [concept.strip() for concept in content.split(',')]
            print(f"Found {len(existing_concepts)} existing concepts: {existing_concepts}")
    else:
        print("No knowledge_index.md found in the root directory.")
    
    # If no existing concepts found, just return the original state
    if not existing_concepts:
        print("No existing concepts found to filter against.")
        return state

    # Use LLM to filter out redundant concepts
    structured_llm = state['llm'].with_structured_output(ConceptList)
    
    # Extract concepts using structured output
    filtered_concepts = structured_llm.invoke(
        f"""
        You are an expert at identifying redundant technical concepts.
        
        I have a list of newly extracted concepts and a list of existing concepts.
        Please analyze both lists and remove any concepts from the new list that are redundant
        or already covered by the existing concepts.
        
        A concept is considered redundant if:
        1. It's identical or nearly identical to an existing concept
        2. It's a subset of an existing concept
        3. It's just a different name for the same concept
        
        New concepts: {state['extracted_concepts']}
        
        Existing concepts: {existing_concepts}
        
        Return only the non-redundant concepts from the new list.
        
        output format: json
        {ConceptList.model_json_schema()}
        """
    )
    
    # Update the state with filtered concepts
    original_count = len(state["extracted_concepts"])
    state["extracted_concepts"] = filtered_concepts.concepts
    filtered_count = len(filtered_concepts.concepts)
    
    print(f"Filtered concepts: {original_count} -> {filtered_count}")
    print(f"Remaining concepts after filtering: {filtered_concepts.concepts}")
    
    return state

def standardize_prerequisites(state: WorkflowState) -> WorkflowState:
    """
    Standardize prerequisite concept names by checking if they exist in extracted_concepts or knowledge_index.
    Updates prerequisites to match existing concept names for proper Obsidian linking.
    """
    print("Standardizing prerequisites for concepts...")
    
    # Get current extracted concepts (already filtered for redundancy)
    extracted_concepts = state["extracted_concepts"].copy()
    
    # Look for knowledge_index.md in the root directory
    knowledge_index_path = Path('./knowledge_index.md')
    existing_concepts = []
    
    # Check if knowledge_index.md exists
    if knowledge_index_path.exists():
        print(f"Found existing knowledge index at {knowledge_index_path}")
        # Read the knowledge index file
        with open(knowledge_index_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Parse comma-separated concepts
        if content:
            existing_concepts = [concept.strip() for concept in content.split(',')]
            print(f"Found {len(existing_concepts)} existing concepts: {existing_concepts}")
    
    # Store existing concepts in state for later use
    state["existing_concepts"] = existing_concepts
    
    # Combine extracted concepts and existing concepts to get all available concepts
    available_concepts = extracted_concepts.copy()
    if existing_concepts:
        available_concepts.extend(existing_concepts)
    
    # Create a dictionary to store filename-friendly versions of concepts
    # This ensures consistency between knowledge_index and Obsidian filenames
    filename_mapping = {}
    for concept in available_concepts:
        # Create filename-friendly version (same logic as in create_markdown_file)
        clean_name = concept.replace('/', '-').replace('\\', '-').replace(' ', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c in ('_', '-'))
        filename_mapping[concept] = clean_name
    
    # Store the filename mapping for use in create_markdown_file
    state["filename_mapping"] = filename_mapping
    
    # No concept mapping needed as we're handling prerequisite standardization in process_concept
    
    print(f"Created filename mapping for {len(filename_mapping)} concepts")
    print(f"Available concepts for prerequisites: {available_concepts}")
    
    return state

def update_knowledge_index(processed_concepts: List[Dict[str, Any]], filename_mapping: Dict[str, str] = None):
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
        
        new_concept_names.append(clean_name)
    
    # Combine existing and new concepts, removing duplicates
    all_concepts = list(set(existing_concepts + new_concept_names))
    
    # Write back to knowledge_index.md
    with open(knowledge_index_path, 'w', encoding='utf-8') as f:
        f.write(', '.join(all_concepts))
    
    print(f"Updated knowledge_index.md with {len(new_concept_names)} new concepts")
    print(f"Total concepts in knowledge_index.md: {len(all_concepts)}")

def should_continue_processing(state: WorkflowState) -> str:
    """Determine if we should continue processing concepts or finalize the output"""
    print(f"current remain step in should_continue_processing: {state['remainStep']} is < 2? {state['remainStep'] < 2}")
    if state["current_concept_index"]<len(state["extracted_concepts"]) and state['remainStep'] >= 2:
        return "process_concept"
    else:
        return "finalize_output"

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
    
    for concept in concepts:
        # Parse the time needed string to get number of days
        days_needed = parse_time_needed(concept['time_needed'])
        
        # Calculate deadline (start_date + days_needed)
        start_date = current_date
        deadline = start_date + datetime.timedelta(days=days_needed)
        
        dates.append((start_date, deadline))
        
        # Set the next start date to the day after the deadline
        current_date = deadline + datetime.timedelta(days=1)
    
    return dates

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
    questions = '\n'.join(f"- {question}" for question in concept['questions'])
    
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

def finalize_output(state: WorkflowState, input_file: str) -> WorkflowState:
    """Finalize the output by organizing all processed concepts and creating markdown files"""
    final_output = {
        "concepts": state["processed_concepts"]
    }
    
    # Generate title from input filename
    title = extract_title_from_filename(input_file)
    
    # Create output directory if it doesn't exist
    output_dir = Path(f'./{title}')
    output_dir.mkdir(exist_ok=True, parents=True)
    

    
    # Create an index file that links to all concepts
    index_content = [f"# {title} Concepts\n\nThis vault contains detailed notes on key concepts from the {title} content.\n\n## Concept List\n"]
    
    # Calculate start and deadline dates for each concept
    dates = calculate_dates(state["processed_concepts"])
    
    # Get concept mapping and existing concepts from state if available
    concept_mapping = state.get("concept_mapping", {})
    existing_concepts = state.get("existing_concepts", [])
    filename_mapping = state.get("filename_mapping", {})
    
    # Process each concept
    for i, concept in enumerate(state["processed_concepts"]):
        start_date, deadline = dates[i]
        file_path = create_markdown_file(
            concept, 
            output_dir, 
            start_date, 
            deadline, 
            concept_mapping, 
            existing_concepts
        )
        # Add link to index
        index_content.append(f"- [[{file_path.stem}|{concept['concept_name']}]]")
    
    # Write the index file
    with open(output_dir / f'{title}_index.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(index_content))
    
    print(f"Created markdown files in {output_dir}")
    print("Created index.md with links to all concepts")
    
    # Update the knowledge_index.md file with the newly processed concepts
    update_knowledge_index(state["processed_concepts"], filename_mapping)
    
    state["final_output"] = final_output
    return state

def build_workflow() -> StateGraph:
    """Build the langgraph workflow for processing the roadmap"""
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("extract_concepts", extract_concepts)
    workflow.add_node("filter_redundant_concepts", filter_redundant_concepts)
    workflow.add_node("standardize_prerequisites", standardize_prerequisites)
    workflow.add_node("process_concept", process_concept)
    
    workflow.add_edge("extract_concepts", "filter_redundant_concepts")
    workflow.add_edge("filter_redundant_concepts", "standardize_prerequisites")
    workflow.add_edge("standardize_prerequisites", "process_concept")
    # Create a closure to pass recursion_limit to should_continue_processing
    
    workflow.add_conditional_edges(
        "process_concept",
        should_continue_processing,
        {
            "process_concept": "process_concept",
            "finalize_output": END
        }
    )
    
    workflow.set_entry_point("extract_concepts")
    
    return workflow

def main(input_file: str = "default.md", model_name: Optional[str] = None, api_key: Optional[str] = None, recursion_limit: int = 25):
    """Main function to run the workflow with configurable model and input file"""
    # Set the API key in environment variables if provided
    # if api_key:
    #     os.environ["OPENAI_API_KEY"] = api_key
        
    roadmap_content = read_roadmap_file(input_file)
    
    # Initialize LLM configuration with specified model or default
    
    initial_state = WorkflowState(
        roadmap_content=roadmap_content,
        extracted_concepts=[],
        current_concept_index=0,
        processed_concepts=[],
        final_output={},
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key) if api_key else None
    )
    
    workflow = build_workflow()
    app = workflow.compile()
    print(f"Starting workflow with model: {model_name}")
    print(f"Processing roadmap content from {input_file}...{roadmap_content[:100]}")
    final_state = app.invoke(initial_state,config={"recursion_limit": recursion_limit})
    
    # Generate title from input filename for output path
    title = extract_title_from_filename(input_file)
    
    # Call finalize_output manually after the workflow completes
    final_state = finalize_output(final_state, input_file)
    

    print(f"Processed {len(final_state['processed_concepts'])} concepts.")
    print(f"Markdown files created in the ./{title}/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a roadmap markdown file and generate concept notes")
    parser.add_argument("--input", "-i", type=str, default="jina_result_.md", 
                        help="Input markdown file path (default: jina_result_.md)")
    parser.add_argument("--model", "-m", type=str, default='gpt-4o-mini',
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--api-key", "-k", type=str, 
                        help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--recursion-limit", "-r", type=int, default=25,
                        help="Maximum number of concepts to process (default: 25)")
    
    args = parser.parse_args()
    print(f"parse api_key: {args.api_key}")
    main(input_file=args.input, model_name=args.model, api_key=args.api_key, recursion_limit=args.recursion_limit)
