#!/usr/bin/env python3
"""
Tree Generator for Nomadica Annotation Tool

This script processes flattened JSON data from Atlas and generates hierarchical tree structures
for each paper, saving them as individual JSON files in a trees/ folder.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional


def nest_flat_keys(flat_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flattened keys back to nested structure.
    Example: 'paper data_sample dataset_type' -> {'paper': {'data_sample': {'dataset_type': value}}}
    """
    nested = {}
    
    for key, val in flat_obj.items():
        segments = key.split(" ")
        curr = nested
        
        for i, segment in enumerate(segments):
            # Remove "_truth" suffix if present
            if segment.endswith("_truth"):
                segment = segment[:-6]
            
            if i == len(segments) - 1:
                curr[segment] = val
            else:
                if segment not in curr or curr[segment] is None:
                    curr[segment] = {}
                curr = curr[segment]
    
    return nested


def create_tree_node(name: str, value: Any, children: List[Dict] = None) -> Dict[str, Any]:
    """Create a tree node structure compatible with the visualizer."""
    node = {
        "name": name,
        "value": str(value) if value is not None else "",
        "children": children or []
    }
    return node


def build_paper_tree(paper_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a hierarchical tree structure from paper data.
    """
    # Extract paper title
    paper_title = paper_data.get('paper title', 'Unknown Paper')
    
    # Create root node
    root = create_tree_node("Paper", paper_title)
    
    # Process flattened data directly
    data_sample_info = {}
    experiment_info = {}
    
    # Collect all data sample related fields
    for key, value in paper_data.items():
        if key.startswith('paper data_sample'):
            remaining_key = key.replace('paper data_sample ', '')
            if remaining_key:
                data_sample_info[remaining_key] = value
        elif key.startswith('paper experiments'):
            remaining_key = key.replace('paper experiments ', '')
            if remaining_key:
                experiment_info[remaining_key] = value
    
    # Add data sample information
    if data_sample_info:
        data_sample_node = create_tree_node("Data Sample", "Dataset Information")
        
        # Dataset type
        if 'dataset_type' in data_sample_info:
            data_sample_node['children'].append(
                create_tree_node("Dataset Type", data_sample_info['dataset_type'])
            )
        
        # Provider
        if 'provider_name' in data_sample_info:
            data_sample_node['children'].append(
                create_tree_node("Provider", data_sample_info['provider_name'])
            )
        
        # Filters
        filter_keys = [k for k in data_sample_info.keys() if 'filters' in k]
        if filter_keys:
            filters_node = create_tree_node("Filters", f"{len([k for k in filter_keys if 'filter_type' in k])} filters")
            
            # Group filter information
            filter_types = [k for k in filter_keys if 'filter_type' in k]
            for i, filter_type_key in enumerate(filter_types):
                filter_type = data_sample_info.get(filter_type_key, 'Unknown')
                filter_key_key = filter_type_key.replace('filter_type', 'filter_key')
                filter_key = data_sample_info.get(filter_key_key, 'Unknown')
                filter_expr_key = filter_type_key.replace('filter_type', 'filter_expression')
                filter_expr = data_sample_info.get(filter_expr_key, 'Unknown')
                
                filter_node = create_tree_node(f"Filter {i+1}", f"{filter_type}: {filter_key}")
                filter_node['children'].append(
                    create_tree_node("Expression", filter_expr)
                )
                filters_node['children'].append(filter_node)
            
            data_sample_node['children'].append(filters_node)
        
        # Completeness statistics
        stat_keys = [k for k in data_sample_info.keys() if 'completeness_statistic' in k]
        if stat_keys:
            stats_node = create_tree_node("Completeness Statistics", f"{len([k for k in stat_keys if 'name' in k])} statistics")
            
            stat_names = [k for k in stat_keys if 'name' in k]
            for i, stat_name_key in enumerate(stat_names):
                stat_name = data_sample_info.get(stat_name_key, 'Unknown')
                stat_value_key = stat_name_key.replace('name', 'value')
                stat_value = data_sample_info.get(stat_value_key, 'Unknown')
                
                stats_node['children'].append(
                    create_tree_node(stat_name, stat_value)
                )
            
            data_sample_node['children'].append(stats_node)
        
        root['children'].append(data_sample_node)
    
    # Add experiments information
    if experiment_info:
        experiments_node = create_tree_node("Experiments", "Research Experiments")
        
        # Domain
        if 'domain' in experiment_info:
            experiments_node['children'].append(
                create_tree_node("Domain", experiment_info['domain'])
            )
        
        # Description
        if 'description' in experiment_info:
            desc = experiment_info['description']
            if len(desc) > 100:
                desc = desc[:100] + "..."
            experiments_node['children'].append(
                create_tree_node("Description", desc)
            )
        
        # Code location
        if 'code_location' in experiment_info and experiment_info['code_location']:
            experiments_node['children'].append(
                create_tree_node("Code Location", experiment_info['code_location'])
            )
        
        # Mobility metrics
        metric_keys = [k for k in experiment_info.keys() if 'mobility_metric' in k]
        if metric_keys:
            metrics_node = create_tree_node("Mobility Metrics", "Analysis Metrics")
            
            # Group metric information
            metric_names = [k for k in metric_keys if 'metric_name' in k]
            for i, metric_name_key in enumerate(metric_names):
                metric_name = experiment_info.get(metric_name_key, 'Unknown')
                
                metric_node = create_tree_node(f"Metric {i+1}", metric_name)
                
                # Temporal aggregation
                temp_agg_key = metric_name_key.replace('metric_name', 'temporal_aggregation')
                if temp_agg_key in experiment_info:
                    metric_node['children'].append(
                        create_tree_node("Temporal Aggregation", experiment_info[temp_agg_key])
                    )
                
                # Spatial aggregation
                spatial_agg_key = metric_name_key.replace('metric_name', 'spatial_aggregation')
                if spatial_agg_key in experiment_info:
                    metric_node['children'].append(
                        create_tree_node("Spatial Aggregation", experiment_info[spatial_agg_key])
                    )
                
                # Processing steps
                step_name_key = metric_name_key.replace('metric_name', 'processing_step name')
                step_type_key = metric_name_key.replace('metric_name', 'processing_step type')
                
                if step_name_key in experiment_info or step_type_key in experiment_info:
                    steps_node = create_tree_node("Processing Steps", "Data Processing")
                    
                    if step_name_key in experiment_info:
                        step_name = experiment_info[step_name_key]
                        step_type = experiment_info.get(step_type_key, 'Unknown')
                        steps_node['children'].append(
                            create_tree_node(step_name, step_type)
                        )
                    
                    metric_node['children'].append(steps_node)
                
                metrics_node['children'].append(metric_node)
            
            experiments_node['children'].append(metrics_node)
        
        root['children'].append(experiments_node)
    
    # Add metadata
    metadata_node = create_tree_node("Metadata", "Paper Information")
    if '_paper_id' in paper_data:
        metadata_node['children'].append(
            create_tree_node("Paper ID", paper_data['_paper_id'])
        )
    if '_version' in paper_data:
        metadata_node['children'].append(
            create_tree_node("Version", str(paper_data['_version']))
        )
    if 'created_at' in paper_data:
        metadata_node['children'].append(
            create_tree_node("Created At", paper_data['created_at'])
        )
    if '_result_id' in paper_data:
        metadata_node['children'].append(
            create_tree_node("Result ID", paper_data['_result_id'])
        )
    
    root['children'].append(metadata_node)
    
    return root


def group_by_paper(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group flattened data by paper ID."""
    papers = defaultdict(list)
    
    for item in data:
        paper_id = item.get('_paper_id', 'unknown')
        papers[paper_id].append(item)
    
    return dict(papers)


def generate_trees(input_data: List[Dict[str, Any]], output_dir: str = "trees") -> Dict[str, str]:
    """
    Generate tree files for each paper.
    
    Args:
        input_data: List of flattened JSON objects
        output_dir: Directory to save tree files
    
    Returns:
        Dictionary mapping paper IDs to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by paper
    papers = group_by_paper(input_data)
    
    generated_files = {}
    
    for paper_id, paper_items in papers.items():
        # Use the first item to get paper title and basic info
        first_item = paper_items[0]
        paper_title = first_item.get('paper title', f'Paper {paper_id}')
        
        # Build tree structure
        tree = build_paper_tree(first_item)
        
        # Generate filename
        filename = f"tree_{paper_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save tree to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        
        generated_files[paper_id] = {
            'filepath': filepath,
            'title': paper_title,
            'items_count': len(paper_items)
        }
        
        print(f"Generated tree for paper: {paper_title}")
        print(f"  File: {filepath}")
        print(f"  Items: {len(paper_items)}")
        print()
    
    return generated_files


def main():
    """Main function to process input data and generate trees."""
    
    # Import example data
    from example_data import example_data
    
    print("Generating trees from input data...")
    generated_files = generate_trees(example_data)
    
    print(f"\nGenerated {len(generated_files)} tree files:")
    for paper_id, info in generated_files.items():
        print(f"  {paper_id}: {info['title']} -> {info['filepath']}")
    
    # Create a summary file with all paper information
    summary_file = os.path.join("trees", "papers_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(generated_files, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
