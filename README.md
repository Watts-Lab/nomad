# Nomadica Annotation Tool

A web-based tool for visualizing and annotating hierarchical tree structures generated from Atlas paper data.

## Overview

This tool processes flattened JSON data from the Atlas project and generates hierarchical tree visualizations for each paper. Users can search for papers, visualize their tree structures, and annotate them for quality control.

## Files Structure

```
nomad/
├── generate_trees.py          # Python script to process data and generate trees
├── example_data.py           # Sample data for testing
├── tree-visualizer.js        # JavaScript tree visualization engine
├── annotation_tool.html     # Main web interface
├── trees/                    # Generated tree files
│   ├── tree_<PAPER_ID>.json # Individual tree files
│   └── papers_summary.json  # Summary of all papers
└── README.md                # This file
```

## Quick Start

### 1. Generate Tree Files

First, run the Python script to process your data and generate tree files:

```bash
python generate_trees.py
```

This will:
- Process the example data from `example_data.py`
- Generate individual tree files in the `trees/` folder
- Create a summary file with paper metadata

### 2. Open the Web Interface

Open `annotation_tool.html` in your web browser to access the annotation tool.

### 3. Using the Interface

- **Search**: Use the search bar to find papers by title
- **Random Paper**: Click "Random Paper" to load a random tree
- **Visualize**: Click on any paper from search results to visualize its tree
- **Navigate**: Use mouse to pan and zoom the tree visualization
- **Annotate**: Click "Annotate" to open the annotation form

## Customizing for Your Data

### Using Your Own Data

1. Replace the data in `example_data.py` with your actual Atlas results
2. Modify `generate_trees.py` if you need different tree structures
3. Run the generation script to create your tree files

### Data Format

Your input data should be a list of dictionaries with flattened keys like:
```python
[
    {
        'paper title': 'Your Paper Title',
        'paper data_sample dataset_type': 'mobility',
        'paper data_sample provider_name': 'carrier',
        'paper experiments domain': 'analysis',
        'paper experiments description': 'Your experiment description',
        '_paper_id': 'unique_paper_id',
        '_version': 1,
        'created_at': '2025-01-01 12:00:00'
    }
]
```

## Features

### Tree Visualization
- Interactive tree layout with pan and zoom
- Color-coded nodes by type (Paper, Data, Experiments, Metadata)
- Hover effects and node selection
- Responsive design for different screen sizes

### Search Functionality
- Search papers by title
- Real-time filtering
- Random paper selection

### Annotation System
- Form-based annotation interface
- Support for different annotation types:
  - Incorrect Subtree
  - Missing Subtree
  - General Feedback
- Local storage of annotations (extend to server-side as needed)

## Technical Details

### Tree Structure

Each generated tree follows this hierarchy:
```
Paper (Root)
├── Data Sample
│   ├── Dataset Type
│   ├── Provider
│   ├── Filters
│   └── Completeness Statistics
├── Experiments
│   ├── Domain
│   ├── Description
│   ├── Code Location
│   └── Mobility Metrics
│       ├── Metric Name
│       ├── Temporal Aggregation
│       ├── Spatial Aggregation
│       └── Processing Steps
└── Metadata
    ├── Paper ID
    ├── Version
    ├── Created At
    └── Result ID
```

### Browser Compatibility

The tool works in modern browsers that support:
- HTML5 Canvas
- ES6 JavaScript features
- CSS Grid and Flexbox

## Extending the Tool

### Adding New Node Types

1. Update the `getNodeColor()` method in `tree-visualizer.js`
2. Add corresponding colors to the `colors` object
3. Modify the tree generation logic in `generate_trees.py`

### Server Integration

To save annotations to a server:
1. Replace the localStorage code in `annotation_tool.html`
2. Add API endpoints for saving/loading annotations
3. Implement user authentication if needed

### Custom Styling

Modify the CSS in `annotation_tool.html` to match your design requirements.

## Troubleshooting

### No Trees Generated
- Check that your input data has the correct format
- Ensure `_paper_id` fields are present
- Verify the Python script runs without errors

### Trees Not Loading in Browser
- Check browser console for JavaScript errors
- Ensure tree files are in the `trees/` folder
- Verify file permissions

### Search Not Working
- Check that `papers_summary.json` exists
- Verify the summary file has the correct format

## Contributing

This is a simple, straightforward implementation designed for the Atlas project. Feel free to extend and modify as needed for your specific requirements.

