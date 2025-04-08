# LexiTrace Temporal Analysis

This module analyzes how clinical trial innovations are evolving over time by processing documents grouped by year.

## Overview

The temporal analysis tool extends LexiTrace to:

1. Analyze clinical trial documents by year (2022-2024)
2. Track innovation adoption trends over time
3. Identify growing and declining innovation categories
4. Measure innovation complexity evolution
5. Generate comprehensive timeline-based reports

## Usage

### Basic Usage

```bash
python temporal_analysis.py clinical_trials lexicon.csv output
```

### Options

- `--threshold INT` - Matching threshold (default: 85)
- `--workers INT` - Number of parallel worker processes  
- `--verbose` - Show detailed debug output
- `--sample` - Run in sample mode with limited files per year
- `--sample-size INT` - Number of files to process per year in sample mode (default: 5)

### Sample Mode

To quickly test the analysis with a small subset of files:

```bash
python temporal_analysis.py clinical_trials lexicon.csv output --sample --sample-size 10
```

### Using the Helper Script

For convenience, you can use the provided helper script:

```bash
# Full analysis
python run_full_analysis.py

# Sample run with 10 files per year
python run_full_analysis.py --sample --sample-size 10
```

## Output

The analysis produces a timestamped directory with:

1. **Visualizations**
   - Overall innovation trend line
   - Category prevalence by year
   - Innovation category heatmap
   - Category growth charts
   - Innovation complexity trend
   - Term type usage over time

2. **Data Files**
   - Per-year term location data
   - Overall temporal metrics
   - Category statistics over time
   - Term type distribution by year

3. **Word Document Report**
   - Executive summary
   - Year-by-year analysis
   - Comparative innovation findings
   - Growth trends
   - Visualizations with explanations

## How It Works

The temporal analysis:

1. Organizes clinical trials by year folders
2. Processes PDFs in each year directory, applying the lexicon term matching
3. Tracks innovation adoption percentages by year
4. Analyzes how specific innovation categories grow or decline
5. Measures the complexity of innovations (multiple categories per file)
6. Examines primary vs. related term usage trends
7. Generates comparative visualizations and a detailed report

## Requirements

The same requirements as the main LexiTrace tool apply. 