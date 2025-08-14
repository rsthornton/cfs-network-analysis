# Interstate Trade Analysis

Python tools for analyzing U.S. interstate commodity flows using the Census Bureau's Commodity Flow Survey (CFS) data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rsthornton/cfs-network-analysis/blob/main/analysis/interstate_commerce_analysis.ipynb)

## Dataset Information

**Data Source**: U.S. Census Bureau Commodity Flow Survey (CFS) 2017 Public Use File
- **Manual Download**: [CFS 2017 PUF CSV.zip](https://www2.census.gov/programs-surveys/cfs/datasets/2017/CFS%202017%20PUF%20CSV.zip) (~140MB)
- **Official Documentation**: [Census Bureau CFS Historical Datasets](https://www.census.gov/data/datasets/2017/econ/cfs/historical-datasets.html)
- **Auto-Download**: The Colab notebook automatically downloads and extracts the dataset

The dataset contains 5.9M records of interstate commodity shipments with survey weights for national estimation.

## Quick Start Options

### Option 1: Run in Google Colab (Recommended)
Click the badge above to launch the interactive analysis notebook. No setup required.

### Option 2: Local Installation

```bash
git clone https://github.com/rsthornton/cfs-network-analysis.git
cd cfs-network-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd analysis
python centrality_analysis.py --help
```

## Key Research Findings

- **Most Influential States**: TX, CA, NY dominate interstate commerce networks
- **Three-Level Analysis**: Macro (bridging), Meso (influence), Micro (distribution) power
- **Bilateral Flows**: $150B+ in trade between top state pairs
- **Network Structure**: Hub-and-spoke patterns with regional clusters

## Repository Structure

- `analysis/interstate_commerce_analysis.ipynb` - **Main Colab notebook** with complete analysis
- `analysis/centrality_analysis.py` - Three-level network centrality analysis
- `analysis/flow_extraction.py` - Bilateral commodity flow extraction
- `lit-review/` - Academic literature and methodology documentation
- `requirements.txt` - Python dependencies

## Academic Usage

This repository supports academic research with:
- **Reproducible analysis** using official Census Bureau data
- **Citable methodology** with three-level network framework
- **Publication-ready visualizations** for papers and presentations
- **Open source code** for peer review and extension

## Citation

If you use this analysis in your research, please cite:
```
Interstate Commerce Network Analysis Repository
https://github.com/rsthornton/cfs-network-analysis
Data: U.S. Census Bureau, Commodity Flow Survey 2017 Public Use File
```