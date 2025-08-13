# Interstate Commerce Analysis Scripts

Python scripts for analyzing U.S. Commodity Flow Survey data using network centrality measures and flow extraction.

*Setup instructions are in the main README.md*

## Usage

**Network centrality analysis:**
```bash
python centrality_analysis.py --data cfs_2017_puf.csv --sample 50000
```

**Flow extraction:**
```bash
python flow_extraction.py --states CA,TX,NY --sample 50000
python flow_extraction.py --examples
```

**Quick test (5K sample):**
```bash
python centrality_analysis.py --data cfs_2017_puf.csv --sample 5000 --visualize
python flow_extraction.py --data cfs_2017_puf.csv --states CA,TX,IL --sample 5000 --visualize
```

## Data

Requires CFS 2017 Public Use File from Census Bureau. Place `cfs_2017_puf.csv` in this directory.

## Scripts

- `centrality_analysis.py` - Three-level network analysis (Jang & Yang 2023 framework)
- `flow_extraction.py` - Bilateral flow extraction and visualization