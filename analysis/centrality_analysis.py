#!/usr/bin/env python3
"""
Interstate Commerce Network Centrality Analysis

This module implements a three-level network analysis framework for U.S. interstate
commodity flows, based on the methodology from Jang & Yang (2023). The framework
examines network power at three hierarchical levels:

- Macro Level: Regional bridgeness (Betweenness Centrality)
- Meso Level: Influence networks (Eigenvector Centrality)  
- Micro Level: Distribution power (Weighted Out-Degree)

Usage:
    Terminal: python centrality_analysis.py --data cfs_2017_puf.csv --output results/
    Programmatic: 
        analyzer = CentralityAnalyzer("cfs_2017_puf.csv")
        results = analyzer.run_three_level_analysis()

Author: Shingai Thornton
Institution: Binghamton University Systems Science
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress NetworkX warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# State FIPS code mapping for display purposes
STATE_FIPS = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
    11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN',
    19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA',
    26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV',
    33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH',
    40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
    48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'
}


class CentralityAnalyzer:
    """
    Analyzes interstate trade networks using graph theory and centrality metrics.
    
    This class implements the three-level framework for understanding network power
    in interstate commerce systems. It handles data loading, network construction,
    centrality calculations, and analysis export.
    
    Attributes:
        data_path (Path): Path to the CFS dataset
        raw_data (pd.DataFrame): Original CFS data
        network_data (pd.DataFrame): Processed network data
        network (nx.DiGraph): NetworkX directed graph representation
        centrality_metrics (dict): Calculated centrality measures
        three_level_results (dict): Three-level analysis results
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with CFS data.
        
        Args:
            data_path (str): Path to the CFS CSV file
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.network_data = None
        self.network = None
        self.centrality_metrics = {}
        self.three_level_results = {}
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        print(f"Initialized CentralityAnalyzer with data: {self.data_path}")
    
    def load_and_process_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and preprocess CFS data for network analysis.
        
        Applies proper survey weighting and filters for interstate flows only.
        Calculates weighted values using the formula: Total Value = Σ(WGT_FACTOR × SHIPMT_VALUE)
        
        Args:
            sample_size (int, optional): Number of records to sample for testing
            
        Returns:
            pd.DataFrame: Processed network data ready for analysis
            
        Raises:
            ValueError: If required columns are missing
        """
        print("Loading CFS data...")
        
        # Load data with chunking for memory efficiency
        if sample_size:
            self.raw_data = pd.read_csv(self.data_path, nrows=sample_size)
            print(f"Loaded sample of {len(self.raw_data):,} records")
        else:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.raw_data):,} total records")
        
        # Verify required columns exist
        required_columns = ['ORIG_STATE', 'DEST_STATE', 'SHIPMT_VALUE', 'WGT_FACTOR', 'SHIPMT_WGHT']
        missing_cols = [col for col in required_columns if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Processing interstate flows...")
        
        # Filter for interstate flows only (exclude intrastate)
        interstate_flows = self.raw_data[
            self.raw_data['ORIG_STATE'] != self.raw_data['DEST_STATE']
        ].copy()
        
        print(f"Filtered to {len(interstate_flows):,} interstate flow records")
        
        # Apply survey weighting for proper statistical estimation
        interstate_flows['weighted_value'] = interstate_flows['WGT_FACTOR'] * interstate_flows['SHIPMT_VALUE']
        interstate_flows['weighted_tons'] = interstate_flows['WGT_FACTOR'] * interstate_flows['SHIPMT_WGHT'] / 2000  # Convert pounds to tons
        
        # Aggregate flows by state pair for network construction
        print("Aggregating flows by state pair...")
        
        self.network_data = interstate_flows.groupby(['ORIG_STATE', 'DEST_STATE']).agg({
            'weighted_value': 'sum',
            'weighted_tons': 'sum',
            'WGT_FACTOR': 'sum'  # Total survey weight
        }).reset_index()
        
        # Calculate value density (dollars per ton)
        self.network_data['value_density'] = self.network_data['weighted_value'] / self.network_data['weighted_tons']
        self.network_data['value_density'] = self.network_data['value_density'].replace([np.inf, -np.inf], 0)
        
        print(f"Created {len(self.network_data)} state-to-state flow relationships")
        
        return self.network_data
    
    def build_network(self, min_value_threshold: float = 0) -> nx.DiGraph:
        """
        Construct directed graph from interstate flow data.
        
        Creates a NetworkX directed graph where nodes are states and edges are
        weighted by total commodity flow value between state pairs.
        
        Args:
            min_value_threshold (float): Minimum flow value to include in network
            
        Returns:
            nx.DiGraph: Weighted directed graph of interstate flows
        """
        if self.network_data is None:
            raise ValueError("Must load and process data first using load_and_process_data()")
        
        print("Building interstate trade network...")
        
        # Filter by minimum value threshold if specified
        filtered_data = self.network_data[
            self.network_data['weighted_value'] >= min_value_threshold
        ] if min_value_threshold > 0 else self.network_data
        
        # Create directed graph
        self.network = nx.DiGraph()
        
        # Add edges with multiple weight attributes
        for _, row in filtered_data.iterrows():
            orig_state = int(row['ORIG_STATE'])
            dest_state = int(row['DEST_STATE'])
            
            self.network.add_edge(
                orig_state, 
                dest_state,
                weight=row['weighted_value'],  # Primary weight for centrality calculations
                tons=row['weighted_tons'],
                density=row['value_density'],
                survey_weight=row['WGT_FACTOR']
            )
        
        print(f"Network constructed: {self.network.number_of_nodes()} states, "
              f"{self.network.number_of_edges()} trade relationships")
        
        return self.network
    
    def calculate_centrality_metrics(self) -> Dict:
        """
        Calculate comprehensive centrality metrics for the network.
        
        Computes all centrality measures needed for the three-level framework:
        - Degree centralities (in, out, total)
        - Betweenness centrality (weighted)
        - Eigenvector centrality (weighted)
        - PageRank (weighted)
        - Closeness centrality
        
        Returns:
            dict: Dictionary containing all centrality measures by state
            
        Raises:
            ValueError: If network hasn't been built yet
        """
        if self.network is None:
            raise ValueError("Must build network first using build_network()")
        
        print("Calculating centrality metrics...")
        
        # Basic degree centralities
        self.centrality_metrics['in_degree'] = dict(self.network.in_degree(weight='weight'))
        self.centrality_metrics['out_degree'] = dict(self.network.out_degree(weight='weight'))
        self.centrality_metrics['total_degree'] = dict(self.network.degree(weight='weight'))
        
        # Normalize degree centralities for comparison
        total_weight = sum(self.centrality_metrics['total_degree'].values())
        self.centrality_metrics['in_degree_normalized'] = {
            state: degree / total_weight for state, degree in self.centrality_metrics['in_degree'].items()
        }
        self.centrality_metrics['out_degree_normalized'] = {
            state: degree / total_weight for state, degree in self.centrality_metrics['out_degree'].items()
        }
        
        # Betweenness centrality (normalized, weighted)
        print("  Computing betweenness centrality...")
        self.centrality_metrics['betweenness'] = nx.betweenness_centrality(
            self.network, weight='weight', normalized=True
        )
        
        # Eigenvector centrality (weighted)
        print("  Computing eigenvector centrality...")
        try:
            self.centrality_metrics['eigenvector'] = nx.eigenvector_centrality(
                self.network, weight='weight', max_iter=1000
            )
        except nx.NetworkXError:
            print("    Warning: Eigenvector centrality failed, using PageRank as alternative")
            self.centrality_metrics['eigenvector'] = nx.pagerank(
                self.network, weight='weight'
            )
        
        # PageRank (Google's algorithm, weighted)
        print("  Computing PageRank...")
        self.centrality_metrics['pagerank'] = nx.pagerank(
            self.network, weight='weight'
        )
        
        # Closeness centrality
        print("  Computing closeness centrality...")
        try:
            self.centrality_metrics['closeness'] = nx.closeness_centrality(
                self.network, distance='weight'
            )
        except:
            # Fallback for disconnected components
            self.centrality_metrics['closeness'] = {}
            for component in nx.strongly_connected_components(self.network):
                subgraph = self.network.subgraph(component)
                closeness_sub = nx.closeness_centrality(subgraph, distance='weight')
                self.centrality_metrics['closeness'].update(closeness_sub)
        
        print("Centrality metrics calculated successfully")
        return self.centrality_metrics
    
    def run_three_level_analysis(self, top_n: int = 10) -> Dict:
        """
        Execute the three-level network power analysis framework.
        
        Implements Jang & Yang (2023) methodology:
        - Macro Level: Betweenness centrality (regional bridgeness)
        - Meso Level: Eigenvector centrality (influence networks)
        - Micro Level: Weighted out-degree (distribution power)
        
        Args:
            top_n (int): Number of top states to identify at each level
            
        Returns:
            dict: Complete three-level analysis results
        """
        if not self.centrality_metrics:
            raise ValueError("Must calculate centrality metrics first")
        
        print("Running three-level network analysis...")
        
        # MACRO LEVEL: Betweenness Centrality (Regional Bridgeness)
        macro_leaders = sorted(
            self.centrality_metrics['betweenness'].items(),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        # MESO LEVEL: Eigenvector Centrality (Influence Networks)
        meso_leaders = sorted(
            self.centrality_metrics['eigenvector'].items(),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        # MICRO LEVEL: Weighted Out-Degree (Distribution Power)
        micro_leaders = sorted(
            self.centrality_metrics['out_degree_normalized'].items(),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        # Multi-level leader identification
        all_leaders = set([state for state, _ in macro_leaders + meso_leaders + micro_leaders])
        multi_level_leaders = []
        
        for state in all_leaders:
            levels = []
            total_score = 0
            
            # Check presence in each level's top performers
            macro_states = [s for s, _ in macro_leaders]
            meso_states = [s for s, _ in meso_leaders]
            micro_states = [s for s, _ in micro_leaders]
            
            if state in macro_states:
                rank = macro_states.index(state) + 1
                score = (top_n + 1 - rank) / top_n  # Score from 1.0 to 0.1
                levels.append(f"Macro-{rank}")
                total_score += score
            
            if state in meso_states:
                rank = meso_states.index(state) + 1
                score = (top_n + 1 - rank) / top_n
                levels.append(f"Meso-{rank}")
                total_score += score
            
            if state in micro_states:
                rank = micro_states.index(state) + 1
                score = (top_n + 1 - rank) / top_n
                levels.append(f"Micro-{rank}")
                total_score += score
            
            if len(levels) > 0:
                multi_level_leaders.append({
                    'state': state,
                    'state_name': STATE_FIPS.get(state, f"State-{state}"),
                    'level_count': len(levels),
                    'levels': levels,
                    'total_score': total_score
                })
        
        # Sort multi-level leaders by total score
        multi_level_leaders.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Store comprehensive results
        self.three_level_results = {
            'macro_level': {
                'description': 'Regional Bridgeness (Betweenness Centrality)',
                'metric': 'betweenness',
                'leaders': [(state, STATE_FIPS.get(state, f"State-{state}"), score) 
                           for state, score in macro_leaders]
            },
            'meso_level': {
                'description': 'Influence Networks (Eigenvector Centrality)',
                'metric': 'eigenvector',
                'leaders': [(state, STATE_FIPS.get(state, f"State-{state}"), score) 
                           for state, score in meso_leaders]
            },
            'micro_level': {
                'description': 'Distribution Power (Weighted Out-Degree)',
                'metric': 'out_degree_normalized',
                'leaders': [(state, STATE_FIPS.get(state, f"State-{state}"), score) 
                           for state, score in micro_leaders]
            },
            'multi_level_leaders': multi_level_leaders,
            'analysis_parameters': {
                'top_n': top_n,
                'total_states': len(self.network.nodes()),
                'total_flows': len(self.network.edges())
            }
        }
        
        print("Three-level analysis completed")
        return self.three_level_results
    
    def get_network_summary(self) -> Dict:
        """
        Generate summary statistics for the interstate trade network.
        
        Returns:
            dict: Network summary including nodes, edges, and trade values
        """
        if self.network is None or self.network_data is None:
            return {"error": "Network not built yet"}
        
        total_value = self.network_data['weighted_value'].sum()
        total_tons = self.network_data['weighted_tons'].sum()
        
        return {
            'nodes': self.network.number_of_nodes(),
            'edges': self.network.number_of_edges(),
            'total_value': total_value,
            'total_tons': total_tons,
            'avg_value_per_flow': total_value / len(self.network_data),
            'network_density': nx.density(self.network)
        }
    
    def analyze_single_state(self, state_code: int) -> Dict:
        """
        Analyze a single state's position in the three-level framework.
        
        Args:
            state_code (int): FIPS code of the state to analyze
            
        Returns:
            dict: State's rankings and scores across all three levels
        """
        if not self.three_level_results:
            raise ValueError("Must run three-level analysis first")
        
        state_name = STATE_FIPS.get(state_code, f"State-{state_code}")
        
        # Find rankings in each level
        macro_ranks = [i+1 for i, (state, _, _) in enumerate(self.three_level_results['macro_level']['leaders']) 
                      if state == state_code]
        meso_ranks = [i+1 for i, (state, _, _) in enumerate(self.three_level_results['meso_level']['leaders']) 
                     if state == state_code]
        micro_ranks = [i+1 for i, (state, _, _) in enumerate(self.three_level_results['micro_level']['leaders']) 
                      if state == state_code]
        
        return {
            'state_code': state_code,
            'state_name': state_name,
            'macro_rank': macro_ranks[0] if macro_ranks else None,
            'meso_rank': meso_ranks[0] if meso_ranks else None,
            'micro_rank': micro_ranks[0] if micro_ranks else None,
            'in_multi_level': any(leader['state'] == state_code 
                                for leader in self.three_level_results['multi_level_leaders'])
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export all centrality results to a pandas DataFrame for further analysis.
        
        Returns:
            pd.DataFrame: Comprehensive centrality results with state names
        """
        if not self.centrality_metrics:
            raise ValueError("Must calculate centrality metrics first")
        
        # Create DataFrame with all states
        states = list(self.network.nodes())
        results_df = pd.DataFrame({
            'state_code': states,
            'state_name': [STATE_FIPS.get(state, f"State-{state}") for state in states]
        })
        
        # Add all centrality metrics
        for metric, scores in self.centrality_metrics.items():
            results_df[metric] = results_df['state_code'].map(scores).fillna(0)
        
        # Add three-level rankings if available
        if self.three_level_results:
            # Initialize ranking columns
            results_df['macro_rank'] = None
            results_df['meso_rank'] = None
            results_df['micro_rank'] = None
            
            # Fill rankings
            for i, (state, _, _) in enumerate(self.three_level_results['macro_level']['leaders']):
                results_df.loc[results_df['state_code'] == state, 'macro_rank'] = i + 1
            
            for i, (state, _, _) in enumerate(self.three_level_results['meso_level']['leaders']):
                results_df.loc[results_df['state_code'] == state, 'meso_rank'] = i + 1
            
            for i, (state, _, _) in enumerate(self.three_level_results['micro_level']['leaders']):
                results_df.loc[results_df['state_code'] == state, 'micro_rank'] = i + 1
        
        # Sort by total degree (overall network importance)
        results_df = results_df.sort_values('total_degree', ascending=False).reset_index(drop=True)
        
        return results_df
    
    def save_results(self, output_dir: str, formats: List[str] = ['json', 'csv']) -> None:
        """
        Save analysis results in multiple formats.
        
        Args:
            output_dir (str): Directory to save results
            formats (list): List of formats to export ('json', 'csv', 'xlsx')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        
        if 'json' in formats:
            # Save complete results as JSON
            json_path = output_path / f"centrality_analysis_{timestamp}.json"
            results_dict = {
                'metadata': {
                    'analysis_date': pd.Timestamp.now().isoformat(),
                    'data_source': str(self.data_path),
                    'network_summary': self.get_network_summary()
                },
                'centrality_metrics': self.centrality_metrics,
                'three_level_analysis': self.three_level_results
            }
            
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            print(f"Results saved to: {json_path}")
        
        if 'csv' in formats:
            # Save DataFrame results as CSV
            csv_path = output_path / f"centrality_results_{timestamp}.csv"
            results_df = self.export_to_dataframe()
            results_df.to_csv(csv_path, index=False)
            print(f"Results saved to: {csv_path}")
        
        if 'xlsx' in formats:
            # Save as Excel with multiple sheets
            xlsx_path = output_path / f"centrality_analysis_{timestamp}.xlsx"
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                self.export_to_dataframe().to_excel(writer, sheet_name='All_Results', index=False)
                
                # Create separate sheets for each level
                if self.three_level_results:
                    for level in ['macro_level', 'meso_level', 'micro_level']:
                        level_df = pd.DataFrame([
                            {'Rank': i+1, 'State_Code': state, 'State_Name': name, 'Score': score}
                            for i, (state, name, score) in enumerate(self.three_level_results[level]['leaders'])
                        ])
                        level_df.to_excel(writer, sheet_name=level.replace('_', ' ').title(), index=False)
            
            print(f"Results saved to: {xlsx_path}")


class CentralityVisualizer:
    """
    Visualization tools for interstate centrality analysis results.
    
    Provides methods to create publication-quality charts and network diagrams
    for three-level centrality analysis. Designed to work in multiple environments
    including Jupyter notebooks, Colab, and standalone scripts.
    """
    
    def __init__(self, analyzer: CentralityAnalyzer):
        """
        Initialize visualizer with analysis results.
        
        Args:
            analyzer (CentralityAnalyzer): Completed centrality analysis
        """
        self.analyzer = analyzer
        self.results = analyzer.three_level_results
        self.metrics = analyzer.centrality_metrics
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_centrality_rankings(self, level: str = 'all', top_n: int = 15, 
                                save_path: Optional[str] = None) -> None:
        """
        Create bar charts showing centrality rankings.
        
        Args:
            level (str): Which level to plot ('macro', 'meso', 'micro', or 'all')
            top_n (int): Number of top states to display
            save_path (str, optional): Path to save the plot
        """
        if not self.results:
            raise ValueError("No analysis results available for visualization")
        
        if level == 'all':
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            levels = ['macro_level', 'meso_level', 'micro_level']
            titles = ['Macro: Regional Bridgeness', 'Meso: Influence Networks', 'Micro: Distribution Power']
            
            for i, (level_key, title) in enumerate(zip(levels, titles)):
                leaders = self.results[level_key]['leaders'][:top_n]
                states = [name for _, name, _ in leaders]
                scores = [score for _, _, score in leaders]
                
                bars = axes[i].barh(range(len(states)), scores)
                axes[i].set_yticks(range(len(states)))
                axes[i].set_yticklabels(states)
                axes[i].set_xlabel('Centrality Score')
                axes[i].set_title(title)
                axes[i].invert_yaxis()
                
                # Color bars by rank
                for j, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(j / len(bars)))
            
            plt.tight_layout()
            
        else:
            # Single level plot
            level_key = f"{level}_level"
            if level_key not in self.results:
                raise ValueError(f"Invalid level: {level}")
            
            leaders = self.results[level_key]['leaders'][:top_n]
            states = [name for _, name, _ in leaders]
            scores = [score for _, _, score in leaders]
            
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(states)), scores)
            plt.yticks(range(len(states)), states)
            plt.xlabel('Centrality Score')
            plt.title(f"{level.title()} Level: {self.results[level_key]['description']}")
            plt.gca().invert_yaxis()
            
            # Color bars by rank
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(i / len(bars)))
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_network_diagram(self, highlight_top: int = 5, layout: str = 'spring',
                           save_path: Optional[str] = None) -> None:
        """
        Create network diagram with nodes sized by centrality.
        
        Args:
            highlight_top (int): Number of top states to highlight
            layout (str): Network layout algorithm ('spring', 'circular', 'shell')
            save_path (str, optional): Path to save the plot
        """
        if self.analyzer.network is None:
            raise ValueError("Network not built yet")
        
        plt.figure(figsize=(12, 10))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.analyzer.network, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.analyzer.network)
        elif layout == 'shell':
            pos = nx.shell_layout(self.analyzer.network)
        
        # Node sizes based on total degree
        node_sizes = [self.metrics['total_degree'].get(node, 0) / 1e10 for node in self.analyzer.network.nodes()]
        
        # Highlight top performers
        if self.results and 'multi_level_leaders' in self.results:
            top_states = [leader['state'] for leader in self.results['multi_level_leaders'][:highlight_top]]
            node_colors = ['red' if node in top_states else 'lightblue' 
                          for node in self.analyzer.network.nodes()]
        else:
            node_colors = 'lightblue'
        
        # Draw network
        nx.draw_networkx_nodes(self.analyzer.network, pos, 
                              node_size=node_sizes, 
                              node_color=node_colors, 
                              alpha=0.7)
        
        nx.draw_networkx_edges(self.analyzer.network, pos, 
                              alpha=0.3, 
                              width=0.5,
                              edge_color='gray')
        
        # Add labels for highlighted nodes
        if self.results:
            top_labels = {node: STATE_FIPS.get(node, str(node)) 
                         for node in top_states if node in self.analyzer.network.nodes()}
            nx.draw_networkx_labels(self.analyzer.network, pos, 
                                  labels=top_labels, 
                                  font_size=8, 
                                  font_weight='bold')
        
        plt.title(f"U.S. Interstate Trade Network\n(Top {highlight_top} Multi-Level Leaders Highlighted)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network diagram saved to: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Create correlation matrix heatmap for centrality measures.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.metrics:
            raise ValueError("No centrality metrics available")
        
        # Create DataFrame with centrality metrics
        states = list(self.analyzer.network.nodes())
        centrality_df = pd.DataFrame({
            'Betweenness': [self.metrics['betweenness'].get(s, 0) for s in states],
            'Eigenvector': [self.metrics['eigenvector'].get(s, 0) for s in states],
            'Out-Degree': [self.metrics['out_degree_normalized'].get(s, 0) for s in states],
            'PageRank': [self.metrics['pagerank'].get(s, 0) for s in states],
            'Closeness': [self.metrics['closeness'].get(s, 0) for s in states]
        })
        
        # Calculate correlation matrix
        corr_matrix = centrality_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix: Centrality Measures')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        
        plt.show()
    
    def save_all_plots(self, output_dir: str) -> None:
        """
        Generate and save all standard visualization plots.
        
        Args:
            output_dir (str): Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        
        print("Generating all visualization plots...")
        
        # Three-level rankings
        self.plot_centrality_rankings('all', save_path=output_path / f"three_level_rankings_{timestamp}.png")
        
        # Network diagram
        self.plot_network_diagram(save_path=output_path / f"network_diagram_{timestamp}.png")
        
        # Correlation matrix
        self.plot_correlation_matrix(save_path=output_path / f"correlation_matrix_{timestamp}.png")
        
        print(f"All plots saved to: {output_path}")


def main():
    """
    Command line interface for interstate centrality analysis.
    """
    parser = argparse.ArgumentParser(
        description="Interstate Commerce Network Centrality Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python centrality_analysis.py --data cfs_2017_puf.csv
    python centrality_analysis.py --data cfs_2017_puf.csv --output results/ --visualize
    python centrality_analysis.py --data cfs_2017_puf.csv --sample 10000 --top-n 15
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to CFS CSV data file')
    parser.add_argument('--output', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--sample', type=int, help='Sample size for testing (use full dataset if not specified)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top states to analyze (default: 10)')
    parser.add_argument('--threshold', type=float, default=0, help='Minimum flow value threshold (default: 0)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--formats', nargs='+', default=['json', 'csv'], 
                       choices=['json', 'csv', 'xlsx'], help='Output formats (default: json csv)')
    
    args = parser.parse_args()
    
    try:
        print("=== Interstate Commerce Network Centrality Analysis ===\n")
        
        # Initialize analyzer
        analyzer = CentralityAnalyzer(args.data)
        
        # Load and process data
        analyzer.load_and_process_data(sample_size=args.sample)
        
        # Build network
        analyzer.build_network(min_value_threshold=args.threshold)
        
        # Calculate centrality metrics
        analyzer.calculate_centrality_metrics()
        
        # Run three-level analysis
        results = analyzer.run_three_level_analysis(top_n=args.top_n)
        
        # Display summary results
        print(f"\n=== THREE-LEVEL ANALYSIS RESULTS ===")
        
        print(f"\nMACRO LEVEL - {results['macro_level']['description']}")
        for i, (state, name, score) in enumerate(results['macro_level']['leaders'][:5], 1):
            print(f"  {i:2d}. {name} (State {state}): {score:.6f}")
        
        print(f"\nMESO LEVEL - {results['meso_level']['description']}")
        for i, (state, name, score) in enumerate(results['meso_level']['leaders'][:5], 1):
            print(f"  {i:2d}. {name} (State {state}): {score:.6f}")
        
        print(f"\nMICRO LEVEL - {results['micro_level']['description']}")
        for i, (state, name, score) in enumerate(results['micro_level']['leaders'][:5], 1):
            print(f"  {i:2d}. {name} (State {state}): {score:.6f}")
        
        if results['multi_level_leaders']:
            print(f"\nMULTI-LEVEL LEADERS (Top 5):")
            for leader in results['multi_level_leaders'][:5]:
                levels_str = ", ".join(leader['levels'])
                print(f"  {leader['state_name']} (State {leader['state']}): "
                      f"{leader['level_count']} levels ({levels_str}) - Score: {leader['total_score']:.4f}")
        
        # Save results
        analyzer.save_results(args.output, formats=args.formats)
        
        # Generate visualizations if requested
        if args.visualize:
            print("\nGenerating visualizations...")
            viz = CentralityVisualizer(analyzer)
            viz.save_all_plots(args.output)
        
        print(f"\nAnalysis complete! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()