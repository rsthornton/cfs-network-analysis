#!/usr/bin/env python3
"""
Interstate Commodity Flow Extraction Tool

This module provides flexible extraction and analysis of commodity flows from 
the U.S. Commodity Flow Survey (CFS) dataset. Supports geographic, commodity, 
industry, and transportation mode filtering for detailed trade flow analysis.

Usage:
    Terminal: python flow_extraction.py --states CA,TX,NY --commodities 35,36 --top-n 20
    Programmatic:
        extractor = FlowExtractor("cfs_2017_puf.csv")
        flows = extractor.extract_bilateral_flows(["CA", "TX"])

Author: Shingai Thornton
Institution: Binghamton University Systems Science
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Reference mappings from CFS User Guide
STATE_FIPS = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
    11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN',
    19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA',
    26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV',
    33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH',
    40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
    48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'
}

# Reverse mapping for flexible input
STATE_NAME_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}

SCTG_CODES = {
    1: 'Animals and Fish (live)', 2: 'Cereal Grains', 3: 'Agricultural Products', 
    4: 'Animal Feed and Products', 5: 'Meat, Poultry, Fish, Seafood',
    6: 'Milled Grain Products and Bakery', 7: 'Other Prepared Foodstuffs', 
    8: 'Alcoholic Beverages', 9: 'Tobacco Products',
    10: 'Monumental or Building Stone', 11: 'Natural Sands', 12: 'Gravel and Crushed Stone',
    13: 'Other Non-Metallic Minerals', 14: 'Metallic Ores and Concentrates',
    15: 'Coal', 16: 'Crude Petroleum', 17: 'Gasoline and Aviation Fuel', 
    18: 'Fuel Oils', 19: 'Other Coal and Petroleum Products',
    20: 'Basic Chemicals', 21: 'Pharmaceutical Products', 22: 'Fertilizers',
    23: 'Other Chemical Products', 24: 'Plastics and Rubber',
    25: 'Logs and Wood in the Rough', 26: 'Wood Products', 27: 'Pulp, Paper, Paperboard',
    28: 'Paper or Paperboard Articles', 29: 'Printed Products', 30: 'Textiles and Leather',
    31: 'Non-Metallic Mineral Products', 32: 'Base Metal in Primary Forms', 
    33: 'Articles of Base Metal', 34: 'Machinery',
    35: 'Electronic and Electrical Equipment', 36: 'Motorized and Other Vehicles',
    37: 'Transportation Equipment', 38: 'Precision Instruments',
    39: 'Furniture and Lighting', 40: 'Miscellaneous Manufactured Products',
    41: 'Waste and Scrap', 43: 'Mixed Freight'
}

COMMODITY_CATEGORIES = {
    'energy': [15, 16, 17, 18, 19],
    'technology': [34, 35, 38],
    'agriculture': [1, 2, 3, 4, 5, 6, 7],
    'manufacturing': [31, 32, 33, 34, 35, 36, 37],
    'chemicals': [20, 21, 22, 23, 24],
    'materials': [10, 11, 12, 13, 14, 25, 26, 27, 28]
}

MODE_CODES = {
    3: 'Truck', 4: 'For-hire truck', 5: 'Company-owned truck',
    6: 'Rail', 7: 'Water', 8: 'Inland Water', 9: 'Great Lakes', 10: 'Deep Sea',
    11: 'Air', 12: 'Pipeline', 13: 'Multiple mode', 14: 'Parcel/Courier',
    15: 'Truck and rail', 16: 'Truck and water', 17: 'Rail and water'
}


class FlowExtractor:
    """
    Extract and analyze commodity flows from CFS data.
    
    This class provides flexible methods to filter and analyze interstate trade flows
    by geography, commodity type, industry, transportation mode, and other dimensions.
    
    Attributes:
        data_path (Path): Path to the CFS dataset
        raw_data (pd.DataFrame): Original CFS data
        flows_data (pd.DataFrame): Processed flow data
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the flow extractor with CFS data.
        
        Args:
            data_path (str): Path to the CFS CSV file
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.flows_data = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        print(f"Initialized FlowExtractor with data: {self.data_path}")
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and preprocess CFS data for flow analysis.
        
        Args:
            sample_size (int, optional): Number of records to sample for testing
            
        Returns:
            pd.DataFrame: Processed flow data with proper survey weighting
        """
        print("Loading CFS data...")
        
        if sample_size:
            self.raw_data = pd.read_csv(self.data_path, nrows=sample_size)
            print(f"Loaded sample of {len(self.raw_data):,} records")
        else:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.raw_data):,} total records")
        
        # Verify required columns
        required_columns = ['ORIG_STATE', 'DEST_STATE', 'SCTG', 'SHIPMT_VALUE', 'WGT_FACTOR', 'SHIPMT_WGHT']
        missing_cols = [col for col in required_columns if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Apply survey weighting and create working dataset
        self.raw_data['weighted_value'] = self.raw_data['WGT_FACTOR'] * self.raw_data['SHIPMT_VALUE']
        self.raw_data['weighted_tons'] = self.raw_data['WGT_FACTOR'] * self.raw_data['SHIPMT_WGHT'] / 2000
        
        # Add human-readable names
        self.raw_data['orig_state_name'] = self.raw_data['ORIG_STATE'].map(STATE_FIPS)
        self.raw_data['dest_state_name'] = self.raw_data['DEST_STATE'].map(STATE_FIPS)
        self.raw_data['commodity_name'] = self.raw_data['SCTG'].map(SCTG_CODES).fillna('SCTG-' + self.raw_data['SCTG'].astype(str))
        
        print("Data preprocessing completed")
        return self.raw_data
    
    def extract_bilateral_flows(self, state_list: List[Union[str, int]], 
                               commodity_filter: Optional[List[int]] = None,
                               top_n: int = 20,
                               include_intrastate: bool = False) -> pd.DataFrame:
        """
        Extract flows between specified states.
        
        Args:
            state_list: State codes (["CA", "TX"] or [6, 48]) 
            commodity_filter: SCTG codes to include (e.g., [35, 36] for electronics)
            top_n: Number of top flows to return
            include_intrastate: Whether to include flows within same state
            
        Returns:
            pd.DataFrame: Filtered and aggregated flow data
            
        Examples:
            # Energy flows between Texas, Oklahoma, Louisiana
            flows = extractor.extract_bilateral_flows(
                ["TX", "OK", "LA"], 
                commodity_filter=[15, 16, 17, 18, 19]
            )
            
            # All flows between major economic centers
            flows = extractor.extract_bilateral_flows(["CA", "TX", "NY", "FL"])
        """
        if self.raw_data is None:
            raise ValueError("Must load data first using load_data()")
        
        # Convert state names to FIPS codes
        state_codes = []
        for state in state_list:
            if isinstance(state, str):
                if state in STATE_NAME_TO_FIPS:
                    state_codes.append(STATE_NAME_TO_FIPS[state])
                else:
                    raise ValueError(f"Unknown state: {state}")
            else:
                state_codes.append(state)
        
        print(f"Extracting flows between states: {[STATE_FIPS.get(s, s) for s in state_codes]}")
        
        # Filter for flows between target states
        flows = self.raw_data[
            (self.raw_data['ORIG_STATE'].isin(state_codes)) & 
            (self.raw_data['DEST_STATE'].isin(state_codes))
        ].copy()
        
        if not include_intrastate:
            flows = flows[flows['ORIG_STATE'] != flows['DEST_STATE']]
        
        # Apply commodity filter if specified
        if commodity_filter:
            flows = flows[flows['SCTG'].isin(commodity_filter)]
            print(f"Filtered to commodities: {[SCTG_CODES.get(c, c) for c in commodity_filter]}")
        
        if len(flows) == 0:
            print("❌ No flows found matching criteria")
            print("💡 Suggestions:")
            print("   • Try a larger sample size (--sample 50000)")
            print("   • Include more states in your analysis")
            print("   • Check available commodities with --show-commodities")
            print("   • View examples with --examples")
            return pd.DataFrame()
        
        # Aggregate flows by route and commodity
        flow_summary = flows.groupby([
            'ORIG_STATE', 'DEST_STATE', 'orig_state_name', 'dest_state_name',
            'SCTG', 'commodity_name'
        ]).agg({
            'weighted_value': 'sum',
            'weighted_tons': 'sum',
            'WGT_FACTOR': 'sum'
        }).reset_index()
        
        # Calculate value density
        flow_summary['value_density'] = flow_summary['weighted_value'] / flow_summary['weighted_tons']
        flow_summary['value_density'] = flow_summary['value_density'].replace([np.inf, -np.inf], 0)
        
        # Create flow description
        flow_summary['flow_route'] = (
            flow_summary['orig_state_name'] + ' → ' + flow_summary['dest_state_name']
        )
        
        # Sort by value and return top N
        flow_summary = flow_summary.sort_values('weighted_value', ascending=False)
        
        print(f"Found {len(flow_summary):,} unique flow relationships")
        print(f"Total value: ${flow_summary['weighted_value'].sum()/1e9:.1f}B")
        
        self.flows_data = flow_summary.head(top_n)
        return self.flows_data
    
    def extract_flows_by_commodity(self, commodity_codes: List[int], 
                                  origin_states: Optional[List[Union[str, int]]] = None,
                                  dest_states: Optional[List[Union[str, int]]] = None,
                                  top_n: int = 20) -> pd.DataFrame:
        """
        Extract flows for specific commodities with optional geographic filtering.
        
        Args:
            commodity_codes: SCTG codes (e.g., [35, 36] for electronics)
            origin_states: Limit to flows from these states
            dest_states: Limit to flows to these states  
            top_n: Number of top flows to return
            
        Returns:
            pd.DataFrame: Commodity-focused flow analysis
            
        Examples:
            # All energy flows nationwide
            flows = extractor.extract_flows_by_commodity([15, 16, 17, 18, 19])
            
            # California tech exports
            flows = extractor.extract_flows_by_commodity([35, 38], origin_states=["CA"])
        """
        if self.raw_data is None:
            raise ValueError("Must load data first using load_data()")
        
        # Filter by commodity
        flows = self.raw_data[self.raw_data['SCTG'].isin(commodity_codes)].copy()
        
        # Apply geographic filters
        if origin_states:
            orig_codes = self._convert_state_names(origin_states)
            flows = flows[flows['ORIG_STATE'].isin(orig_codes)]
        
        if dest_states:
            dest_codes = self._convert_state_names(dest_states)
            flows = flows[flows['DEST_STATE'].isin(dest_codes)]
        
        if len(flows) == 0:
            print("❌ No flows found matching criteria")
            print("💡 Suggestions:")
            print("   • Try broader geographic filters")
            print("   • Use a larger sample size")
            print("   • Check commodity availability with --show-commodities")
            return pd.DataFrame()
        
        # Aggregate and analyze
        commodity_summary = flows.groupby([
            'ORIG_STATE', 'DEST_STATE', 'orig_state_name', 'dest_state_name',
            'SCTG', 'commodity_name'
        ]).agg({
            'weighted_value': 'sum',
            'weighted_tons': 'sum'
        }).reset_index()
        
        commodity_summary['flow_route'] = (
            commodity_summary['orig_state_name'] + ' → ' + commodity_summary['dest_state_name']
        )
        
        commodity_summary = commodity_summary.sort_values('weighted_value', ascending=False)
        
        print(f"Commodity analysis: {[SCTG_CODES.get(c, c) for c in commodity_codes]}")
        print(f"Found {len(commodity_summary):,} flow relationships")
        
        return commodity_summary.head(top_n)
    
    def extract_state_profile(self, state: Union[str, int], 
                             analysis_type: str = 'both') -> Dict:
        """
        Generate comprehensive trade profile for a single state.
        
        Args:
            state: State code or name (e.g., "CA" or 6)
            analysis_type: 'exports', 'imports', or 'both'
            
        Returns:
            dict: Complete trade profile including top partners and commodities
        """
        if self.raw_data is None:
            raise ValueError("Must load data first using load_data()")
        
        state_code = self._convert_state_names([state])[0]
        state_name = STATE_FIPS.get(state_code, f"State-{state_code}")
        
        profile = {
            'state_code': state_code,
            'state_name': state_name,
            'analysis_type': analysis_type
        }
        
        if analysis_type in ['exports', 'both']:
            exports = self.raw_data[self.raw_data['ORIG_STATE'] == state_code]
            export_summary = exports.groupby(['DEST_STATE', 'dest_state_name']).agg({
                'weighted_value': 'sum'
            }).reset_index().sort_values('weighted_value', ascending=False)
            
            profile['top_export_partners'] = export_summary.head(10).to_dict('records')
            profile['total_exports'] = exports['weighted_value'].sum()
        
        if analysis_type in ['imports', 'both']:
            imports = self.raw_data[self.raw_data['DEST_STATE'] == state_code]
            import_summary = imports.groupby(['ORIG_STATE', 'orig_state_name']).agg({
                'weighted_value': 'sum'
            }).reset_index().sort_values('weighted_value', ascending=False)
            
            profile['top_import_partners'] = import_summary.head(10).to_dict('records')
            profile['total_imports'] = imports['weighted_value'].sum()
        
        return profile
    
    def show_available_commodities(self, category: Optional[str] = None) -> None:
        """
        Display available commodity codes and descriptions.
        
        Args:
            category: Optional category filter ('energy', 'technology', etc.)
        """
        print("AVAILABLE COMMODITIES")
        print("=" * 50)
        
        if category and category in COMMODITY_CATEGORIES:
            relevant_codes = COMMODITY_CATEGORIES[category]
            print(f"Category: {category.title()}")
            for code in relevant_codes:
                if code in SCTG_CODES:
                    print(f"  {code:2d}: {SCTG_CODES[code]}")
        else:
            if category:
                print(f"Available categories: {', '.join(COMMODITY_CATEGORIES.keys())}")
                print()
            
            for code, name in SCTG_CODES.items():
                print(f"  {code:2d}: {name}")
    
    def show_available_states(self) -> None:
        """Display available state codes and names."""
        print("AVAILABLE STATES")
        print("=" * 50)
        for code, name in sorted(STATE_FIPS.items()):
            print(f"  {code:2d}: {name}")
    
    def show_example_flows(self) -> None:
        """Display common flow extraction examples."""
        examples = [
            ("Energy corridor", "extract_bilateral_flows(['TX', 'OK', 'LA'], commodity_filter=[15,16,17,18,19])"),
            ("Tech flows", "extract_bilateral_flows(['CA', 'WA'], commodity_filter=[35,38])"),
            ("Manufacturing belt", "extract_bilateral_flows(['OH', 'MI', 'PA'], commodity_filter=[31,32,33,34])"),
            ("California exports", "extract_flows_by_commodity([35,36,38], origin_states=['CA'])"),
            ("All energy flows", "extract_flows_by_commodity([15,16,17,18,19])"),
            ("State profile", "extract_state_profile('TX')")
        ]
        
        print("EXAMPLE FLOW EXTRACTIONS")
        print("=" * 50)
        for desc, code in examples:
            print(f"{desc}:")
            print(f"  extractor.{code}")
            print()
    
    def _convert_state_names(self, states: List[Union[str, int]]) -> List[int]:
        """Convert state names/codes to FIPS codes."""
        state_codes = []
        for state in states:
            if isinstance(state, str):
                if state in STATE_NAME_TO_FIPS:
                    state_codes.append(STATE_NAME_TO_FIPS[state])
                else:
                    raise ValueError(f"Unknown state: {state}")
            else:
                state_codes.append(state)
        return state_codes
    
    def export_flows(self, flows_df: pd.DataFrame, output_path: str) -> None:
        """
        Export flow analysis results to CSV.
        
        Args:
            flows_df: DataFrame with flow analysis results
            output_path: Path for output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        flows_df.to_csv(output_file, index=False)
        print(f"Flow analysis exported to: {output_file}")


class FlowVisualizer:
    """
    Create visualizations for commodity flow analysis.
    
    Provides methods to generate charts, maps, and diagrams for flow data
    that work well in both standalone scripts and Jupyter notebooks.
    """
    
    def __init__(self, flows_data: pd.DataFrame):
        """
        Initialize visualizer with flow data.
        
        Args:
            flows_data: DataFrame from FlowExtractor methods
        """
        self.flows_data = flows_data
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_top_flows(self, top_n: int = 15, save_path: Optional[str] = None) -> None:
        """
        Create horizontal bar chart of top flows by value.
        
        Args:
            top_n: Number of top flows to display
            save_path: Optional path to save the plot
        """
        if len(self.flows_data) == 0:
            print("No flow data available for visualization")
            return
        
        top_flows = self.flows_data.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Create labels combining route and commodity
        labels = [f"{row['flow_route']}\n{row['commodity_name'][:30]}" 
                 for _, row in top_flows.iterrows()]
        
        values = top_flows['weighted_value'] / 1e9  # Convert to billions
        
        bars = plt.barh(range(len(labels)), values)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Flow Value (Billions $)')
        plt.title(f'Top {top_n} Interstate Commodity Flows by Value')
        plt.gca().invert_yaxis()
        
        # Color bars by value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_flow_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create summary dashboard with multiple flow perspectives.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.flows_data) == 0:
            print("No flow data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top flows by value
        top_10 = self.flows_data.head(10)
        axes[0, 0].barh(range(len(top_10)), top_10['weighted_value'] / 1e9)
        axes[0, 0].set_yticks(range(len(top_10)))
        axes[0, 0].set_yticklabels([f"{row['flow_route']}" for _, row in top_10.iterrows()])
        axes[0, 0].set_xlabel('Value (Billions $)')
        axes[0, 0].set_title('Top 10 Flows by Value')
        axes[0, 0].invert_yaxis()
        
        # Commodity breakdown
        commodity_totals = self.flows_data.groupby('commodity_name')['weighted_value'].sum().sort_values(ascending=False).head(8)
        axes[0, 1].pie(commodity_totals.values, labels=commodity_totals.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Flow Value by Commodity')
        
        # Origin state analysis
        origin_totals = self.flows_data.groupby('orig_state_name')['weighted_value'].sum().sort_values(ascending=False).head(10)
        axes[1, 0].bar(range(len(origin_totals)), origin_totals.values / 1e9)
        axes[1, 0].set_xticks(range(len(origin_totals)))
        axes[1, 0].set_xticklabels(origin_totals.index, rotation=45)
        axes[1, 0].set_ylabel('Value (Billions $)')
        axes[1, 0].set_title('Top Origin States')
        
        # Value vs tonnage scatter
        axes[1, 1].scatter(self.flows_data['weighted_tons'] / 1e6, 
                          self.flows_data['weighted_value'] / 1e9, 
                          alpha=0.6)
        axes[1, 1].set_xlabel('Weight (Million Tons)')
        axes[1, 1].set_ylabel('Value (Billions $)')
        axes[1, 1].set_title('Value vs Weight Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary dashboard saved to: {save_path}")
        
        plt.show()


def main():
    """Command line interface for flow extraction."""
    parser = argparse.ArgumentParser(
        description="Interstate Commodity Flow Extraction and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python flow_extraction.py --states CA,TX,NY --top-n 25
    python flow_extraction.py --commodities 35,36,38 --origins CA,WA --top-n 15
    python flow_extraction.py --show-commodities energy
    python flow_extraction.py --show-states
    python flow_extraction.py --examples
        """
    )
    
    parser.add_argument('--data', default='cfs_2017_puf.csv', help='Path to CFS CSV data file')
    parser.add_argument('--states', help='Comma-separated state codes for bilateral analysis (e.g., CA,TX,NY)')
    parser.add_argument('--commodities', help='Comma-separated SCTG codes (e.g., 35,36,38)')
    parser.add_argument('--origins', help='Origin states filter (e.g., CA,TX)')
    parser.add_argument('--destinations', help='Destination states filter (e.g., NY,FL)')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top flows to display')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--show-commodities', help='Show available commodities (optionally by category)')
    parser.add_argument('--show-states', action='store_true', help='Show available state codes')
    parser.add_argument('--examples', action='store_true', help='Show usage examples')
    
    args = parser.parse_args()
    
    # Handle informational flags that don't need data files
    if args.show_states:
        print("AVAILABLE STATES")
        print("=" * 50)
        for code, name in sorted(STATE_FIPS.items()):
            print(f"  {code:2d}: {name}")
        return
    
    if args.show_commodities is not None:
        print("AVAILABLE COMMODITIES")
        print("=" * 50)
        category = args.show_commodities if args.show_commodities else None
        if category and category in COMMODITY_CATEGORIES:
            relevant_codes = COMMODITY_CATEGORIES[category]
            print(f"Category: {category.title()}")
            for code in relevant_codes:
                if code in SCTG_CODES:
                    print(f"  {code:2d}: {SCTG_CODES[code]}")
        else:
            if category:
                print(f"Available categories: {', '.join(COMMODITY_CATEGORIES.keys())}")
                print()
            for code, name in SCTG_CODES.items():
                print(f"  {code:2d}: {name}")
        return
    
    if args.examples:
        examples = [
            ("Energy corridor", "extract_bilateral_flows(['TX', 'OK', 'LA'], commodity_filter=[15,16,17,18,19])"),
            ("Tech flows", "extract_bilateral_flows(['CA', 'WA'], commodity_filter=[35,38])"),
            ("Manufacturing belt", "extract_bilateral_flows(['OH', 'MI', 'PA'], commodity_filter=[31,32,33,34])"),
            ("California exports", "extract_flows_by_commodity([35,36,38], origin_states=['CA'])"),
            ("All energy flows", "extract_flows_by_commodity([15,16,17,18,19])"),
            ("State profile", "extract_state_profile('TX')")
        ]
        print("EXAMPLE FLOW EXTRACTIONS")
        print("=" * 50)
        for desc, code in examples:
            print(f"{desc}:")
            print(f"  extractor.{code}")
            print()
        return
    
    try:
        print("=== Interstate Commodity Flow Extraction ===\n")
        
        # Initialize extractor
        extractor = FlowExtractor(args.data)
        extractor.load_data(sample_size=args.sample)
        
        # Determine analysis type and extract flows
        if args.states:
            state_list = args.states.split(',')
            commodity_filter = None
            if args.commodities:
                commodity_filter = [int(c.strip()) for c in args.commodities.split(',')]
            
            flows = extractor.extract_bilateral_flows(
                state_list, commodity_filter=commodity_filter, top_n=args.top_n
            )
            analysis_type = "bilateral_flows"
            
        elif args.commodities:
            commodity_codes = [int(c.strip()) for c in args.commodities.split(',')]
            origin_states = None
            dest_states = None
            
            if args.origins:
                origin_states = args.origins.split(',')
            if args.destinations:
                dest_states = args.destinations.split(',')
            
            flows = extractor.extract_flows_by_commodity(
                commodity_codes, origin_states=origin_states, 
                dest_states=dest_states, top_n=args.top_n
            )
            analysis_type = "commodity_flows"
            
        else:
            print("Error: Must specify either --states or --commodities")
            parser.print_help()
            return
        
        if len(flows) == 0:
            print("❌ No flows found matching criteria")
            print("💡 Try different states, larger sample, or --examples for guidance")
            return
        
        # Display results
        print(f"\n=== TOP {args.top_n} FLOWS ===")
        for i, (_, row) in enumerate(flows.head(args.top_n).iterrows(), 1):
            value_billions = row['weighted_value'] / 1e9
            tons_millions = row['weighted_tons'] / 1e6
            print(f"{i:2d}. {row['flow_route']}")
            print(f"    {row['commodity_name']}")
            print(f"    Value: ${value_billions:.2f}B | Weight: {tons_millions:.1f}M tons")
            print()
        
        # Export results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        output_file = output_dir / f"{analysis_type}_{timestamp}.csv"
        extractor.export_flows(flows, output_file)
        
        # Generate visualizations if requested
        if args.visualize:
            print("Generating visualizations...")
            viz = FlowVisualizer(flows)
            
            plot_file = output_dir / f"{analysis_type}_chart_{timestamp}.png"
            viz.plot_top_flows(args.top_n, save_path=plot_file)
            
            summary_file = output_dir / f"{analysis_type}_summary_{timestamp}.png"
            viz.plot_flow_summary(save_path=summary_file)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()