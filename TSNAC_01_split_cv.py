#!/usr/bin/env python
"""
TSNAC_01_split_cv.py - Create train/validation/test splits for the dataset

This script reads a list of case IDs and splits them into training, validation, and test sets.
The split is saved as a JSON file for later use in the pipeline.

Usage:
    python TSNAC_01_split_cv.py [--output_dir OUTPUT_DIR] [--output_file OUTPUT_FILE]

Example:
    python TSNAC_01_split_cv.py --output_dir ./my_dataset --output_file split.json
"""

import os
import json
import argparse

# List of all case IDs - CUSTOMIZE THIS WITH YOUR OWN CASE IDs
# Format: Each entry should be a unique identifier for a case (e.g., "PATIENT001", "case_0123")
case_name_list = sorted([
    # Example case IDs - Replace with your actual case IDs
    "PATIENT001", "PATIENT002", "PATIENT003", "PATIENT004",
    "PATIENT005", "PATIENT006", "PATIENT007", "PATIENT008", 
    "PATIENT009", "PATIENT010", "PATIENT011", "PATIENT012",
    "PATIENT013", "PATIENT014", "PATIENT015", "PATIENT016",
    "PATIENT017", "PATIENT018", "PATIENT019", "PATIENT020",
    # Additional cases from different centers could use a different prefix
    "CENTER2_001", "CENTER2_002", "CENTER2_003", "CENTER2_004",
    "CENTER2_005", "CENTER2_006", "CENTER2_007", "CENTER2_008",
    "CENTER2_009", "CENTER2_010", "CENTER2_011", "CENTER2_012",
    "CENTER3_001", "CENTER3_002", "CENTER3_003", "CENTER3_004",
    "CENTER3_005", "CENTER3_006", "CENTER3_007", "CENTER3_008"
])

# List of test case IDs - CUSTOMIZE THIS WITH YOUR OWN TEST CASES
# These cases will be used for final evaluation and not for training or validation
test_cases = [
    # Example test cases - Replace with your actual test cases
    "PATIENT001", "PATIENT002", "PATIENT003", "PATIENT004",
    "PATIENT005", "PATIENT006", "PATIENT007", "PATIENT008", 
    "PATIENT009", "PATIENT010", "PATIENT011", "PATIENT012"
]

def create_splits(output_dir="./TS_NAC", output_file="dataset_split.json", train_ratio=0.75):
    """
    Create and save train/validation/test splits
    
    Args:
        output_dir (str): Directory to save the split file
        output_file (str): Name of the output JSON file
        train_ratio (float): Ratio of non-test data to use for training (default: 0.75)
    
    Returns:
        dict: The split dictionary containing train, validation and test lists
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize split dictionary
    split_dict = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # First separate test cases
    for case in case_name_list:
        if case in test_cases:
            split_dict["test"].append(case)
    
    # Get remaining cases for train/val split
    remaining_cases = [case for case in case_name_list if case not in test_cases]
    remaining_cases.sort()  # Sort for consistency
    
    # Calculate split point based on ratio
    train_size = int(len(remaining_cases) * train_ratio)
    
    # Split remaining cases into train and val
    split_dict["train"] = remaining_cases[:train_size]
    split_dict["val"] = remaining_cases[train_size:]
    
    # Sort all lists for consistency
    split_dict["train"].sort()
    split_dict["val"].sort()
    split_dict["test"].sort()
    
    # Save to JSON file
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
    
    print(f"Split created successfully:")
    print(f"  - Training cases: {len(split_dict['train'])}")
    print(f"  - Validation cases: {len(split_dict['val'])}")
    print(f"  - Test cases: {len(split_dict['test'])}")
    print(f"Split saved to: {output_path}")
    
    return split_dict

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create dataset splits for training")
    parser.add_argument("--output_dir", type=str, default="./TS_NAC", 
                        help="Directory to save the split file")
    parser.add_argument("--output_file", type=str, default="dataset_split.json", 
                        help="Name of the output JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.75, 
                        help="Ratio of non-test data to use for training")
    args = parser.parse_args()
    
    # Create and save splits
    create_splits(args.output_dir, args.output_file, args.train_ratio)

