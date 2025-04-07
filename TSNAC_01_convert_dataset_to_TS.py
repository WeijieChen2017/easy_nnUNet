#!/usr/bin/env python
"""
TSNAC_01_convert_dataset_to_TS.py - Organize dataset files into TotalSegmentator format

This script takes a directory containing medical image files and organizes them into a
directory structure compatible with TotalSegmentator and the nnUNet pipeline.

For each case, it creates a directory and copies/renames the input files to:
- ct.nii.gz: The input CT scan (from NAC_{case}_256.nii.gz by default)
- label.nii.gz: The segmentation labels (from CTAC_{case}_TS.nii.gz by default)
- gt.nii.gz: The ground truth attenuation corrected image (from CTAC_{case}_cropped.nii.gz by default)

Usage:
    python TSNAC_01_convert_dataset_to_TS.py [--source SOURCE_DIR] [--output OUTPUT_DIR]
                                           [--input_pattern INPUT_PATTERN]
                                           [--label_pattern LABEL_PATTERN]
                                           [--gt_pattern GT_PATTERN]

Example:
    python TSNAC_01_convert_dataset_to_TS.py --source ./raw_data --output ./TS_NAC --input_pattern "NAC_{}.nii.gz"
"""

import os
import shutil
import argparse
from pathlib import Path

# Default list of case IDs - CUSTOMIZE THIS WITH YOUR OWN CASE IDs
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

def organize_files(source_dir, output_base_dir, input_pattern="NAC_{}_256.nii.gz", 
                 label_pattern="CTAC_{}_TS.nii.gz", gt_pattern="CTAC_{}_cropped.nii.gz",
                 cases=None):
    """
    Organize dataset files into TotalSegmentator format
    
    Args:
        source_dir (str): Directory containing the source files
        output_base_dir (str): Base directory where organized files will be stored
        input_pattern (str): Pattern for input image filenames with {} as placeholder for case ID
        label_pattern (str): Pattern for label image filenames with {} as placeholder for case ID
        gt_pattern (str): Pattern for ground truth image filenames with {} as placeholder for case ID
        cases (list): List of case IDs to process. If None, uses default case_name_list
    """
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Use provided cases or default list
    if cases is None:
        cases = case_name_list
    
    print(f"Organizing files for {len(cases)} cases")
    print(f"  - Source directory: {source_dir}")
    print(f"  - Output directory: {output_base_dir}")
    
    processed_cases = 0
    missing_files = 0
    
    for case_name in cases:
        # Create case directory
        case_dir = os.path.join(output_base_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Define source and destination file paths
        file_mappings = {
            label_pattern.format(case_name): "label.nii.gz",
            input_pattern.format(case_name): "ct.nii.gz",
            gt_pattern.format(case_name): "gt.nii.gz"
        }
        
        case_missing_files = 0
        
        # Copy and rename files
        for src_name, dst_name in file_mappings.items():
            src_path = os.path.join(source_dir, src_name)
            dst_path = os.path.join(case_dir, dst_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"  Copied {src_path} to {dst_path}")
            else:
                print(f"  Warning: Source file not found: {src_path}")
                case_missing_files += 1
                missing_files += 1
        
        if case_missing_files == 0:
            processed_cases += 1
    
    print("\nOrganization complete:")
    print(f"  - Successfully processed {processed_cases}/{len(cases)} cases")
    if missing_files > 0:
        print(f"  - {missing_files} files were missing")
    print(f"  - Organized files are in: {output_base_dir}")

def load_cases_from_json(json_path):
    """Load case IDs from a JSON file (e.g., from TSNAC_01_split_cv.py)"""
    import json
    
    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Combine all splits
        all_cases = []
        for split in ["train", "val", "test"]:
            if split in data:
                all_cases.extend(data[split])
        
        return sorted(all_cases)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Organize dataset files into TotalSegmentator format")
    parser.add_argument("--source", type=str, default="TS_NAC",
                        help="Directory containing the source files")
    parser.add_argument("--output", type=str, default="TS_NAC",
                        help="Base directory where organized files will be stored")
    parser.add_argument("--input_pattern", type=str, default="NAC_{}_256.nii.gz",
                        help="Pattern for input image filenames with {} as placeholder for case ID")
    parser.add_argument("--label_pattern", type=str, default="CTAC_{}_TS.nii.gz",
                        help="Pattern for label image filenames with {} as placeholder for case ID")
    parser.add_argument("--gt_pattern", type=str, default="CTAC_{}_cropped.nii.gz",
                        help="Pattern for ground truth image filenames with {} as placeholder for case ID")
    parser.add_argument("--cases_json", type=str, default=None,
                        help="Path to JSON file containing case IDs (e.g., from TSNAC_01_split_cv.py)")
    args = parser.parse_args()
    
    # Load cases from JSON if provided
    if args.cases_json:
        cases = load_cases_from_json(args.cases_json)
        if cases:
            print(f"Loaded {len(cases)} cases from {args.cases_json}")
        else:
            cases = case_name_list
            print(f"Using default case list ({len(cases)} cases)")
    else:
        cases = case_name_list
    
    # Organize files
    organize_files(
        args.source, 
        args.output, 
        args.input_pattern, 
        args.label_pattern, 
        args.gt_pattern,
        cases
    )