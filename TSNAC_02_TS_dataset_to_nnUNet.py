#!/usr/bin/env python
"""
TSNAC_02_TS_dataset_to_nnUNet.py - Convert TotalSegmentator dataset to nnUNet format

This script takes a dataset in TotalSegmentator format and converts it to nnUNet format,
creating the necessary directory structure and metadata files. It also maps labels from
different anatomical regions to consecutive label values suitable for nnUNet training.

Usage:
    python TSNAC_02_TS_dataset_to_nnUNet.py <dataset_path> <nnunet_dataset_path> <class_map_name>
                                           [--split_file SPLIT_FILE] [--skip_confirmation]
                                           [--input_filename INPUT_FILENAME] [--label_filename LABEL_FILENAME]

Arguments:
    dataset_path: Path to the dataset in TotalSegmentator format
    nnunet_dataset_path: Path to store the converted nnUNet dataset
    class_map_name: Which anatomical region to extract (e.g., class_map_part_organs)

Example:
    python TSNAC_02_TS_dataset_to_nnUNet.py ./TS_NAC ./nnUNet/Dataset101_OR0 class_map_part_organs --skip_confirmation
"""

import sys
import os
from pathlib import Path
import shutil
import json
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# from TS_NAC.map_to_binary import class_map_5_parts

class_map_5_parts = {
    # 24 classes
    "class_map_part_organs": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right"
    },

    # 26 classes
    "class_map_part_vertebrae": {
        1: "sacrum",
        2: "vertebrae_S1",
        3: "vertebrae_L5",
        4: "vertebrae_L4",
        5: "vertebrae_L3",
        6: "vertebrae_L2",
        7: "vertebrae_L1",
        8: "vertebrae_T12",
        9: "vertebrae_T11",
        10: "vertebrae_T10",
        11: "vertebrae_T9",
        12: "vertebrae_T8",
        13: "vertebrae_T7",
        14: "vertebrae_T6",
        15: "vertebrae_T5",
        16: "vertebrae_T4",
        17: "vertebrae_T3",
        18: "vertebrae_T2",
        19: "vertebrae_T1",
        20: "vertebrae_C7",
        21: "vertebrae_C6",
        22: "vertebrae_C5",
        23: "vertebrae_C4",
        24: "vertebrae_C3",
        25: "vertebrae_C2",
        26: "vertebrae_C1"
    },

    # 18
    "class_map_part_cardiac": {
        1: "heart",
        2: "aorta",
        3: "pulmonary_vein",
        4: "brachiocephalic_trunk",
        5: "subclavian_artery_right",
        6: "subclavian_artery_left",
        7: "common_carotid_artery_right",
        8: "common_carotid_artery_left",
        9: "brachiocephalic_vein_left",
        10: "brachiocephalic_vein_right",
        11: "atrial_appendage_left",
        12: "superior_vena_cava",
        13: "inferior_vena_cava",
        14: "portal_vein_and_splenic_vein",
        15: "iliac_artery_left",
        16: "iliac_artery_right",
        17: "iliac_vena_left",
        18: "iliac_vena_right"
    },

    # 23
    "class_map_part_muscles": {
        1: "humerus_left",
        2: "humerus_right",
        3: "scapula_left",
        4: "scapula_right",
        5: "clavicula_left",
        6: "clavicula_right",
        7: "femur_left",
        8: "femur_right",
        9: "hip_left",
        10: "hip_right",
        11: "spinal_cord",
        12: "gluteus_maximus_left",
        13: "gluteus_maximus_right",
        14: "gluteus_medius_left",
        15: "gluteus_medius_right",
        16: "gluteus_minimus_left",
        17: "gluteus_minimus_right",
        18: "autochthon_left",
        19: "autochthon_right",
        20: "iliopsoas_left",
        21: "iliopsoas_right",
        22: "brain",
        23: "skull"
    },

    # 26 classes
    # 12. ribs start from vertebrae T12
    # Small subset of population (roughly 8%) have 13. rib below 12. rib
    #  (would start from L1 then)
    #  -> this has label rib_12
    # Even smaller subset (roughly 1%) has extra rib above 1. rib   ("Halsrippe")
    #  (the extra rib would start from C7)
    #  -> this has label rib_1
    #
    # Quite often only 11 ribs (12. ribs probably so small that not found). Those
    # cases often wrongly segmented.
    "class_map_part_ribs": {
        1: "rib_left_1",
        2: "rib_left_2",
        3: "rib_left_3",
        4: "rib_left_4",
        5: "rib_left_5",
        6: "rib_left_6",
        7: "rib_left_7",
        8: "rib_left_8",
        9: "rib_left_9",
        10: "rib_left_10",
        11: "rib_left_11",
        12: "rib_left_12",
        13: "rib_right_1",
        14: "rib_right_2",
        15: "rib_right_3",
        16: "rib_right_4",
        17: "rib_right_5",
        18: "rib_right_6",
        19: "rib_right_7",
        20: "rib_right_8",
        21: "rib_right_9",
        22: "rib_right_10",
        23: "rib_right_11",
        24: "rib_right_12",
        25: "sternum",
        26: "costal_cartilages"
    },   # "test": class_map["test"]
}

offset_labels = [
    0,
    len(class_map_5_parts["class_map_part_organs"]),
    len(class_map_5_parts["class_map_part_organs"])+len(class_map_5_parts["class_map_part_vertebrae"]),
    len(class_map_5_parts["class_map_part_organs"])+len(class_map_5_parts["class_map_part_vertebrae"])+len(class_map_5_parts["class_map_part_cardiac"]),
    len(class_map_5_parts["class_map_part_organs"])+len(class_map_5_parts["class_map_part_vertebrae"])+len(class_map_5_parts["class_map_part_cardiac"])+len(class_map_5_parts["class_map_part_muscles"]),
]

# offset_labels = [
#     0,
#     24,
#     24+26,
#     24+26+28,
#     24+26+28+23,
# ]


def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels, dataset_name="TS_NAC", description=None):
    """
    Generate dataset.json and splits_final.json for nnUNet
    
    Args:
        foldername (str): Name of the nnUNet dataset folder (e.g., Dataset101_OR0)
        subjects_train (list): List of training subject IDs
        subjects_val (list): List of validation subject IDs
        labels (list): List of label values
        dataset_name (str): Name of the dataset
        description (str): Description of the dataset
    """
    print("Creating dataset.json...")
    out_base = Path(os.environ.get('nnUNet_raw', '.')) / foldername
    out_base.mkdir(parents=True, exist_ok=True)

    # Create dataset.json
    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = description or "NAC to AC conversion using TotalSegmentator"
    json_dict['reference'] = ""
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "1.0"
    json_dict['channel_names'] = {"0": "NAC"}
    json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + list(labels))}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)
    print(f"Created dataset.json at {out_base / 'dataset.json'}")

    # Create splits_final.json
    print("Creating splits_final.json...")
    output_folder_pkl = Path(os.environ.get('nnUNet_preprocessed', '.')) / foldername
    output_folder_pkl.mkdir(parents=True, exist_ok=True)

    # Create splits format expected by nnUNet
    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })

    print(f"  - Number of folds: {len(splits)}")
    print(f"  - Training subjects (fold 0): {len(splits[0]['train'])}")
    print(f"  - Validation subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)
    print(f"Created splits_final.json at {output_folder_pkl / 'splits_final.json'}")


def combine_labels(ref_img, file_out, masks):
    """
    Combine multiple binary masks into a single label map
    
    Args:
        ref_img (str): Path to reference image to get dimensions and affine
        file_out (str): Path to output file
        masks (list): List of paths to input mask files
    """
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx+1
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


def get_label_range(class_map_name):
    """
    Get the label range for the given class map part
    
    Args:
        class_map_name (str): Name of the class map part
        
    Returns:
        range: Range of label values for the specified class map
    """
    if class_map_name == "class_map_part_organs":
        return range(1, len(class_map_5_parts["class_map_part_organs"]) + 1)
    elif class_map_name == "class_map_part_vertebrae":
        start = offset_labels[0] + 1
        end = start + len(class_map_5_parts["class_map_part_vertebrae"])
        return range(start, end)
    elif class_map_name == "class_map_part_cardiac":
        start = offset_labels[1] + 1
        end = start + len(class_map_5_parts["class_map_part_cardiac"])
        return range(start, end)
    elif class_map_name == "class_map_part_muscles":
        start = offset_labels[2] + 1
        end = start + len(class_map_5_parts["class_map_part_muscles"])
        return range(start, end)
    elif class_map_name == "class_map_part_ribs":
        start = offset_labels[3] + 1
        end = start + len(class_map_5_parts["class_map_part_ribs"])
        return range(start, end)
    return range(0)


def get_label_mapping(class_map_name, class_map, skip_confirmation=False):
    """
    Get and verify the label mapping for the given class map part
    
    Args:
        class_map_name (str): Name of the class map part
        class_map (dict): Dictionary mapping label values to organ names
        skip_confirmation (bool): Whether to skip confirmation prompt
        
    Returns:
        dict: Mapping from offset labels to consecutive numbers starting from 1
    """
    # Get the part index to determine the offset
    part_index = list(class_map_5_parts.keys()).index(class_map_name)
    offset = offset_labels[part_index]
    
    # Create mapping from offset labels to consecutive numbers starting from 1
    new_label_mapping = {}
    for idx, (label_val, organ_name) in enumerate(class_map.items(), start=1):
        offset_label = label_val + offset
        new_label_mapping[offset_label] = idx
    
    # Print mapping for verification
    print(f"\nLabel mapping for {class_map_name}:")
    for offset_label, new_val in new_label_mapping.items():
        try:
            # Get the original label value (without offset)
            orig_label = offset_label - offset
            organ_name = class_map[orig_label]
            print(f"Offset label {offset_label} ({organ_name}) -> New label {new_val}")
        except KeyError:
            print(f"Warning: No organ name found for offset label {offset_label} -> New label {new_val}")
            continue
    
    # Ask for confirmation before proceeding
    if not skip_confirmation:
        proceed = input("\nDoes the mapping look correct? (yes/no): ")
        if proceed.lower() != 'yes':
            print("Aborting operation...")
            sys.exit(1)
    
    return new_label_mapping


def extract_selected_labels(label_file, output_file, label_mapping):
    """
    Create a new label file containing only the selected organ classes
    
    Args:
        label_file (str): Path to input label file
        output_file (str): Path to output label file
        label_mapping (dict): Mapping from original label values to new label values
    """
    try:
        img = nib.load(label_file)
        data = img.get_fdata()
        
        # Create empty array for new labels
        new_data = np.zeros_like(data)
        
        # Copy only the selected organ classes with new label values
        for orig_val, new_val in label_mapping.items():
            new_data[data == orig_val] = new_val
        
        # Save the new label file
        new_img = nib.Nifti1Image(new_data.astype(np.uint8), img.affine)
        nib.save(new_img, output_file)
    except Exception as e:
        print(f"Error processing {label_file}: {e}")
        return False
    
    return True


def main():
    """Main function to convert TotalSegmentator dataset to nnUNet format"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert TotalSegmentator dataset to nnUNet format")
    parser.add_argument("dataset_path", type=str, help="Path to dataset in TotalSegmentator format")
    parser.add_argument("nnunet_path", type=str, help="Path to store the converted nnUNet dataset")
    parser.add_argument("class_map_name", type=str, choices=list(class_map_5_parts.keys()),
                        help="Which anatomical region to extract")
    parser.add_argument("--split_file", type=str, default="data_split.json",
                        help="Name of the JSON file containing train/val/test splits")
    parser.add_argument("--skip_confirmation", action="store_true",
                        help="Skip confirmation prompt for label mapping")
    parser.add_argument("--input_filename", type=str, default="ct.nii.gz",
                        help="Filename of input images within case directories")
    parser.add_argument("--label_filename", type=str, default="label.nii.gz",
                        help="Filename of label images within case directories")
    parser.add_argument("--dataset_name", type=str, default="TS_NAC",
                        help="Name for the dataset in dataset.json")
    parser.add_argument("--description", type=str, default=None,
                        help="Description for the dataset in dataset.json")
    
    # Check for positional arguments for backward compatibility
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        args = parser.parse_args()
    else:
        print("Error: Required positional arguments missing")
        parser.print_help()
        sys.exit(1)

    # Convert paths to Path objects
    dataset_path = Path(args.dataset_path)
    nnunet_path = Path(args.nnunet_path)
    
    # Verify class map exists
    if args.class_map_name not in class_map_5_parts:
        print(f"Error: Class map '{args.class_map_name}' not found. Available maps:")
        for name in class_map_5_parts.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    # Get the class map
    class_map = class_map_5_parts[args.class_map_name]
    
    # Create nnUNet directory structure
    print(f"Creating directory structure in {nnunet_path}")
    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)
    
    # Find split JSON file
    split_json = dataset_path / args.split_file
    if not split_json.exists():
        print(f"Error: Split file not found: {split_json}")
        print("Please run TSNAC_01_split_cv.py first or specify correct split file path")
        sys.exit(1)
    
    # Load split information
    try:
        with open(split_json, 'r') as f:
            split_data = json.load(f)
        
        subjects_train = split_data.get('train', [])
        subjects_val = split_data.get('val', [])
        subjects_test = split_data.get('test', [])
        
        print(f"Loaded split information from {split_json}:")
        print(f"  - Training subjects: {len(subjects_train)}")
        print(f"  - Validation subjects: {len(subjects_val)}")
        print(f"  - Test subjects: {len(subjects_test)}")
    except Exception as e:
        print(f"Error loading split file: {e}")
        sys.exit(1)
    
    # Get and verify label mapping
    label_mapping = get_label_mapping(args.class_map_name, class_map, args.skip_confirmation)
    
    # Process training and validation data
    print("\nCopying training and validation data...")
    success_count = 0
    error_count = 0
    
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        input_file = subject_path / args.input_filename
        label_file = subject_path / args.label_filename
        
        # Check if files exist
        if not input_file.exists():
            print(f"Warning: Input file not found for subject {subject}: {input_file}")
            error_count += 1
            continue
        
        if not label_file.exists():
            print(f"Warning: Label file not found for subject {subject}: {label_file}")
            error_count += 1
            continue
        
        # Copy input file
        try:
            shutil.copy(input_file, nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        except Exception as e:
            print(f"Error copying input file for subject {subject}: {e}")
            error_count += 1
            continue
        
        # Create new label file
        if extract_selected_labels(
            label_file,
            nnunet_path / "labelsTr" / f"{subject}.nii.gz",
            label_mapping
        ):
            success_count += 1
        else:
            error_count += 1
    
    # Process test data
    print("\nCopying test data...")
    test_success = 0
    test_error = 0
    
    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        input_file = subject_path / args.input_filename
        label_file = subject_path / args.label_filename
        
        # Check if files exist
        if not input_file.exists():
            print(f"Warning: Input file not found for test subject {subject}: {input_file}")
            test_error += 1
            continue
        
        if not label_file.exists():
            print(f"Warning: Label file not found for test subject {subject}: {label_file}")
            test_error += 1
            continue
        
        # Copy input file
        try:
            shutil.copy(input_file, nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        except Exception as e:
            print(f"Error copying input file for test subject {subject}: {e}")
            test_error += 1
            continue
        
        # Create new label file
        if extract_selected_labels(
            label_file,
            nnunet_path / "labelsTs" / f"{subject}.nii.gz",
            label_mapping
        ):
            test_success += 1
        else:
            test_error += 1
    
    # Generate dataset.json and splits_final.json
    labels = list(class_map.keys())
    generate_json_from_dir_v2(
        nnunet_path.name, 
        subjects_train, 
        subjects_val, 
        labels,
        args.dataset_name,
        args.description
    )
    
    # Print summary
    print("\nConversion complete:")
    print(f"  - Training/validation subjects processed: {success_count}/{len(subjects_train) + len(subjects_val)}")
    print(f"  - Test subjects processed: {test_success}/{len(subjects_test)}")
    if error_count > 0 or test_error > 0:
        print(f"  - Errors encountered: {error_count + test_error}")
    print(f"  - Data saved to: {nnunet_path}")


if __name__ == "__main__":
    main()


