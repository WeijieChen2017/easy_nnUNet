# easy_nnUNet

A simplified implementation of nnUNet based on TotalSegmentator for robust medical image segmentation. This pipeline enables conversion of non-attenuation corrected (NAC) images to attenuation corrected (AC) images.

## Overview

This repository provides an end-to-end pipeline for medical image segmentation using nnUNet, focusing on 5 anatomical regions:
- Organs (24 classes)
- Vertebrae (26 classes)
- Cardiac (18 classes) 
- Muscles (23 classes)
- Ribs (26 classes)

**Important:** This pipeline trains 5 separate part models, one for each anatomical region. Each model can be run on a separate GPU for efficient parallel processing.

## Pipeline Overview

This workflow consists of the following steps:

1. **Data Organization**:
   - Split the dataset into train/val/test sets
   - Convert to TotalSegmentator format, organizing files by case ID

2. **Data Conversion**:
   - Transform data into nnUNet format
   - Create 5 separate datasets, one for each anatomical region
   - Map original labels to region-specific consecutive labels

3. **Preprocessing**:
   - Run nnUNet's planning and preprocessing for each dataset
   - Generate preprocessed files optimized for training

4. **Training**:
   - Train 5 separate models, one for each anatomical region
   - Use nnUNetTrainerNoMirroring to avoid data augmentation with mirroring

5. **Inference**:
   - Generate predictions for test data
   - Output files are created in region-specific prediction directories

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker
- Medical imaging dataset organized in standard format

## Dataset Structure and Case Naming

The code currently uses case IDs in a specific format. To use your own dataset:

### Required File Structure

For each case in your dataset, you need the following files:
- `NAC_[case_id]_input.nii.gz`: The non-attenuation corrected image (input)
- `CTAC_[case_id]_segmap.nii.gz`: The segmentation label map
- `CTAC_[case_id]_gt.nii.gz`: The ground truth CTAC images

### Adapting for Your Dataset

1. **Case ID Format**: Replace the default case IDs with your own naming convention by editing `case_name_list` in:
   - `TSNAC_01_split_cv.py`
   - `TSNAC_01_convert_dataset_to_TS.py`

   For example, if your case IDs follow a patient identifier format:
   ```python
   # In both TSNAC_01_split_cv.py and TSNAC_01_convert_dataset_to_TS.py
   case_name_list = sorted([
       "PATIENT001", "PATIENT002", "PATIENT003", 
       # ... rest of your cases
   ])
   
   # Also update test_cases in TSNAC_01_split_cv.py if needed
   test_cases = [
       "PATIENT001", "PATIENT002", "PATIENT003",
       # ... your test cases
   ]
   ```

   Alternative formats could include:
   ```python
   # For study dates with anonymous IDs
   case_name_list = sorted([
       "STUDY20220101_001", "STUDY20220102_002", "STUDY20220103_003",
       # ... rest of your cases
   ])
   
   # For center-based identifiers
   case_name_list = sorted([
       "CENTER1_CASE001", "CENTER1_CASE002", "CENTER2_CASE001",
       # ... rest of your cases
   ])
   ```

2. **File Naming Convention**: If your files follow a different naming pattern, modify the following in `TSNAC_01_convert_dataset_to_TS.py`:
   ```python
   file_mappings = {
       f"CTAC_{case_name}_segmap.nii.gz": "label.nii.gz",
       f"NAC_{case_name}_input.nii.gz": "ct.nii.gz",
       f"CTAC_{case_name}_gt.nii.gz": "gt.nii.gz"
   }
   ```

   For example, if your files are named differently:
   ```python
   file_mappings = {
       f"SEGMENTATION_{case_name}.nii.gz": "label.nii.gz",
       f"INPUT_{case_name}.nii.gz": "ct.nii.gz",
       f"GROUNDTRUTH_{case_name}.nii.gz": "gt.nii.gz"
   }
   ```

3. **Directory Structure**: The default structure after processing will be:
   ```
   TS_NAC/
   ├── [case_id_1]/
   │   ├── ct.nii.gz  (from NAC_[case_id]_input.nii.gz)
   │   ├── label.nii.gz  (from CTAC_[case_id]_segmap.nii.gz)
   │   └── gt.nii.gz  (from CTAC_[case_id]_gt.nii.gz)
   ├── [case_id_2]/
   │   ├── ct.nii.gz
   │   ├── label.nii.gz
   │   └── gt.nii.gz
   └── ...
   ```

4. **Data Split**: The train/validation/test split is defined in `TSNAC_01_split_cv.py`. By default, it creates a JSON file at `./TS_NAC/data_split.json` with the split information. You can modify this path if needed:
   ```python
   # Change the output path and filename in TSNAC_01_split_cv.py
   output_dir = "./YOUR_OUTPUT_DIRECTORY"  # Change to your preferred directory
   output_file = os.path.join(output_dir, "your_split_file.json")  # Change filename
   ```

5. **Source and Output Directories**: Update the source and output directories in `TSNAC_01_convert_dataset_to_TS.py`:
   ```python
   if __name__ == "__main__":
       # Set your source and output directories here
       source_directory = "YOUR_SOURCE_DIRECTORY"  # Directory containing your original files
       output_directory = "YOUR_OUTPUT_DIRECTORY"  # Where to create the organized structure
       
       organize_files(source_directory, output_directory)
   ```

6. **nnUNet Dataset Directories**: When running the conversion to nnUNet format, you'll specify the output directories:
   ```bash
   python TSNAC_02_TS_dataset_to_nnUNet.py YOUR_DATA_DIR nnUNet/Dataset101_OR0 class_map_part_organs
   ```
   Replace `YOUR_DATA_DIR` with the directory containing your organized data (e.g., "TS_NAC" or your custom directory).

## Setup Instructions

### 1. Create Screen Sessions (Optional)

For managing multiple training jobs on different GPUs - each screen session will run one of the five part models:

```bash
screen -S GPU_0  # For Dataset101 (Organs)
screen -S GPU_1  # For Dataset102 (Vertebrae)
screen -S GPU_2  # For Dataset103 (Cardiac)
screen -S GPU_3  # For Dataset104 (Muscles)
screen -S GPU_4  # For Dataset105 (Ribs)
```

### 2. Start Docker Containers

Run Docker containers with GPU access (one container per GPU):

```bash
docker run --gpus device=0 -ti -v ./:/local --ipc=host petermcgor/nnunetv2:2.0.1  # For Dataset101 (Organs)
docker run --gpus device=1 -ti -v ./:/local --ipc=host petermcgor/nnunetv2:2.0.1  # For Dataset102 (Vertebrae)
docker run --gpus device=2 -ti -v ./:/local --ipc=host petermcgor/nnunetv2:2.0.1  # For Dataset103 (Cardiac)
docker run --gpus device=3 -ti -v ./:/local --ipc=host petermcgor/nnunetv2:2.0.1  # For Dataset104 (Muscles)
docker run --gpus device=4 -ti -v ./:/local --ipc=host petermcgor/nnunetv2:2.0.1  # For Dataset105 (Ribs)
```

### 3. Set Environment Variables

In each container, set the following environment variables:

```bash
cd /local/easy_nnUNet/
export nnUNet_raw=/local/easy_nnUNet/nnUNet
export nnUNet_results=/local/diffusion101/maisi/nnUNet/results
export nnUNet_preprocessed=/local/easy_nnUNet/nnUNet/preprocessed
```

## Data Preparation

### 1. Split Dataset

Split the dataset into train/validation/test sets:

```bash
python TSNAC_01_split_cv.py
```

### 2. Convert to TotalSegmentator Format

Format the dataset according to TotalSegmentator requirements:

```bash
python TSNAC_01_convert_dataset_to_TS.py
```

### 3. Convert to nnUNet Format

Transform the data into nnUNet-compatible format for each anatomical region:

```bash
python TSNAC_02_TS_dataset_to_nnUNet.py TS_NAC nnUNet/Dataset101_OR0 class_map_part_organs
python TSNAC_02_TS_dataset_to_nnUNet.py TS_NAC nnUNet/Dataset102_VE0 class_map_part_vertebrae
python TSNAC_02_TS_dataset_to_nnUNet.py TS_NAC nnUNet/Dataset103_CA0 class_map_part_cardiac
python TSNAC_02_TS_dataset_to_nnUNet.py TS_NAC nnUNet/Dataset104_MU0 class_map_part_muscles
python TSNAC_02_TS_dataset_to_nnUNet.py TS_NAC nnUNet/Dataset105_RI0 class_map_part_ribs
```

## Preprocessing

Preprocess the datasets with nnUNet's planning tools. In each screen session, run the appropriate command for its assigned model:

```bash
# In GPU_0 screen (Organs)
nnUNetv2_plan_and_preprocess -d 101 -pl ExperimentPlanner -c 3d_fullres -np 8

# In GPU_1 screen (Vertebrae)
nnUNetv2_plan_and_preprocess -d 102 -pl ExperimentPlanner -c 3d_fullres -np 8

# In GPU_2 screen (Cardiac)
nnUNetv2_plan_and_preprocess -d 103 -pl ExperimentPlanner -c 3d_fullres -np 8

# In GPU_3 screen (Muscles)
nnUNetv2_plan_and_preprocess -d 104 -pl ExperimentPlanner -c 3d_fullres -np 8

# In GPU_4 screen (Ribs)
nnUNetv2_plan_and_preprocess -d 105 -pl ExperimentPlanner -c 3d_fullres -np 8
```

Alternative preprocessing with specific GPU memory target:

```bash
# In GPU_0 screen (Organs)
nnUNetv2_plan_and_preprocess -pl ExperimentPlanner -c 3d_fullres -np 8 -gpu_memory_target 40 -d 101

# In GPU_1 screen (Vertebrae)
nnUNetv2_plan_and_preprocess -pl ExperimentPlanner -c 3d_fullres -np 8 -gpu_memory_target 40 -d 102

# In GPU_2 screen (Cardiac)
nnUNetv2_plan_and_preprocess -pl ExperimentPlanner -c 3d_fullres -np 8 -gpu_memory_target 40 -d 103

# In GPU_3 screen (Muscles)
nnUNetv2_plan_and_preprocess -pl ExperimentPlanner -c 3d_fullres -np 8 -gpu_memory_target 40 -d 104

# In GPU_4 screen (Ribs)
nnUNetv2_plan_and_preprocess -pl ExperimentPlanner -c 3d_fullres -np 8 -gpu_memory_target 40 -d 105
```

## Training

Train models for each anatomical region (run the appropriate command in each screen session):

```bash
# In GPU_0 screen (Organs)
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerNoMirroring

# In GPU_1 screen (Vertebrae)
nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainerNoMirroring

# In GPU_2 screen (Cardiac)
nnUNetv2_train 103 3d_fullres 0 -tr nnUNetTrainerNoMirroring

# In GPU_3 screen (Muscles)
nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerNoMirroring

# In GPU_4 screen (Ribs)
nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainerNoMirroring
```

## Inference

Generate predictions on test data (run in the appropriate screen session):

```bash
# In GPU_0 screen (Organs)
nnUNetv2_predict -i nnUNet/Dataset101_OR0/imagesTs -o nnUNet/Dataset101_OR0/imagesTs_pred -d 101 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

# In GPU_1 screen (Vertebrae)
nnUNetv2_predict -i nnUNet/Dataset102_VE0/imagesTs -o nnUNet/Dataset102_VE0/imagesTs_pred -d 102 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

# In GPU_2 screen (Cardiac)
nnUNetv2_predict -i nnUNet/Dataset103_CA0/imagesTs -o nnUNet/Dataset103_CA0/imagesTs_pred -d 103 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

# In GPU_3 screen (Muscles)
nnUNetv2_predict -i nnUNet/Dataset104_MU0/imagesTs -o nnUNet/Dataset104_MU0/imagesTs_pred -d 104 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0

# In GPU_4 screen (Ribs)
nnUNetv2_predict -i nnUNet/Dataset105_RI0/imagesTs -o nnUNet/Dataset105_RI0/imagesTs_pred -d 105 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0
```

## Dataset Class Mappings

The repository segments the following anatomical regions, each with its own dataset ID:

- Dataset101: Organs (24 classes)
- Dataset102: Vertebrae (26 classes)
- Dataset103: Cardiac (18 classes)
- Dataset104: Muscles (23 classes)
- Dataset105: Ribs (26 classes)

### Detailed Label Mappings

#### Organs (Dataset101)
```
Offset label 1 (spleen) -> New label 1
Offset label 2 (kidney_right) -> New label 2
Offset label 3 (kidney_left) -> New label 3
Offset label 4 (gallbladder) -> New label 4
Offset label 5 (liver) -> New label 5
Offset label 6 (stomach) -> New label 6
Offset label 7 (pancreas) -> New label 7
Offset label 8 (adrenal_gland_right) -> New label 8
Offset label 9 (adrenal_gland_left) -> New label 9
Offset label 10 (lung_upper_lobe_left) -> New label 10
Offset label 11 (lung_lower_lobe_left) -> New label 11
Offset label 12 (lung_upper_lobe_right) -> New label 12
Offset label 13 (lung_middle_lobe_right) -> New label 13
Offset label 14 (lung_lower_lobe_right) -> New label 14
Offset label 15 (esophagus) -> New label 15
Offset label 16 (trachea) -> New label 16
Offset label 17 (thyroid_gland) -> New label 17
Offset label 18 (small_bowel) -> New label 18
Offset label 19 (duodenum) -> New label 19
Offset label 20 (colon) -> New label 20
Offset label 21 (urinary_bladder) -> New label 21
Offset label 22 (prostate) -> New label 22
Offset label 23 (kidney_cyst_left) -> New label 23
Offset label 24 (kidney_cyst_right) -> New label 24
```

#### Vertebrae (Dataset102)
```
Offset label 25 (sacrum) -> New label 1
Offset label 26 (vertebrae_S1) -> New label 2
Offset label 27 (vertebrae_L5) -> New label 3
Offset label 28 (vertebrae_L4) -> New label 4
Offset label 29 (vertebrae_L3) -> New label 5
Offset label 30 (vertebrae_L2) -> New label 6
Offset label 31 (vertebrae_L1) -> New label 7
Offset label 32 (vertebrae_T12) -> New label 8
Offset label 33 (vertebrae_T11) -> New label 9
Offset label 34 (vertebrae_T10) -> New label 10
Offset label 35 (vertebrae_T9) -> New label 11
Offset label 36 (vertebrae_T8) -> New label 12
Offset label 37 (vertebrae_T7) -> New label 13
Offset label 38 (vertebrae_T6) -> New label 14
Offset label 39 (vertebrae_T5) -> New label 15
Offset label 40 (vertebrae_T4) -> New label 16
Offset label 41 (vertebrae_T3) -> New label 17
Offset label 42 (vertebrae_T2) -> New label 18
Offset label 43 (vertebrae_T1) -> New label 19
Offset label 44 (vertebrae_C7) -> New label 20
Offset label 45 (vertebrae_C6) -> New label 21
Offset label 46 (vertebrae_C5) -> New label 22
Offset label 47 (vertebrae_C4) -> New label 23
Offset label 48 (vertebrae_C3) -> New label 24
Offset label 49 (vertebrae_C2) -> New label 25
Offset label 50 (vertebrae_C1) -> New label 26
```

#### Cardiac (Dataset103)
```
Offset label 51 (heart) -> New label 1
Offset label 52 (aorta) -> New label 2
Offset label 53 (pulmonary_vein) -> New label 3
Offset label 54 (brachiocephalic_trunk) -> New label 4
Offset label 55 (subclavian_artery_right) -> New label 5
Offset label 56 (subclavian_artery_left) -> New label 6
Offset label 57 (common_carotid_artery_right) -> New label 7
Offset label 58 (common_carotid_artery_left) -> New label 8
Offset label 59 (brachiocephalic_vein_left) -> New label 9
Offset label 60 (brachiocephalic_vein_right) -> New label 10
Offset label 61 (atrial_appendage_left) -> New label 11
Offset label 62 (superior_vena_cava) -> New label 12
Offset label 63 (inferior_vena_cava) -> New label 13
Offset label 64 (portal_vein_and_splenic_vein) -> New label 14
Offset label 65 (iliac_artery_left) -> New label 15
Offset label 66 (iliac_artery_right) -> New label 16
Offset label 67 (iliac_vena_left) -> New label 17
Offset label 68 (iliac_vena_right) -> New label 18
```

#### Muscles (Dataset104)
```
Offset label 69 (humerus_left) -> New label 1
Offset label 70 (humerus_right) -> New label 2
Offset label 71 (scapula_left) -> New label 3
Offset label 72 (scapula_right) -> New label 4
Offset label 73 (clavicula_left) -> New label 5
Offset label 74 (clavicula_right) -> New label 6
Offset label 75 (femur_left) -> New label 7
Offset label 76 (femur_right) -> New label 8
Offset label 77 (hip_left) -> New label 9
Offset label 78 (hip_right) -> New label 10
Offset label 79 (spinal_cord) -> New label 11
Offset label 80 (gluteus_maximus_left) -> New label 12
Offset label 81 (gluteus_maximus_right) -> New label 13
Offset label 82 (gluteus_medius_left) -> New label 14
Offset label 83 (gluteus_medius_right) -> New label 15
Offset label 84 (gluteus_minimus_left) -> New label 16
Offset label 85 (gluteus_minimus_right) -> New label 17
Offset label 86 (autochthon_left) -> New label 18
Offset label 87 (autochthon_right) -> New label 19
Offset label 88 (iliopsoas_left) -> New label 20
Offset label 89 (iliopsoas_right) -> New label 21
Offset label 90 (brain) -> New label 22
Offset label 91 (skull) -> New label 23
```

#### Ribs (Dataset105)
```
Offset label 92 (rib_left_1) -> New label 1
Offset label 93 (rib_left_2) -> New label 2
Offset label 94 (rib_left_3) -> New label 3
Offset label 95 (rib_left_4) -> New label 4
Offset label 96 (rib_left_5) -> New label 5
Offset label 97 (rib_left_6) -> New label 6
Offset label 98 (rib_left_7) -> New label 7
Offset label 99 (rib_left_8) -> New label 8
Offset label 100 (rib_left_9) -> New label 9
Offset label 101 (rib_left_10) -> New label 10
Offset label 102 (rib_left_11) -> New label 11
Offset label 103 (rib_left_12) -> New label 12
Offset label 104 (rib_right_1) -> New label 13
Offset label 105 (rib_right_2) -> New label 14
Offset label 106 (rib_right_3) -> New label 15
Offset label 107 (rib_right_4) -> New label 16
Offset label 108 (rib_right_5) -> New label 17
Offset label 109 (rib_right_6) -> New label 18
Offset label 110 (rib_right_7) -> New label 19
Offset label 111 (rib_right_8) -> New label 20
Offset label 112 (rib_right_9) -> New label 21
Offset label 113 (rib_right_10) -> New label 22
Offset label 114 (rib_right_11) -> New label 23
Offset label 115 (rib_right_12) -> New label 24
Offset label 116 (sternum) -> New label 25
Offset label 117 (costal_cartilages) -> New label 26
```

For the complete list of label mappings, please refer to the label mapping tables in the `TSNAC_02_TS_dataset_to_nnUNet.py` file.


