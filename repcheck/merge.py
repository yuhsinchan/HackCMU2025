"""
Simple script to merge multiple pose data JSON files into a single file
"""

import json
import glob
import os
import sys


def merge_json_files(input_pattern, output_file):
    """ 
    Merge multiple JSON files with pose data into a single file
    
    Args:
        input_pattern: Pattern to match JSON files (e.g., "data/*.json" or ["file1.json", "file2.json"])
        output_file: Output filename for merged data
    """
    merged_data = []
    
    # Get list of files
    if isinstance(input_pattern, str):
        files = glob.glob(input_pattern)
    else:
        files = input_pattern
    
    files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(files)} files to merge:")
    for file in files:
        print(f"  - {file}")
    
    # Process each file
    for file_path in files:
        try:
            print(f"\nProcessing {file_path}...")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list, skipping")
                continue
            
            # Add each sample from this file
            valid_samples = 0
            for sample in data:
                if validate_sample(sample):
                    merged_data.append(sample)
                    valid_samples += 1
                else:
                    print(f"Warning: Invalid sample found in {file_path}")
            
            print(f"Added {valid_samples} valid samples")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Save merged data
    print(f"\nSaving {len(merged_data)} total samples to {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print("Merge completed!")
    
    # Print summary
    print_summary(merged_data)


def validate_sample(sample):
    """
    Validate that a sample has the correct structure
    
    Expected structure:
    {
        "label": 0 or 1,
        "data": [
            {
                "ts": timestamp,
                "keypoints_3d": [[x,y,z], [x,y,z], ...]
            },
            ...
        ]
    }
    """
    try:
        # Check required fields
        if 'label' not in sample or 'data' not in sample:
            print("Missing 'label' or 'data' field")
            return False
        
        # Check label is a list of length 6 with binary values
        label = sample['label']
        if not isinstance(label, list) or len(label) != 6:
            print(f"Label must be a list of length 6, got: {label}")
            return False
        
        # Check all values are 0 or 1
        if not all(isinstance(x, (int, float)) and x in [0, 1] for x in label):
            print(f"Label values must be 0 or 1, got: {label}")
            return False
        
        # Check data is a list
        if not isinstance(sample['data'], list):
            print("'data' is not a list")
            return False
        
        # Check at least one frame
        if len(sample['data']) == 0:
            print("No frames found")
            return False
        
        # Check first frame structure
        first_frame = sample['data'][0]
        if 'ts' not in first_frame or 'keypoints_3d' not in first_frame:
            print("Frame missing 'ts' or 'keypoints_3d'")
            return False
        
        # Check keypoints structure
        keypoints = first_frame['keypoints_3d']
        if not isinstance(keypoints, list) or len(keypoints) == 0:
            print("Invalid 'keypoints_3d' structure")
            return False
        
        # Check first keypoint has 3 coordinates
        if not isinstance(keypoints[0], list) or len(keypoints[0]) != 3:
            print("Keypoint does not have 3 coordinates")
            return False
        
        return True
        
    except:
        return False


def print_summary(merged_data):
    """Print summary statistics of merged data"""
    print(f"\n{'='*50}")
    print("MERGE SUMMARY")
    print(f"{'='*50}")
    
    total_samples = len(merged_data)
    
    # Count occurrences of each label type
    label_counts = [0] * 6  # One counter for each class
    samples_per_class = [0] * 6  # Number of samples with each class present
    
    for sample in merged_data:
        label = sample['label']
        for i, value in enumerate(label):
            if value == 1:
                label_counts[i] += 1
                samples_per_class[i] += 1
    
    print(f"Total samples: {total_samples}")
    
    print(f"\nLabel distribution:")
    for i in range(6):
        count = samples_per_class[i]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  Class {i}: {count} samples ({percentage:.1f}%)")
    
    # Print co-occurrence statistics if any exist
    cooccurrences = 0
    for sample in merged_data:
        if sum(sample['label']) > 1:
            cooccurrences += 1
    
    if cooccurrences > 0:
        print(f"\nMulti-label statistics:")
        print(f"  Samples with multiple labels: {cooccurrences}")
        percentage = (cooccurrences / total_samples * 100) if total_samples > 0 else 0
        print(f"  Percentage with multiple labels: {percentage:.1f}%")
    
    if total_samples > 0:
        # Frame statistics
        frame_counts = [len(sample['data']) for sample in merged_data]
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)
        avg_frames = sum(frame_counts) / len(frame_counts)
        
        print(f"\nFrame statistics:")
        print(f"Min frames per sample: {min_frames}")
        print(f"Max frames per sample: {max_frames}")
        print(f"Average frames per sample: {avg_frames:.1f}")
        
        # Keypoint statistics
        if merged_data[0]['data']:
            num_keypoints = len(merged_data[0]['data'][0]['keypoints_3d'])
            print(f"Keypoints per frame: {num_keypoints}")


def main():
    """Main function with command line interface"""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python merge_data.py <input_pattern> <output_file>")
        print("\nExamples:")
        print("  python merge_data.py 'data/*.json' merged_data.json")
        print("  python merge_data.py 'file1.json file2.json file3.json' merged_data.json")
        print("  python merge_data.py '*.json' all_data.json")
        return
    
    input_pattern = sys.argv[1]
    output_file = sys.argv[2]
    
    # Handle space-separated file list
    if not ('*' in input_pattern or '?' in input_pattern):
        # Split space-separated files
        files = input_pattern.split()
        merge_json_files(files, output_file)
    else:
        # Use glob pattern
        merge_json_files(input_pattern, output_file)


def quick_merge(folder_path=".", output_file="merged_data.json"):
    """
    Quick merge function for interactive use
    Merges all JSON files in a folder
    """
    pattern = os.path.join(folder_path, "*.json")
    merge_json_files(pattern, output_file)


if __name__ == "__main__":
    main()