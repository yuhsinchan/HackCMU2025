#!/usr/bin/env python3
"""
Example usage script for Body Tracking Data Collector

This script demonstrates different ways to use the body tracking data collector
and provides examples of how to analyze the collected data.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add the parent directory to the path to import the data collector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_basic_collection():
    """Basic example of running the data collector"""
    print("=== Basic Data Collection Example ===")
    print("This will start the data collector with default settings.")
    print("Controls:")
    print("  'r' - Start/Stop recording")
    print("  's' - Save session data")
    print("  'q' - Quit")
    print("\nPress Enter to continue or Ctrl+C to skip...")
    
    try:
        input()
        from body_tracking_data_collector import BodyTrackingDataCollector
        
        class MockOpt:
            def __init__(self):
                self.input_svo_file = ''
                self.ip_address = ''
                self.resolution = 'HD1080'
                
        opt = MockOpt()
        collector = BodyTrackingDataCollector(opt)
        collector.run()
        
    except KeyboardInterrupt:
        print("Skipped basic collection example")
    except Exception as e:
        print(f"Error running basic collection: {e}")


def visualize_collected_data(data_file, mode='report'):
    """Visualize collected body tracking data using the visualizer"""
    print(f"\n=== Visualizing Data: {data_file} ===")
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
        
    try:
        from visualize_data import BodyTrackingVisualizer
        
        visualizer = BodyTrackingVisualizer(data_file)
        
        if mode == 'report':
            # Generate comprehensive report
            visualizer.generate_report()
        elif mode == 'interactive':
            # Launch interactive viewer
            visualizer.interactive_viewer()
        elif mode == 'skeleton':
            # Show 3D skeleton
            visualizer.plot_skeleton_3d()
        elif mode == 'trails':
            # Show motion trails
            visualizer.plot_motion_trails()
        elif mode == 'animation':
            # Create animation
            visualizer.create_animation()
        elif mode == 'heatmap':
            # Show keypoint heatmap
            visualizer.plot_keypoint_heatmap()
        else:
            print(f"Unknown visualization mode: {mode}")
            
    except ImportError as e:
        print(f"Visualization requires additional packages: {e}")
        print("Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"Error during visualization: {e}")


def analyze_collected_data(data_file):
    """Analyze collected body tracking data"""
    print(f"\n=== Analyzing Data: {data_file} ===")
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
        
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        session_info = data.get('session_info', {})
        frame_data = data.get('data', [])
        
        print(f"Session: {session_info.get('session_name', 'Unknown')}")
        print(f"Total frames captured: {session_info.get('total_frames', 0)}")
        print(f"Frames with body data: {session_info.get('recorded_frames', 0)}")
        print(f"Body format: {session_info.get('body_format', 'Unknown')}")
        
        if frame_data:
            # Analyze body detection
            total_bodies = 0
            body_ids = set()
            frame_counts = []
            
            for frame in frame_data:
                bodies_in_frame = len(frame.get('bodies', []))
                total_bodies += bodies_in_frame
                frame_counts.append(bodies_in_frame)
                
                for body in frame.get('bodies', []):
                    body_ids.add(body.get('id', -1))
                    
            print(f"\nBody Detection Analysis:")
            print(f"  Unique bodies detected: {len(body_ids)}")
            print(f"  Total body detections: {total_bodies}")
            print(f"  Average bodies per frame: {np.mean(frame_counts):.2f}")
            print(f"  Max bodies in single frame: {max(frame_counts) if frame_counts else 0}")
            
            # Analyze timing
            timestamps = [frame.get('timestamp', 0) for frame in frame_data]
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                fps = len(timestamps) / duration if duration > 0 else 0
                print(f"\nTiming Analysis:")
                print(f"  Recording duration: {duration:.2f} seconds")
                print(f"  Average FPS: {fps:.2f}")
                
            # Analyze keypoint data quality
            analyze_keypoint_quality(frame_data)
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
    except Exception as e:
        print(f"Error analyzing data: {e}")


def analyze_keypoint_quality(frame_data):
    """Analyze the quality of keypoint data"""
    print(f"\nKeypoint Quality Analysis:")
    
    all_keypoints_2d = []
    all_keypoints_3d = []
    
    for frame in frame_data:
        for body in frame.get('bodies', []):
            kp_2d = body.get('keypoints_2d', [])
            kp_3d = body.get('keypoints_3d', [])
            
            if kp_2d:
                all_keypoints_2d.extend(kp_2d)
            if kp_3d:
                all_keypoints_3d.extend(kp_3d)
                
    if all_keypoints_2d:
        kp_2d_array = np.array(all_keypoints_2d)
        valid_2d = np.all(np.isfinite(kp_2d_array), axis=1)
        print(f"  Valid 2D keypoints: {np.sum(valid_2d)}/{len(valid_2d)} ({100*np.mean(valid_2d):.1f}%)")
        
    if all_keypoints_3d:
        kp_3d_array = np.array(all_keypoints_3d)
        valid_3d = np.all(np.isfinite(kp_3d_array), axis=1)
        print(f"  Valid 3D keypoints: {np.sum(valid_3d)}/{len(valid_3d)} ({100*np.mean(valid_3d):.1f}%)")
        
        # Analyze 3D keypoint distribution
        if np.any(valid_3d):
            valid_kp_3d = kp_3d_array[valid_3d]
            print(f"  3D position ranges:")
            print(f"    X: {np.min(valid_kp_3d[:, 0]):.2f} to {np.max(valid_kp_3d[:, 0]):.2f} m")
            print(f"    Y: {np.min(valid_kp_3d[:, 1]):.2f} to {np.max(valid_kp_3d[:, 1]):.2f} m")
            print(f"    Z: {np.min(valid_kp_3d[:, 2]):.2f} to {np.max(valid_kp_3d[:, 2]):.2f} m")


def convert_to_csv(json_file, output_csv=None):
    """Convert JSON data to CSV format for easier analysis"""
    if output_csv is None:
        base_name = os.path.splitext(json_file)[0]
        output_csv = f"{base_name}_converted.csv"
        
    print(f"\nConverting {json_file} to {output_csv}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        rows = []
        frame_data = data.get('data', [])
        
        for frame in frame_data:
            frame_number = frame.get('frame_number', 0)
            timestamp = frame.get('timestamp', 0)
            
            for body in frame.get('bodies', []):
                body_id = body.get('id', -1)
                confidence = body.get('confidence', 0)
                
                # Create base row
                base_row = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'body_id': body_id,
                    'confidence': confidence
                }
                
                # Add keypoint data
                kp_2d = body.get('keypoints_2d', [])
                kp_3d = body.get('keypoints_3d', [])
                
                for i, (kp2, kp3) in enumerate(zip(kp_2d, kp_3d)):
                    base_row.update({
                        f'kp_{i}_x_2d': kp2[0] if len(kp2) > 0 else np.nan,
                        f'kp_{i}_y_2d': kp2[1] if len(kp2) > 1 else np.nan,
                        f'kp_{i}_x_3d': kp3[0] if len(kp3) > 0 else np.nan,
                        f'kp_{i}_y_3d': kp3[1] if len(kp3) > 1 else np.nan,
                        f'kp_{i}_z_3d': kp3[2] if len(kp3) > 2 else np.nan,
                    })
                    
                rows.append(base_row)
                
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
            print(f"Successfully converted {len(rows)} rows to CSV")
        else:
            print("No data found to convert")
            
    except Exception as e:
        print(f"Error converting to CSV: {e}")


def list_collected_data(data_dir="collected_data"):
    """List all collected data files"""
    print(f"\n=== Collected Data in {data_dir} ===")
    
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return []
        
    json_files = []
    csv_files = []
    
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith('.json'):
            json_files.append(filepath)
        elif filename.endswith('.csv'):
            csv_files.append(filepath)
            
    print(f"JSON files found: {len(json_files)}")
    for f in json_files:
        print(f"  {f}")
        
    print(f"CSV files found: {len(csv_files)}")
    for f in csv_files:
        print(f"  {f}")
        
    return json_files


def main():
    parser = argparse.ArgumentParser(description="Body Tracking Data Collector Examples")
    parser.add_argument('--mode', choices=['collect', 'analyze', 'convert', 'list', 'visualize'], 
                       default='list', help='Mode of operation')
    parser.add_argument('--data_file', type=str, help='Data file to analyze or convert')
    parser.add_argument('--data_dir', type=str, default='collected_data', 
                       help='Directory containing collected data')
    parser.add_argument('--output_csv', type=str, help='Output CSV file for conversion')
    parser.add_argument('--viz_mode', type=str, default='report',
                       choices=['report', 'interactive', 'skeleton', 'trails', 'animation', 'heatmap'],
                       help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        example_basic_collection()
    elif args.mode == 'analyze':
        if args.data_file:
            analyze_collected_data(args.data_file)
        else:
            # Analyze the most recent file
            json_files = list_collected_data(args.data_dir)
            if json_files:
                analyze_collected_data(json_files[-1])
            else:
                print("No data files found to analyze")
    elif args.mode == 'convert':
        if args.data_file:
            convert_to_csv(args.data_file, args.output_csv)
        else:
            print("Please specify --data_file for conversion")
    elif args.mode == 'list':
        list_collected_data(args.data_dir)
    elif args.mode == 'visualize':
        if args.data_file:
            visualize_collected_data(args.data_file, args.viz_mode)
        else:
            # Visualize the most recent file
            json_files = list_collected_data(args.data_dir)
            if json_files:
                visualize_collected_data(json_files[-1], args.viz_mode)
            else:
                print("No data files found to visualize")
        
    print("\nExample usage commands:")
    print("  python examples.py --mode collect")
    print("  python examples.py --mode list")
    print("  python examples.py --mode analyze --data_file collected_data/session.json")
    print("  python examples.py --mode convert --data_file collected_data/session.json")
    print("  python examples.py --mode visualize --data_file collected_data/session.json")
    print("  python examples.py --mode visualize --data_file collected_data/session.json --viz_mode interactive")


if __name__ == '__main__':
    main()