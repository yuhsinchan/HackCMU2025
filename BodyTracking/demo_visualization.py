#!/usr/bin/env python3
"""
Quick demo script to test visualization with sample data
"""

import json
import numpy as np
import os
from datetime import datetime
import time

def create_sample_data(filename="sample_body_tracking_data.json"):
    """Create sample body tracking data for testing visualization"""
    
    # Create sample session with walking motion
    session_info = {
        "session_name": "sample_walking_session",
        "start_time": time.time(),
        "total_frames": 100,
        "recorded_frames": 50,
        "body_format": "BODY_FORMAT.BODY_34",
        "detection_model": "BODY_TRACKING_MODEL.HUMAN_BODY_FAST"
    }
    
    frame_data = []
    
    # Simulate a person walking (50 frames)
    for frame_num in range(50):
        # Simple walking simulation
        t = frame_num * 0.1  # Time in seconds
        
        # Walking pattern: forward movement with oscillating legs
        base_x = t * 0.5  # Walking forward
        base_y = 0.0
        base_z = 1.0  # Standing height
        
        # Oscillating step pattern
        step_phase = np.sin(t * 3.0) * 0.3  # Leg separation
        arm_swing = np.sin(t * 3.0) * 0.2   # Arm swing
        
        # Create 34 keypoints for BODY_34 format
        keypoints_3d = []
        keypoints_2d = []
        
        # Define base skeleton positions (standing upright)
        skeleton_template = [
            [0, 0, 0.8],      # 0: PELVIS
            [0, 0, 0.9],      # 1: NAVAL_SPINE  
            [0, 0, 1.1],      # 2: CHEST_SPINE
            [0, 0, 1.3],      # 3: NECK
            [-0.1, 0, 1.25],  # 4: LEFT_CLAVICLE
            [-0.15, 0, 1.2],  # 5: LEFT_SHOULDER
            [-0.2, 0, 1.0],   # 6: LEFT_ELBOW
            [-0.25, 0, 0.8],  # 7: LEFT_WRIST
            [-0.3, 0, 0.75],  # 8: LEFT_HAND
            [-0.32, 0, 0.73], # 9: LEFT_HANDTIP
            [-0.28, 0, 0.77], # 10: LEFT_THUMB
            [0.1, 0, 1.25],   # 11: RIGHT_CLAVICLE
            [0.15, 0, 1.2],   # 12: RIGHT_SHOULDER
            [0.2, 0, 1.0],    # 13: RIGHT_ELBOW
            [0.25, 0, 0.8],   # 14: RIGHT_WRIST
            [0.3, 0, 0.75],   # 15: RIGHT_HAND
            [0.32, 0, 0.73],  # 16: RIGHT_HANDTIP
            [0.28, 0, 0.77],  # 17: RIGHT_THUMB
            [-0.1, 0, 0.8],   # 18: LEFT_HIP
            [-0.1, 0, 0.4],   # 19: LEFT_KNEE
            [-0.1, 0, 0.0],   # 20: LEFT_ANKLE
            [-0.1, 0.1, 0.0], # 21: LEFT_FOOT
            [0.1, 0, 0.8],    # 22: RIGHT_HIP
            [0.1, 0, 0.4],    # 23: RIGHT_KNEE
            [0.1, 0, 0.0],    # 24: RIGHT_ANKLE
            [0.1, 0.1, 0.0],  # 25: RIGHT_FOOT
            [0, 0, 1.4],      # 26: HEAD
            [0, 0.05, 1.42],  # 27: NOSE
            [-0.03, 0.04, 1.41], # 28: LEFT_EYE
            [-0.06, 0.02, 1.40], # 29: LEFT_EAR
            [0.03, 0.04, 1.41],  # 30: RIGHT_EYE
            [0.06, 0.02, 1.40],  # 31: RIGHT_EAR
            [-0.1, -0.05, 0.0],  # 32: LEFT_HEEL
            [0.1, -0.05, 0.0]    # 33: RIGHT_HEEL
        ]
        
        # Apply walking motion
        for i, (x, y, z) in enumerate(skeleton_template):
            # Base position with walking offset
            new_x = x + base_x
            new_y = y + base_y
            new_z = z + base_z
            
            # Add walking motion to legs
            if i in [18, 19, 20, 21, 32]:  # Left leg points
                new_x += step_phase
            elif i in [22, 23, 24, 25, 33]:  # Right leg points
                new_x -= step_phase
                
            # Add arm swing
            elif i in [5, 6, 7, 8, 9, 10]:  # Left arm
                new_x += arm_swing
            elif i in [12, 13, 14, 15, 16, 17]:  # Right arm  
                new_x -= arm_swing
                
            # Add some noise for realism
            noise = np.random.normal(0, 0.01, 3)
            keypoint_3d = [new_x + noise[0], new_y + noise[1], new_z + noise[2]]
            keypoints_3d.append(keypoint_3d)
            
            # Project to 2D (simple perspective projection)
            # Assume camera at (0, -2, 1) looking forward
            camera_x = 640 + (new_x * 400)  # Scale to image coordinates
            camera_y = 360 - ((new_z - 1.0) * 400)
            keypoints_2d.append([camera_x, camera_y])
            
        # Create frame data
        frame_data.append({
            "timestamp": session_info["start_time"] + t,
            "frame_number": frame_num,
            "bodies": [{
                "id": 0,
                "confidence": 85.0 + np.random.normal(0, 5),
                "tracking_state": "OBJECT_TRACKING_STATE.OK",
                "keypoints_2d": keypoints_2d,
                "keypoints_3d": keypoints_3d,
                "bounding_box_2d": [[200, 100], [500, 600]]
            }]
        })
    
    # Save sample data
    sample_data = {
        "session_info": session_info,
        "data": frame_data
    }
    
    os.makedirs("collected_data", exist_ok=True)
    filepath = os.path.join("collected_data", filename)
    
    with open(filepath, 'w') as f:
        json.dump(sample_data, f, indent=2)
        
    print(f"Created sample data: {filepath}")
    return filepath

def demo_visualization():
    """Demonstrate the visualization capabilities"""
    print("=== Body Tracking Visualization Demo ===")
    
    # Create sample data
    sample_file = create_sample_data()
    
    try:
        from visualize_data import BodyTrackingVisualizer
        
        # Create visualizer
        visualizer = BodyTrackingVisualizer(sample_file)
        
        print("\nGenerating visualizations...")
        
        # Generate all visualizations
        visualizer.plot_skeleton_3d()
        print("✓ 3D skeleton plot displayed")
        
        visualizer.plot_motion_trails()
        print("✓ Motion trails plot displayed")
        
        visualizer.plot_keypoint_heatmap()
        print("✓ Keypoint heatmap displayed")
        
        print("\nGenerating animation...")
        visualizer.create_animation(fps=5)
        print("✓ Animation displayed")
        
        print("\nGenerating comprehensive report...")
        visualizer.generate_report("demo_output")
        print("✓ Report saved to demo_output/")
        
    except ImportError as e:
        print(f"Missing visualization dependencies: {e}")
        print("Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_visualization()