#!/usr/bin/env python3
"""
Body Tracking Data Visualizer

This module provides comprehensive visualization tools for analyzing
body tracking data collected from the BodyTrackingDataCollector.

Features:
- 2D and 3D skeleton visualization
- Motion trails and trajectories
- Statistical         # XZ projection (top-down view)
        ax2 = fig.add_subplot(2, 2, 2)
        for i, kp_idx in enumerate(keypoint_indices):
            if kp_idx >= keypoints_3d_transformed.shape[1]:
                continue
                
            trail = keypoints_3d_transformed[:, kp_idx, :]
            valid_points = ~np.any(np.isnan(trail), axis=1)
            
            if np.any(valid_points):
                trail_valid = trail[valid_points]
                ax2.plot(trail_valid[:, 0], trail_valid[:, 2], 
                        color=colors[i], label=KEYPOINT_NAMES[kp_idx], linewidth=2, alpha=0.8)
                
        ax2.set_xlabel('X (m) - Left/Right')
        ax2.set_ylabel('Z (m) - Forward/Back')
        ax2.set_title('XZ Projection (Top-Down View)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # YZ projection (side view)
        ax3 = fig.add_subplot(2, 2, 3)
        for i, kp_idx in enumerate(keypoint_indices):
            if kp_idx >= keypoints_3d_transformed.shape[1]:
                continue
                
            trail = keypoints_3d_transformed[:, kp_idx, :]
            valid_points = ~np.any(np.isnan(trail), axis=1)
            
            if np.any(valid_points):
                trail_valid = trail[valid_points]
                ax3.plot(trail_valid[:, 2], trail_valid[:, 1], 
                        color=colors[i], label=KEYPOINT_NAMES[kp_idx], linewidth=2, alpha=0.8)
                
        ax3.set_xlabel('Z (m) - Forward/Back')
        ax3.set_ylabel('Y (m) - Up/Down')
        ax3.set_title('ZY Projection (Side View)')
        ax3.grid(True, alpha=0.3)teractive playback
- Keypoint heatmaps
- Multi-person comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse
import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import seaborn as sns

# ZED Body34 keypoint connections for skeleton visualization
BODY_34_BONES = [
    # Torso
    (0, 1), (1, 2), (2, 3),  # Pelvis -> Naval -> Chest -> Neck
    
    # Left arm
    (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),  # Neck -> Left arm chain
    
    # Right arm  
    (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),  # Neck -> Right arm chain
    
    # Left leg
    (0, 18), (18, 19), (19, 20), (20, 21), (20, 32),  # Pelvis -> Left leg chain
    
    # Right leg
    (0, 22), (22, 23), (23, 24), (24, 25), (24, 33),  # Pelvis -> Right leg chain
    
    # Head
    (3, 26), (26, 27), (27, 28), (27, 30), (28, 29), (30, 31)  # Neck -> Head features
]

KEYPOINT_NAMES = [
    "PELVIS", "NAVAL_SPINE", "CHEST_SPINE", "NECK", "LEFT_CLAVICLE", "LEFT_SHOULDER",
    "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HAND", "LEFT_HANDTIP", "LEFT_THUMB", "RIGHT_CLAVICLE",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HAND", "RIGHT_HANDTIP", "RIGHT_THUMB",
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT", "RIGHT_HIP", "RIGHT_KNEE",
    "RIGHT_ANKLE", "RIGHT_FOOT", "HEAD", "NOSE", "LEFT_EYE", "LEFT_EAR", "RIGHT_EYE", "RIGHT_EAR",
    "LEFT_HEEL", "RIGHT_HEEL"
]

class BodyTrackingVisualizer:
    """Main class for visualizing body tracking data"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = None
        self.session_info = None
        self.frame_data = None
        self.bodies_data = {}  # Organized by body ID
        
        self.load_data()
        self.process_data()
        
    def load_data(self):
        """Load data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                raw_data = json.load(f)
                
            self.session_info = raw_data.get('session_info', {})
            self.frame_data = raw_data.get('data', [])
            
            print(f"Loaded session: {self.session_info.get('session_name', 'Unknown')}")
            print(f"Frames with data: {len(self.frame_data)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def process_data(self):
        """Process and organize data by body ID"""
        for frame in self.frame_data:
            frame_num = frame.get('frame_number', 0)
            timestamp = frame.get('timestamp', 0)
            
            for body in frame.get('bodies', []):
                body_id = body.get('id', -1)
                
                if body_id not in self.bodies_data:
                    self.bodies_data[body_id] = {
                        'frames': [],
                        'timestamps': [],
                        'keypoints_2d': [],
                        'keypoints_3d': [],
                        'confidences': []
                    }
                    
                self.bodies_data[body_id]['frames'].append(frame_num)
                self.bodies_data[body_id]['timestamps'].append(timestamp)
                self.bodies_data[body_id]['keypoints_2d'].append(body.get('keypoints_2d', []))
                self.bodies_data[body_id]['keypoints_3d'].append(body.get('keypoints_3d', []))
                self.bodies_data[body_id]['confidences'].append(body.get('confidence', 0))
                
        # Convert to numpy arrays for easier processing
        for body_id in self.bodies_data:
            for key in ['keypoints_2d', 'keypoints_3d']:
                if self.bodies_data[body_id][key]:
                    self.bodies_data[body_id][key] = np.array(self.bodies_data[body_id][key])
                    
        print(f"Processed data for {len(self.bodies_data)} bodies")
        
    def plot_skeleton_3d(self, body_id: int = None, frame_range: Tuple[int, int] = None, 
                        save_path: str = None):
        """Create 3D skeleton visualization"""
        if body_id is None:
            body_id = list(self.bodies_data.keys())[0]
            
        if body_id not in self.bodies_data:
            print(f"Body ID {body_id} not found")
            return
            
        body_data = self.bodies_data[body_id]
        keypoints_3d = body_data['keypoints_3d']
        
        if len(keypoints_3d) == 0:
            print("No 3D keypoint data available")
            return
            
        # Determine frame range
        if frame_range is None:
            start_idx, end_idx = 0, min(10, len(keypoints_3d))  # Show first 10 frames
        else:
            start_idx, end_idx = frame_range
            
        fig = plt.figure(figsize=(15, 5))
        
        # Plot multiple frames
        n_frames = min(3, end_idx - start_idx)
        frame_indices = np.linspace(start_idx, end_idx-1, n_frames, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(1, n_frames, i+1, projection='3d')
            
            keypoints = keypoints_3d[frame_idx]
            if len(keypoints) == 0:
                continue
                
            # Transform coordinates to correct orientation
            # ZED: X=right, Y=up, Z=forward (into scene)
            # But data shows Y negative (below camera), Z negative (in front)
            # Transform to standard visualization: X=left/right, Y=up/down, Z=forward/back
            keypoints_transformed = keypoints.copy()
            # Swap Y and Z, then flip to get correct orientation
            temp_y = keypoints_transformed[:, 1].copy()  # Original Y (up/down)
            temp_z = keypoints_transformed[:, 2].copy()  # Original Z (forward/back)
            
            keypoints_transformed[:, 1] = -temp_z  # Z becomes Y (up/down), flipped
            keypoints_transformed[:, 2] = temp_y  # Y becomes Z (forward/back), flipped
            
            # Plot keypoints
            xs, ys, zs = keypoints_transformed[:, 0], keypoints_transformed[:, 1], keypoints_transformed[:, 2]
            ax.scatter(xs, ys, zs, c='red', s=50, alpha=0.8)
            
            # Plot skeleton connections
            for bone in BODY_34_BONES:
                if bone[0] < len(keypoints_transformed) and bone[1] < len(keypoints_transformed):
                    x_vals = [keypoints_transformed[bone[0]][0], keypoints_transformed[bone[1]][0]]
                    y_vals = [keypoints_transformed[bone[0]][1], keypoints_transformed[bone[1]][1]]
                    z_vals = [keypoints_transformed[bone[0]][2], keypoints_transformed[bone[1]][2]]
                    ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2, alpha=0.7)
                    
            ax.set_xlabel('X (m) - Left/Right')
            ax.set_ylabel('Y (m) - Up/Down')
            ax.set_zlabel('Z (m) - Forward/Back')
            ax.set_title(f'Frame {body_data["frames"][frame_idx]}')
            
            # Set proper bounds based on actual data range
            # Calculate bounds from all keypoints
            all_x = keypoints_transformed[:, 0]
            all_y = keypoints_transformed[:, 1] 
            all_z = keypoints_transformed[:, 2]
            
            valid_points = ~(np.isnan(all_x) | np.isnan(all_y) | np.isnan(all_z))
            if np.any(valid_points):
                x_valid = all_x[valid_points]
                y_valid = all_y[valid_points]
                z_valid = all_z[valid_points]
                
                # Set bounds with some padding
                x_center, x_range = np.mean(x_valid), np.ptp(x_valid)
                y_center, y_range = np.mean(y_valid), np.ptp(y_valid)
                z_center, z_range = np.mean(z_valid), np.ptp(z_valid)
                
                # Ensure minimum range for visibility
                x_range = max(x_range, 0.5)
                y_range = max(y_range, 1.0)  # Humans are tall
                z_range = max(z_range, 0.5)
                
                padding = 0.2
                ax.set_xlim([x_center - x_range/2 - padding, x_center + x_range/2 + padding])
                ax.set_ylim([y_center - y_range/2 - padding, y_center + y_range/2 + padding])
                ax.set_zlim([z_center - z_range/2 - padding, z_center + z_range/2 + padding])
            else:
                # Fallback bounds
                ax.set_xlim([-1, 1])
                ax.set_ylim([0, 2])
                ax.set_zlim([0, 2])
            
            # Set viewing angle to show person standing upright
            ax.view_init(elev=0, azim=-90)  # Side view, looking from the side
            
        plt.suptitle(f'3D Skeleton - Body {body_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D skeleton plot to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def plot_motion_trails(self, keypoint_indices: List[int] = None, body_id: int = None,
                          save_path: str = None):
        """Plot motion trails for specific keypoints"""
        if body_id is None:
            body_id = list(self.bodies_data.keys())[0]
            
        if keypoint_indices is None:
            # Default to key joints
            keypoint_indices = [0, 5, 12, 18, 22, 26]  # Pelvis, shoulders, hips, head
            
        body_data = self.bodies_data[body_id]
        keypoints_3d = body_data['keypoints_3d']
        
        if len(keypoints_3d) == 0:
            print("No 3D keypoint data available")
            return
            
        # Transform coordinates for proper orientation
        # Swap Y and Z coordinates, then flip to get standard Y-up visualization
        keypoints_3d_transformed = keypoints_3d.copy()
        temp_y = keypoints_3d_transformed[:, :, 1].copy()  # Original Y
        temp_z = keypoints_3d_transformed[:, :, 2].copy()  # Original Z
        
        keypoints_3d_transformed[:, :, 1] = -temp_z  # Z becomes Y (up/down), flipped
        keypoints_3d_transformed[:, :, 2] = -temp_y  # Y becomes Z (forward/back), flipped
            
        fig = plt.figure(figsize=(15, 10))
        
        # 3D motion trails
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(keypoint_indices)))
        
        for i, kp_idx in enumerate(keypoint_indices):
            if kp_idx >= keypoints_3d_transformed.shape[1]:
                continue
                
            trail = keypoints_3d_transformed[:, kp_idx, :]
            valid_points = ~np.any(np.isnan(trail), axis=1)
            
            if np.any(valid_points):
                trail_valid = trail[valid_points]
                ax1.plot(trail_valid[:, 0], trail_valid[:, 1], trail_valid[:, 2], 
                        color=colors[i], label=KEYPOINT_NAMES[kp_idx], linewidth=2, alpha=0.8)
                
        ax1.set_xlabel('X (m) - Left/Right')
        ax1.set_ylabel('Y (m) - Up/Down')
        ax1.set_zlabel('Z (m) - Forward/Back')
        ax1.set_title('3D Motion Trails')
        ax1.legend()
        
        # Set proper viewing angle
        ax1.view_init(elev=10, azim=-60)
        
        # XY projection (top-down view)
        ax2 = fig.add_subplot(2, 2, 2)
        for i, kp_idx in enumerate(keypoint_indices):
            if kp_idx >= keypoints_3d_transformed.shape[1]:
                continue
                
            trail = keypoints_3d_transformed[:, kp_idx, :]
            valid_points = ~np.any(np.isnan(trail), axis=1)
            
            if np.any(valid_points):
                trail_valid = trail[valid_points]
                ax2.plot(trail_valid[:, 0], trail_valid[:, 2], 
                        color=colors[i], label=KEYPOINT_NAMES[kp_idx], linewidth=2, alpha=0.8)
                
        ax2.set_xlabel('X (m) - Left/Right')
        ax2.set_ylabel('Z (m) - Forward/Back')
        ax2.set_title('XZ Projection (Top-Down View)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # XZ projection (side view)
        ax3 = fig.add_subplot(2, 2, 3)
        for i, kp_idx in enumerate(keypoint_indices):
            if kp_idx >= keypoints_3d_transformed.shape[1]:
                continue
                
            trail = keypoints_3d_transformed[:, kp_idx, :]
            valid_points = ~np.any(np.isnan(trail), axis=1)
            
            if np.any(valid_points):
                trail_valid = trail[valid_points]
                ax3.plot(trail_valid[:, 2], trail_valid[:, 1], 
                        color=colors[i], label=KEYPOINT_NAMES[kp_idx], linewidth=2, alpha=0.8)
                
        ax3.set_xlabel('Z (m) - Forward/Back')
        ax3.set_ylabel('Y (m) - Up/Down')
        ax3.set_title('ZY Projection (Side View)')
        ax3.grid(True, alpha=0.3)
        
        # Velocity analysis
        ax4 = fig.add_subplot(2, 2, 4)
        
        for i, kp_idx in enumerate(keypoint_indices):
            if kp_idx >= keypoints_3d_transformed.shape[1]:
                continue
                
            trail = keypoints_3d_transformed[:, kp_idx, :]
            valid_points = ~np.any(np.isnan(trail), axis=1)
            
            if np.any(valid_points) and np.sum(valid_points) > 1:
                trail_valid = trail[valid_points]
                timestamps = np.array(body_data['timestamps'])[valid_points]
                
                # Calculate velocity
                velocities = []
                for j in range(1, len(trail_valid)):
                    dt = timestamps[j] - timestamps[j-1]
                    if dt > 0:
                        dp = np.linalg.norm(trail_valid[j] - trail_valid[j-1])
                        velocities.append(dp / dt)
                    else:
                        velocities.append(0)
                        
                if velocities:
                    ax4.plot(timestamps[1:], velocities, color=colors[i], 
                            label=KEYPOINT_NAMES[kp_idx], linewidth=2, alpha=0.8)
                    
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Keypoint Velocities')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle(f'Motion Analysis - Body {body_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved motion trails plot to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def create_animation(self, body_id: int = None, save_path: str = None, 
                        fps: int = 10, view_mode: str = '3d'):
        """Create animated visualization of the motion"""
        if body_id is None:
            body_id = list(self.bodies_data.keys())[0]
            
        body_data = self.bodies_data[body_id]
        keypoints_3d = body_data['keypoints_3d']
        
        if len(keypoints_3d) == 0:
            print("No 3D keypoint data available")
            return
            
        # Transform coordinates for proper orientation
        # Swap Y and Z coordinates, then flip to get standard Y-up visualization
        keypoints_3d_transformed = keypoints_3d.copy()
        temp_y = keypoints_3d_transformed[:, :, 1].copy()  # Original Y
        temp_z = keypoints_3d_transformed[:, :, 2].copy()  # Original Z
        
        keypoints_3d_transformed[:, :, 1] = -temp_z  # Z becomes Y (up/down), flipped
        keypoints_3d_transformed[:, :, 2] = temp_y  # Y becomes Z (forward/back), flipped
            
        # Limit frames for reasonable animation size
        max_frames = min(100, len(keypoints_3d_transformed))
        keypoints_3d_transformed = keypoints_3d_transformed[:max_frames]
        
        if view_mode == '3d':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        # Calculate global bounds for consistent scaling
        all_keypoints = keypoints_3d_transformed.reshape(-1, 3)
        valid_points = ~np.any(np.isnan(all_keypoints), axis=1)
        if np.any(valid_points):
            valid_kp = all_keypoints[valid_points]
            x_bounds = [np.min(valid_kp[:, 0]), np.max(valid_kp[:, 0])]
            y_bounds = [np.min(valid_kp[:, 1]), np.max(valid_kp[:, 1])]
            z_bounds = [np.min(valid_kp[:, 2]), np.max(valid_kp[:, 2])]
            
            # Add padding
            x_padding = (x_bounds[1] - x_bounds[0]) * 0.1
            y_padding = (y_bounds[1] - y_bounds[0]) * 0.1
            z_padding = (z_bounds[1] - z_bounds[0]) * 0.1
            
            x_bounds = [x_bounds[0] - x_padding, x_bounds[1] + x_padding]
            y_bounds = [y_bounds[0] - y_padding, y_bounds[1] + y_padding]
            z_bounds = [z_bounds[0] - z_padding, z_bounds[1] + z_padding]
        else:
            x_bounds = [-1, 1]
            y_bounds = [0, 2]
            z_bounds = [0, 2]
            
        def animate(frame_idx):
            ax.clear()
            
            if frame_idx >= len(keypoints_3d_transformed):
                return
                
            keypoints = keypoints_3d_transformed[frame_idx]
            
            if len(keypoints) == 0:
                return
                
            if view_mode == '3d':
                x_idx = 0
                y_idx = 1
                z_idx = 2
                # Plot keypoints
                xs, ys, zs = keypoints[:, x_idx], keypoints[:, y_idx], keypoints[:, z_idx]
                ax.scatter(xs, ys, zs, c='red', s=50, alpha=0.8)
                
                # Plot skeleton
                for bone in BODY_34_BONES:
                    if bone[0] < len(keypoints) and bone[1] < len(keypoints):
                        x_vals = [keypoints[bone[0]][x_idx], keypoints[bone[1]][x_idx]]
                        y_vals = [keypoints[bone[0]][y_idx], keypoints[bone[1]][y_idx]]
                        z_vals = [keypoints[bone[0]][z_idx], keypoints[bone[1]][z_idx]]
                        ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2, alpha=0.7)
                        
                ax.set_xlabel('Left/Right')
                ax.set_ylabel('Forward/Back')
                ax.set_zlabel('Up/Down')
                ax.set_xlim(x_bounds)
                ax.set_ylim(y_bounds)
                ax.set_zlim(z_bounds)
                
                # Set good viewing angle
                ax.view_init(elev=10, azim=-60)
                
            else:  # 2D projection (side view)
                # Plot keypoints (ZY projection - side view)
                zs, ys = keypoints[:, 2], keypoints[:, 1]
                ax.scatter(zs, ys, c='red', s=50, alpha=0.8)
                
                # Plot skeleton
                for bone in BODY_34_BONES:
                    if bone[0] < len(keypoints) and bone[1] < len(keypoints):
                        z_vals = [keypoints[bone[0]][2], keypoints[bone[1]][2]]
                        y_vals = [keypoints[bone[0]][1], keypoints[bone[1]][1]]
                        ax.plot(z_vals, y_vals, 'b-', linewidth=2, alpha=0.7)
                        
                ax.set_xlabel('Z (m) - Forward/Back')
                ax.set_ylabel('Y (m) - Up/Down')
                ax.set_xlim(z_bounds)
                ax.set_ylim(y_bounds)
                ax.grid(True, alpha=0.3)
                
            frame_num = body_data['frames'][min(frame_idx, len(body_data['frames'])-1)]
            ax.set_title(f'Body {body_id} - Frame {frame_num}')
            
        anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                     interval=1000//fps, blit=False)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            else:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Saved animation to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def plot_keypoint_heatmap(self, body_id: int = None, save_path: str = None):
        """Create heatmap showing keypoint detection reliability"""
        if body_id is None:
            body_id = list(self.bodies_data.keys())[0]
            
        body_data = self.bodies_data[body_id]
        keypoints_3d = body_data['keypoints_3d']
        
        if len(keypoints_3d) == 0:
            print("No 3D keypoint data available")
            return
            
        # Calculate detection rates for each keypoint
        detection_rates = []
        mean_confidences = []
        
        for kp_idx in range(34):  # BODY_34 format
            if kp_idx < keypoints_3d.shape[2]:
                valid_detections = ~np.any(np.isnan(keypoints_3d[:, kp_idx, :]), axis=1)
                detection_rate = np.mean(valid_detections) * 100
            else:
                detection_rate = 0
                
            detection_rates.append(detection_rate)
            
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Detection rate heatmap - arrange in a reasonable grid
        # Use 7x5 grid for 34 keypoints (35 slots, last one empty)
        detection_matrix = np.zeros((7, 5))
        for i, rate in enumerate(detection_rates):
            if i < 35:  # 7*5 = 35
                row, col = divmod(i, 5)
                detection_matrix[row, col] = rate
                
        im1 = ax1.imshow(detection_matrix, cmap='RdYlGn', vmin=0, vmax=100)
        ax1.set_title('Keypoint Detection Rates (%)')
        
        # Add text annotations
        for i in range(7):
            for j in range(5):
                idx = i * 5 + j
                if idx < len(detection_rates):
                    text = ax1.text(j, i, f'{detection_rates[idx]:.1f}%',
                                   ha="center", va="center", color="black", fontsize=8)
                    
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Bar chart of detection rates
        y_pos = np.arange(len(KEYPOINT_NAMES))
        bars = ax2.barh(y_pos, detection_rates, color=plt.cm.RdYlGn(np.array(detection_rates)/100))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(KEYPOINT_NAMES, fontsize=8)
        ax2.set_xlabel('Detection Rate (%)')
        ax2.set_title('Keypoint Detection Reliability')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, detection_rates)):
            ax2.text(rate + 1, bar.get_y() + bar.get_height()/2, 
                    f'{rate:.1f}%', va='center', fontsize=8)
                    
        plt.suptitle(f'Keypoint Analysis - Body {body_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved keypoint heatmap to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def generate_report(self, output_dir: str = "visualization_output"):
        """Generate a comprehensive visualization report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = self.session_info.get('session_name', 'unknown_session')
        
        print(f"Generating visualization report for {session_name}...")
        
        for body_id in self.bodies_data.keys():
            body_dir = os.path.join(output_dir, f"{session_name}_body_{body_id}")
            os.makedirs(body_dir, exist_ok=True)
            
            # 3D skeleton visualization
            skeleton_path = os.path.join(body_dir, "3d_skeleton.png")
            self.plot_skeleton_3d(body_id=body_id, save_path=skeleton_path)
            
            # Motion trails
            trails_path = os.path.join(body_dir, "motion_trails.png")
            self.plot_motion_trails(body_id=body_id, save_path=trails_path)
            
            # Keypoint heatmap
            heatmap_path = os.path.join(body_dir, "keypoint_heatmap.png")
            self.plot_keypoint_heatmap(body_id=body_id, save_path=heatmap_path)
            
            # Animation
            animation_path = os.path.join(body_dir, "motion_animation.gif")
            self.create_animation(body_id=body_id, save_path=animation_path, fps=5)
            
        print(f"Visualization report saved to: {output_dir}")
        
    def interactive_viewer(self):
        """Launch interactive matplotlib viewer"""
        plt.ion()  # Interactive mode
        
        body_ids = list(self.bodies_data.keys())
        current_body = 0
        
        fig = plt.figure(figsize=(15, 10))
        
        def update_display():
            body_id = body_ids[current_body] if body_ids else 0
            
            # Clear the entire figure
            fig.clear()
                
            if body_id not in self.bodies_data:
                plt.suptitle("No data available")
                plt.draw()
                plt.pause(0.1)
                return
                
            body_data = self.bodies_data[body_id]
            keypoints_3d = body_data['keypoints_3d']
            
            if len(keypoints_3d) == 0:
                plt.suptitle(f"No 3D keypoint data for Body {body_id}")
                plt.draw()
                plt.pause(0.1)
                return
                
            # Create 3D skeleton view
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            if len(keypoints_3d) > 0:
                keypoints = keypoints_3d[0].copy()
                if len(keypoints) > 0:
                    # Transform coordinates - swap Y and Z, then flip
                    temp_y = keypoints[:, 1].copy()
                    temp_z = keypoints[:, 2].copy()
                    keypoints[:, 1] = -temp_z  # Z becomes Y (up/down), flipped
                    keypoints[:, 2] = -temp_y  # Y becomes Z (forward/back), flipped
                    
                    xs, ys, zs = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
                    ax1.scatter(xs, ys, zs, c='red', s=50)
                    
                    for bone in BODY_34_BONES:
                        if bone[0] < len(keypoints) and bone[1] < len(keypoints):
                            x_vals = [keypoints[bone[0]][0], keypoints[bone[1]][0]]
                            y_vals = [keypoints[bone[0]][1], keypoints[bone[1]][1]]
                            z_vals = [keypoints[bone[0]][2], keypoints[bone[1]][2]]
                            ax1.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2)
                            
                    ax1.set_xlabel('X (m) - Left/Right')
                    ax1.set_ylabel('Y (m) - Up/Down')
                    ax1.set_zlabel('Z (m) - Forward/Back')
                    ax1.set_title(f'3D Skeleton - Frame 0')
                    
                    # Set proper bounds
                    valid_points = ~(np.isnan(xs) | np.isnan(ys) | np.isnan(zs))
                    if np.any(valid_points):
                        xs_valid, ys_valid, zs_valid = xs[valid_points], ys[valid_points], zs[valid_points]
                        x_center, x_range = np.mean(xs_valid), np.ptp(xs_valid)
                        y_center, y_range = np.mean(ys_valid), np.ptp(ys_valid)
                        z_center, z_range = np.mean(zs_valid), np.ptp(zs_valid)
                        
                        x_range = max(x_range, 0.5)
                        y_range = max(y_range, 1.0)
                        z_range = max(z_range, 0.5)
                        
                        padding = 0.2
                        ax1.set_xlim([x_center - x_range/2 - padding, x_center + x_range/2 + padding])
                        ax1.set_ylim([y_center - y_range/2 - padding, y_center + y_range/2 + padding])
                        ax1.set_zlim([z_center - z_range/2 - padding, z_center + z_range/2 + padding])
                        
                    ax1.view_init(elev=10, azim=-60)
            
            # Create motion trail view (top-down)
            ax2 = fig.add_subplot(2, 2, 2)
            if len(keypoints_3d) > 1:
                # Transform coordinates for trails
                keypoints_3d_transformed = keypoints_3d.copy()
                temp_y = keypoints_3d_transformed[:, :, 1].copy()
                temp_z = keypoints_3d_transformed[:, :, 2].copy()
                keypoints_3d_transformed[:, :, 1] = -temp_z  # Z becomes Y (up/down)
                keypoints_3d_transformed[:, :, 2] = -temp_y  # Y becomes Z (forward/back)
                
                # Show trajectory of head (keypoint 26) and pelvis (keypoint 0)
                head_trail = keypoints_3d_transformed[:, 26, :] if keypoints_3d_transformed.shape[1] > 26 else None
                pelvis_trail = keypoints_3d_transformed[:, 0, :] if keypoints_3d_transformed.shape[1] > 0 else None
                
                if head_trail is not None:
                    valid_head = ~np.any(np.isnan(head_trail), axis=1)
                    if np.any(valid_head):
                        trail_valid = head_trail[valid_head]
                        ax2.plot(trail_valid[:, 0], trail_valid[:, 2], 'r-', label='Head', linewidth=2)
                
                if pelvis_trail is not None:
                    valid_pelvis = ~np.any(np.isnan(pelvis_trail), axis=1)
                    if np.any(valid_pelvis):
                        trail_valid = pelvis_trail[valid_pelvis]
                        ax2.plot(trail_valid[:, 0], trail_valid[:, 2], 'b-', label='Pelvis', linewidth=2)
                        
                ax2.set_xlabel('X (m) - Left/Right')
                ax2.set_ylabel('Z (m) - Forward/Back')
                ax2.set_title('Motion Trails (Top-Down View)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Create confidence plot
            ax3 = fig.add_subplot(2, 2, 3)
            if body_data['confidences']:
                frames = body_data['frames']
                confidences = body_data['confidences']
                ax3.plot(frames, confidences, 'g-', linewidth=2)
                ax3.set_xlabel('Frame Number')
                ax3.set_ylabel('Detection Confidence')
                ax3.set_title('Detection Confidence Over Time')
                ax3.grid(True, alpha=0.3)
            
            # Create session info
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            info_text = f"""
            Body ID: {body_id}
            Total Frames: {len(keypoints_3d)}
            Frame Range: {min(body_data['frames'])} - {max(body_data['frames'])}
            Avg Confidence: {np.mean(body_data['confidences']):.1f}%
            Session: {self.session_info.get('session_name', 'Unknown')}
            """
            ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
            ax4.set_title('Session Information')
                    
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
        update_display()
        
        print("Interactive viewer controls:")
        print("  'n' - Next body")
        print("  'p' - Previous body") 
        print("  'q' - Quit")
        
        while True:
            try:
                key = input("Command (n/p/q): ").lower()
                if key == 'q':
                    break
                elif key == 'n' and len(body_ids) > 0:
                    current_body = (current_body + 1) % len(body_ids)
                    update_display()
                elif key == 'p' and len(body_ids) > 0:
                    current_body = (current_body - 1) % len(body_ids)
                    update_display()
            except KeyboardInterrupt:
                break
                
        plt.ioff()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Body Tracking Data Visualizer")
    parser.add_argument('data_file', help='JSON file containing body tracking data')
    parser.add_argument('--mode', choices=['skeleton', 'trails', 'animation', 'heatmap', 'report', 'interactive'],
                       default='report', help='Visualization mode')
    parser.add_argument('--body_id', type=int, help='Specific body ID to visualize')
    parser.add_argument('--output', type=str, help='Output file/directory for saving')
    parser.add_argument('--fps', type=int, default=10, help='FPS for animation')
    parser.add_argument('--view_mode', choices=['3d', '2d'], default='3d', help='View mode for animation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Data file not found: {args.data_file}")
        return
        
    try:
        visualizer = BodyTrackingVisualizer(args.data_file)
        
        if args.mode == 'skeleton':
            visualizer.plot_skeleton_3d(body_id=args.body_id, save_path=args.output)
        elif args.mode == 'trails':
            visualizer.plot_motion_trails(body_id=args.body_id, save_path=args.output)
        elif args.mode == 'animation':
            visualizer.create_animation(body_id=args.body_id, save_path=args.output, 
                                      fps=args.fps, view_mode=args.view_mode)
        elif args.mode == 'heatmap':
            visualizer.plot_keypoint_heatmap(body_id=args.body_id, save_path=args.output)
        elif args.mode == 'report':
            output_dir = args.output or "visualization_output"
            visualizer.generate_report(output_dir)
        elif args.mode == 'interactive':
            visualizer.interactive_viewer()
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()