########################################################################
#
# Body Tracking Data Collector
# Based on ZED Camera Body Tracking sample
#
# This module provides real-time body tracking with data collection
# capabilities, allowing users to view motion during data collection
# and save body tracking data for later analysis.
#
########################################################################

import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import time


class BodyTrackingDataCollector:
    """
    A data collector for body tracking that allows real-time viewing
    and data collection from ZED camera body tracking.
    """

    def __init__(self, opt):
        self.opt = opt
        self.zed = None
        self.viewer = None
        self.bodies = sl.Bodies()
        self.image = sl.Mat()

        # Data collection properties
        self.is_recording = False
        self.collected_data = []
        self.session_start_time = None
        self.frame_count = 0
        self.recorded_frame_count = 0

        # RGB image collection properties
        self.collect_rgb_images = True  # Enable RGB image collection
        self.rgb_fps = 5  # Target FPS for RGB images
        self.rgb_frame_interval = None  # Will be calculated based on camera FPS
        self.last_rgb_frame_time = 0
        self.rgb_image_count = 0

        # Session management
        self.session_name = None
        self.output_dir = "collected_data"
        self.images_dir = None  # Will be set when session starts

        # Display properties
        self.display_resolution = None
        self.image_scale = None

        # Camera and tracking parameters
        self.body_param = None
        self.body_runtime_param = None

    def initialize_camera(self):
        """Initialize ZED camera with tracking parameters"""
        print("Initializing ZED camera...")

        # Create camera object
        self.zed = sl.Camera()

        # Create initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Parse command line arguments
        self._parse_args(init_params)

        # Open camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera failed to open: {err}")
            return False

        # Enable positional tracking
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        # Configure body tracking
        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = False
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        self.body_param.body_format = sl.BODY_FORMAT.BODY_34

        # Enable body tracking
        self.zed.enable_body_tracking(self.body_param)

        # Runtime parameters
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 40

        # Calculate RGB frame interval based on camera FPS
        camera_info = self.zed.get_camera_information()
        camera_fps = camera_info.camera_configuration.fps
        self.rgb_frame_interval = 1.0 / self.rgb_fps

        print(
            f"Camera FPS: {camera_fps}, RGB collection interval: {self.rgb_frame_interval:.3f}s"
        )

        return True

    def initialize_viewer(self):
        """Initialize the OpenGL viewer"""
        camera_info = self.zed.get_camera_information()

        # Setup display resolution and scaling
        self.display_resolution = sl.Resolution(
            min(camera_info.camera_configuration.resolution.width, 1280),
            min(camera_info.camera_configuration.resolution.height, 720),
        )
        self.image_scale = [
            self.display_resolution.width
            / camera_info.camera_configuration.resolution.width,
            self.display_resolution.height
            / camera_info.camera_configuration.resolution.height,
        ]

        # Create OpenGL viewer
        self.viewer = gl.GLViewer()
        self.viewer.init(
            camera_info.camera_configuration.calibration_parameters.left_cam,
            self.body_param.enable_tracking,
            self.body_param.body_format,
        )

    def start_session(self, session_name: Optional[str] = None):
        """Start a new data collection session"""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"body_tracking_session_{timestamp}"

        self.session_name = session_name
        self.session_start_time = time.time()
        self.collected_data = []
        self.frame_count = 0
        self.recorded_frame_count = 0
        self.last_rgb_frame_time = 0
        self.rgb_image_count = 0

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)

        # Create images directory for this session
        if self.collect_rgb_images:
            self.images_dir = os.path.join(
                self.output_dir, f"{self.session_name}_images"
            )
            os.makedirs(self.images_dir, exist_ok=True)
            print(f"RGB images will be saved to: {self.images_dir}")

        print(f"Started session: {self.session_name}")
        print(
            f"RGB collection: {'Enabled' if self.collect_rgb_images else 'Disabled'} at {self.rgb_fps} fps"
        )

    def start_recording(self):
        """Start recording body tracking data"""
        if not self.is_recording:
            self.is_recording = True
            print("🔴 Recording started - Press 'r' to stop recording")

    def stop_recording(self):
        """Stop recording body tracking data"""
        if self.is_recording:
            self.is_recording = False
            print("⏹️  Recording stopped")

    def save_session_data(self):
        """Save collected data to files"""
        if not self.collected_data:
            print("No data to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.session_name}_{timestamp}"

        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(json_path, "w") as f:
            session_info = {
                "session_name": self.session_name,
                "start_time": self.session_start_time,
                "total_frames": self.frame_count,
                "recorded_frames": self.recorded_frame_count,
                "body_format": str(self.body_param.body_format),
                "detection_model": str(self.body_param.detection_model),
                "rgb_collection_enabled": self.collect_rgb_images,
                "rgb_fps": self.rgb_fps,
                "rgb_images_collected": self.rgb_image_count,
                "images_directory": (
                    self.images_dir if self.collect_rgb_images else None
                ),
            }

            json.dump(
                {"session_info": session_info, "data": self.collected_data}, f, indent=2
            )

        # Save as CSV (flattened keypoints)
        csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")
        self._save_csv_data(csv_path)

        print(f"💾 Data saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        print(f"  Total frames recorded: {self.recorded_frame_count}")
        if self.collect_rgb_images:
            print(f"  RGB images collected: {self.rgb_image_count}")
            print(f"  Images directory: {self.images_dir}")

    def _save_csv_data(self, csv_path: str):
        """Save data in CSV format with flattened keypoints"""
        if not self.collected_data:
            return

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            header = [
                "timestamp",
                "frame_number",
                "body_id",
                "confidence",
                "rgb_image_path",
            ]

            # Add keypoint columns (assuming BODY_34 format with 34 keypoints)
            for i in range(34):  # BODY_34 has 34 keypoints
                header.extend(
                    [
                        f"kp_{i}_x_2d",
                        f"kp_{i}_y_2d",
                        f"kp_{i}_x_3d",
                        f"kp_{i}_y_3d",
                        f"kp_{i}_z_3d",
                    ]
                )

            writer.writerow(header)

            # Write data
            for frame_data in self.collected_data:
                for body in frame_data["bodies"]:
                    row = [
                        frame_data["timestamp"],
                        frame_data["frame_number"],
                        body["id"],
                        body["confidence"],
                        frame_data.get("rgb_image_path", ""),
                    ]

                    # Add keypoint data
                    for kp_2d, kp_3d in zip(body["keypoints_2d"], body["keypoints_3d"]):
                        row.extend([kp_2d[0], kp_2d[1], kp_3d[0], kp_3d[1], kp_3d[2]])

                    writer.writerow(row)

    def collect_frame_data(self):
        """Collect body tracking data for current frame"""
        if not self.is_recording:
            return

        current_time = time.time()
        frame_data = {
            "timestamp": current_time,
            "frame_number": self.frame_count,
            "bodies": [],
            "rgb_image_path": None,  # Will be set if RGB image is saved
        }

        # Collect RGB image if it's time (based on target FPS)
        if (
            self.collect_rgb_images
            and (current_time - self.last_rgb_frame_time) >= self.rgb_frame_interval
        ):
            rgb_image_path = self._save_rgb_image(current_time)
            if rgb_image_path:
                frame_data["rgb_image_path"] = rgb_image_path
                self.last_rgb_frame_time = current_time
                self.rgb_image_count += 1

        for body in self.bodies.body_list:
            # Import render_object from utils
            from cv_viewer.utils import render_object

            if render_object(body, self.body_param.enable_tracking):
                body_data = {
                    "id": body.id,
                    "confidence": body.confidence,
                    "tracking_state": str(body.tracking_state),
                    "keypoints_2d": [
                        [float(kp[0]), float(kp[1])] for kp in body.keypoint_2d
                    ],
                    "keypoints_3d": [
                        [float(kp[0]), float(kp[1]), float(kp[2])]
                        for kp in body.keypoint
                    ],
                    "bounding_box_2d": [
                        [
                            float(body.bounding_box_2d[0][0]),
                            float(body.bounding_box_2d[0][1]),
                        ],
                        [
                            float(body.bounding_box_2d[1][0]),
                            float(body.bounding_box_2d[1][1]),
                        ],
                    ],
                }
                frame_data["bodies"].append(body_data)

        if frame_data["bodies"]:  # Only save frames with detected bodies
            self.collected_data.append(frame_data)
            self.recorded_frame_count += 1

    def _save_rgb_image(self, timestamp: float) -> Optional[str]:
        """Save RGB image from current frame"""
        try:
            # Get full resolution RGB image
            rgb_image = sl.Mat()
            self.zed.retrieve_image(rgb_image, sl.VIEW.LEFT, sl.MEM.CPU)

            # Convert to OpenCV format
            rgb_image_ocv = rgb_image.get_data()

            # Generate filename with timestamp
            timestamp_str = datetime.fromtimestamp(timestamp).strftime(
                "%Y%m%d_%H%M%S_%f"
            )[:-3]
            filename = f"rgb_{timestamp_str}_frame_{self.rgb_image_count:06d}.jpg"
            image_path = os.path.join(self.images_dir, filename)

            # Save image with high quality JPEG compression
            cv2.imwrite(image_path, rgb_image_ocv, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Return relative path for JSON storage
            return os.path.relpath(image_path, self.output_dir)

        except Exception as e:
            print(f"Error saving RGB image: {e}")
            return None

    def run(self):
        """Main execution loop"""
        print("Body Tracking Data Collector")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Start/Stop recording")
        print("  's' - Save session data")
        print("  'm' - Pause/Resume display")
        print("  'n' - Start new session")
        print("  'i' - Toggle RGB image collection")
        print()
        print(
            f"RGB Image Collection: {'Enabled' if self.collect_rgb_images else 'Disabled'}"
        )
        print(f"Target RGB FPS: {self.rgb_fps}")
        print()

        if not self.initialize_camera():
            return

        self.initialize_viewer()
        self.start_session()

        key_wait = 10

        try:
            while self.viewer.is_available():
                # Grab frame
                if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                    self.frame_count += 1

                    # Retrieve image and bodies
                    self.zed.retrieve_image(
                        self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution
                    )
                    self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)

                    # Collect data if recording
                    self.collect_frame_data()

                    # Update GL view
                    self.viewer.update_view(self.image, self.bodies)

                    # Update CV view with recording status
                    image_left_ocv = self.image.get_data()
                    self._render_enhanced_2d_view(image_left_ocv)

                    cv2.imshow("ZED | Body Tracking Data Collector", image_left_ocv)
                    key = cv2.waitKey(key_wait)

                    if key == 113:  # 'q' - quit
                        print("Exiting...")
                        break
                    elif key == 114:  # 'r' - toggle recording
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            self.start_recording()
                    elif key == 115:  # 's' - save data
                        self.save_session_data()
                    elif key == 109:  # 'm' - pause/resume
                        if key_wait > 0:
                            print("Display paused")
                            key_wait = 0
                        else:
                            print("Display resumed")
                            key_wait = 10
                    elif key == 110:  # 'n' - new session
                        if self.is_recording:
                            self.stop_recording()
                        self.save_session_data()
                        self.start_session()
                    elif key == 105:  # 'i' - toggle RGB collection
                        self.collect_rgb_images = not self.collect_rgb_images
                        print(
                            f"RGB Collection: {'Enabled' if self.collect_rgb_images else 'Disabled'}"
                        )
                        if self.collect_rgb_images and self.images_dir is None:
                            # Create images directory if it doesn't exist
                            self.images_dir = os.path.join(
                                self.output_dir, f"{self.session_name}_images"
                            )
                            os.makedirs(self.images_dir, exist_ok=True)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def _render_enhanced_2d_view(self, image_left_ocv):
        """Enhanced 2D view with data collection info"""
        # Render standard body tracking
        cv_viewer.render_2D(
            image_left_ocv,
            self.image_scale,
            self.bodies.body_list,
            self.body_param.enable_tracking,
            self.body_param.body_format,
        )

        # Add recording status and info overlay
        self._add_info_overlay(image_left_ocv)

    def _add_info_overlay(self, image):
        """Add information overlay to the image"""
        h, w = image.shape[:2]

        # Recording status
        status_color = (0, 0, 255) if self.is_recording else (100, 100, 100)
        status_text = "RECORDING" if self.is_recording else "STANDBY"
        cv2.putText(
            image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )

        # Session info
        info_lines = [
            f"Session: {self.session_name}",
            f"Frame: {self.frame_count}",
            f"Recorded: {self.recorded_frame_count}",
            f"Bodies: {len(self.bodies.body_list)}",
            (
                f"RGB Images: {self.rgb_image_count} ({self.rgb_fps}fps)"
                if self.collect_rgb_images
                else "RGB: Disabled"
            ),
        ]

        for i, line in enumerate(info_lines):
            y_pos = h - 140 + (i * 25)
            cv2.putText(
                image,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
            self.save_session_data()

        if self.viewer:
            self.viewer.exit()

        if self.image:
            self.image.free(sl.MEM.CPU)

        if self.zed:
            self.zed.disable_body_tracking()
            self.zed.disable_positional_tracking()
            self.zed.close()

        cv2.destroyAllWindows()

    def _parse_args(self, init_params):
        """Parse command line arguments (same as original)"""
        if len(self.opt.input_svo_file) > 0 and self.opt.input_svo_file.endswith(
            ".svo"
        ):
            init_params.set_from_svo_file(self.opt.input_svo_file)
            print(f"[Sample] Using SVO File input: {self.opt.input_svo_file}")
        elif len(self.opt.ip_address) > 0:
            ip_str = self.opt.ip_address
            if (
                ip_str.replace(":", "").replace(".", "").isdigit()
                and len(ip_str.split(".")) == 4
                and len(ip_str.split(":")) == 2
            ):
                init_params.set_from_stream(
                    ip_str.split(":")[0], int(ip_str.split(":")[1])
                )
                print(f"[Sample] Using Stream input, IP: {ip_str}")
            elif (
                ip_str.replace(":", "").replace(".", "").isdigit()
                and len(ip_str.split(".")) == 4
            ):
                init_params.set_from_stream(ip_str)
                print(f"[Sample] Using Stream input, IP: {ip_str}")
            else:
                print("Invalid IP format. Using live stream")

        # Resolution settings
        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1200": sl.RESOLUTION.HD1200,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "SVGA": sl.RESOLUTION.SVGA,
            "VGA": sl.RESOLUTION.VGA,
        }

        if self.opt.resolution in resolution_map:
            init_params.camera_resolution = resolution_map[self.opt.resolution]
            print(f"[Sample] Using Camera in resolution {self.opt.resolution}")
        elif len(self.opt.resolution) > 0:
            print("[Sample] No valid resolution entered. Using default")
        else:
            print("[Sample] Using default resolution")


def main():
    parser = argparse.ArgumentParser(description="Body Tracking Data Collector")
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default="",
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Address, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup",
        default="",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for collected data",
        default="collected_data",
    )
    parser.add_argument(
        "--rgb_fps",
        type=int,
        help="Target FPS for RGB image collection (default: 5)",
        default=5,
    )
    parser.add_argument(
        "--no_rgb", action="store_true", help="Disable RGB image collection"
    )

    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print(
            "Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program"
        )
        exit()

    # Create and run data collector
    collector = BodyTrackingDataCollector(opt)
    collector.output_dir = opt.output_dir
    collector.collect_rgb_images = not opt.no_rgb
    collector.rgb_fps = opt.rgb_fps

    print(
        f"RGB Image Collection: {'Disabled' if opt.no_rgb else f'Enabled at {opt.rgb_fps} fps'}"
    )

    collector.run()


if __name__ == "__main__":
    main()
