# Body Tracking Data Collector & Visualizer

A real-time body tracking data collector based on ZED Camera SDK that allows you to view motion during data collection, save body tracking data for analysis, and create comprehensive visualizations of the collected data.

## Features

- **Real-time Motion Viewing**: Watch body tracking in both 2D and 3D views during data collection
- **Interactive Recording**: Start/stop recording with keyboard controls
- **Multiple Data Formats**: Save data in both JSON and CSV formats
- **Session Management**: Automatic session naming and metadata tracking
- **Comprehensive Visualization**: 3D skeletons, motion trails, animations, and heatmaps
- **Quality Analysis**: Built-in tools to analyze collected data quality
- **Flexible Configuration**: Support for SVO files, live camera, and streaming

## Installation

### Prerequisites

- ZED Camera SDK (ZED SDK 4.0+)
- Python 3.7+
- OpenCV
- NumPy
- pandas (for data analysis)
- matplotlib & seaborn (for visualization)

### Required Python Packages

```bash
pip install opencv-python numpy pandas matplotlib seaborn
```

Make sure you have the ZED SDK installed and the `pyzed` package available.

## Quick Start

### Basic Data Collection

1. **Run the data collector:**
   ```bash
   python body_tracking_data_collector.py
   ```

2. **Controls during collection:**
   - `r` - Start/Stop recording
   - `s` - Save current session data
   - `q` - Quit application
   - `m` - Pause/Resume display
   - `n` - Start new session

3. **View real-time information:**
   - Recording status (red dot when recording)
   - Current session name
   - Frame count and recorded frame count
   - Number of detected bodies

### Data Collection with Options

```bash
# Use specific resolution
python body_tracking_data_collector.py --resolution HD720

# Use SVO file for testing
python body_tracking_data_collector.py --input_svo_file path/to/file.svo

# Specify output directory
python body_tracking_data_collector.py --output_dir my_data

# Use camera stream
python body_tracking_data_collector.py --ip_address 192.168.1.100:30000
```

## Data Analysis & Visualization

### List Collected Data

```bash
python examples.py --mode list
```

### Analyze a Session

```bash
python examples.py --mode analyze --data_file collected_data/session.json
```

### Visualize Data

```bash
# Generate comprehensive visualization report
python examples.py --mode visualize --data_file collected_data/session.json

# Specific visualization types
python examples.py --mode visualize --data_file collected_data/session.json --viz_mode skeleton
python examples.py --mode visualize --data_file collected_data/session.json --viz_mode trails
python examples.py --mode visualize --data_file collected_data/session.json --viz_mode animation
python examples.py --mode visualize --data_file collected_data/session.json --viz_mode heatmap
python examples.py --mode visualize --data_file collected_data/session.json --viz_mode interactive

# Direct visualization tool
python visualize_data.py collected_data/session.json --mode report
python visualize_data.py collected_data/session.json --mode animation --output motion.gif
```

### Demo Visualization

```bash
# Create sample data and test all visualization features
python demo_visualization.py
```

### Convert JSON to CSV

```bash
python examples.py --mode convert --data_file collected_data/session.json
```

## Visualization Features

### 3D Skeleton Visualization
- **Multiple frame views** showing body pose progression
- **Anatomically correct connections** between keypoints
- **Customizable viewpoints** and frame ranges

### Motion Trails & Trajectories
- **3D motion paths** for key body joints
- **XY and XZ projections** for different viewing angles
- **Velocity analysis** showing movement speed over time
- **Multi-keypoint comparison** with color-coded trails

### Interactive Animations
- **Smooth motion playback** with customizable frame rates
- **3D and 2D viewing modes**
- **Export to GIF or MP4** formats
- **Frame-by-frame navigation**

### Keypoint Quality Analysis
- **Detection reliability heatmaps** showing tracking quality
- **Statistical analysis** of keypoint accuracy
- **Missing data visualization** and quality metrics
- **Per-keypoint confidence scores**

### Comprehensive Reports
Automatically generates:
- 3D skeleton visualizations
- Motion trail analyses  
- Quality assessment heatmaps
- Animated motion sequences
- Statistical summaries

## Data Format

### JSON Format

The JSON file contains:

```json
{
  "session_info": {
    "session_name": "body_tracking_session_20231215_143022",
    "start_time": 1702649422.123,
    "total_frames": 1500,
    "recorded_frames": 450,
    "body_format": "BODY_FORMAT.BODY_34",
    "detection_model": "BODY_TRACKING_MODEL.HUMAN_BODY_FAST"
  },
  "data": [
    {
      "timestamp": 1702649422.567,
      "frame_number": 1,
      "bodies": [
        {
          "id": 0,
          "confidence": 85.5,
          "tracking_state": "OBJECT_TRACKING_STATE.OK",
          "keypoints_2d": [[x1, y1], [x2, y2], ...],
          "keypoints_3d": [[x1, y1, z1], [x2, y2, z2], ...],
          "bounding_box_2d": [[x1, y1], [x2, y2]]
        }
      ]
    }
  ]
}
```

### CSV Format

The CSV format flattens the keypoint data for easier analysis:

- `timestamp`: Frame timestamp
- `frame_number`: Sequential frame number
- `body_id`: Unique body identifier
- `confidence`: Detection confidence
- `kp_i_x_2d`, `kp_i_y_2d`: 2D keypoint coordinates
- `kp_i_x_3d`, `kp_i_y_3d`, `kp_i_z_3d`: 3D keypoint coordinates

## Body Keypoint Format

The collector uses ZED's BODY_34 format with 34 keypoints:

0. PELVIS
1. NAVAL_SPINE
2. CHEST_SPINE
3. NECK
4. LEFT_CLAVICLE
5. LEFT_SHOULDER
6. LEFT_ELBOW
7. LEFT_WRIST
8. LEFT_HAND
9. LEFT_HANDTIP
10. LEFT_THUMB
11. RIGHT_CLAVICLE
12. RIGHT_SHOULDER
13. RIGHT_ELBOW
14. RIGHT_WRIST
15. RIGHT_HAND
16. RIGHT_HANDTIP
17. RIGHT_THUMB
18. LEFT_HIP
19. LEFT_KNEE
20. LEFT_ANKLE
21. LEFT_FOOT
22. RIGHT_HIP
23. RIGHT_KNEE
24. RIGHT_ANKLE
25. RIGHT_FOOT
26. HEAD
27. NOSE
28. LEFT_EYE
29. LEFT_EAR
30. RIGHT_EYE
31. RIGHT_EAR
32. LEFT_HEEL
33. RIGHT_HEEL

## Use Cases

### Research and Development
- Motion analysis studies
- Gait analysis
- Exercise form evaluation
- Human-robot interaction research

### Sports and Fitness
- Athletic performance analysis
- Form correction for exercises
- Movement pattern studies

### Healthcare
- Rehabilitation progress tracking
- Mobility assessment
- Physical therapy evaluation

## Tips for Best Results

1. **Camera Setup**:
   - Ensure good lighting conditions
   - Position camera at appropriate height and distance
   - Minimize background clutter

2. **Recording**:
   - Start recording after subjects are in frame
   - Keep recording sessions focused on specific activities
   - Use descriptive session names

3. **Data Quality**:
   - Check the analysis output for keypoint validity
   - Ensure consistent frame rates
   - Monitor confidence scores

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check ZED SDK installation
   - Verify camera connection
   - Try different USB ports

2. **Low detection confidence**:
   - Improve lighting conditions
   - Adjust camera position
   - Reduce background clutter

3. **Performance issues**:
   - Lower camera resolution
   - Close other applications
   - Use SSD for data storage

### Error Messages

- `Camera failed to open`: Check ZED SDK and camera connection
- `No data to save`: No bodies were detected during recording
- `Invalid IP format`: Check network camera IP address format

## Advanced Usage

### Custom Analysis Scripts

You can create custom analysis scripts using the collected data:

```python
import json
import numpy as np

# Load data
with open('collected_data/session.json', 'r') as f:
    data = json.load(f)

# Extract keypoints for analysis
for frame in data['data']:
    for body in frame['bodies']:
        keypoints_3d = np.array(body['keypoints_3d'])
        # Your analysis code here
```

### Integration with Other Tools

The CSV format can be easily imported into:
- MATLAB for signal processing
- R for statistical analysis
- Python pandas for data science
- Excel for basic analysis

## Contributing

Feel free to extend the data collector with additional features:
- Different body formats (BODY_18, BODY_38)
- Real-time data streaming
- Custom visualization options
- Integration with other sensors

## License

Based on ZED SDK samples. See ZED SDK license for details.