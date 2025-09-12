# ZED SDK - Body Tracking Data Collection & Visualization

This sample shows how to detect and track human bodies in space, now enhanced with comprehensive data collection and visualization capabilities.

## Features

### Basic Body Tracking
- Real-time body detection and tracking
- Display bodies bounding boxes by pressing the `b` key

### Data Collection System
- **Real-time data collection** with `body_tracking_data_collector.py`
- **Live viewing** during data collection using OpenGL or CV2 viewers
- **Session management** with automatic timestamping
- **Multiple output formats**: JSON (structured) and CSV (analysis-ready)
- **Keyboard controls** during collection:
  - `r`: Start/stop recording
  - `s`: Save current session
  - `q`: Quit application

### Visualization System
- **3D skeleton visualization** with proper human pose orientation
- **Motion trails** showing movement patterns over time
- **Interactive viewer** with multiple projection views
- **Animated sequences** for motion analysis
- **Keypoint heatmaps** for activity analysis
- **Comprehensive reports** with automatically generated plots

### Coordinate System
- **Standard orientation**: Y-axis represents up/down (vertical)
- **X-axis**: Left/right movement
- **Z-axis**: Forward/back movement
- **Data transformation**: Automatically converts from ZED camera coordinates to standard visualization coordinates

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)

## Dependencies
```
python3 -m pip install pyopengl
```
 
## Usage Examples

### Basic Body Tracking
To run the original body tracking program:
```bash
python body_tracking.py
```

### Data Collection
To collect body tracking data with real-time viewing:
```bash
python body_tracking_data_collector.py
```
- Press `r` to start/stop recording
- Press `s` to save the current session
- Press `q` to quit

### Data Visualization
To visualize collected data:
```bash
python examples.py
```
This will demonstrate both data collection and comprehensive visualization.

### Advanced Usage
For custom analysis workflows, see the example integration in `examples.py`.

## File Structure
- `body_tracking.py`: Original ZED body tracking sample
- `body_tracking_data_collector.py`: Enhanced data collection system
- `visualize_data.py`: Comprehensive visualization toolkit
- `examples.py`: Integration examples and workflows
- `demo_visualization.py`: Demo data generator and testing
- `collected_data/`: Directory for saved sessions
- `cv_viewer/`: OpenCV-based viewer components
- `ogl_viewer/`: OpenGL-based viewer components

## Run the program
*NOTE: The ZED v1 is not compatible with this module*

To run the program, use the following command in your terminal : 
```bash
python body_tracking.py
```
If you wish to run the program from an input_svo_file, or an IP adress, or specify a resolution run : 

```bash
python body_tracking.py --input_svo_file <input_svo_file> --ip_address <ip_address> --resolution <resolution> 
```
Arguments: 
  - --input_svo_file A path to an existing .svo file, that will be playbacked. If this parameter and ip_adress are not specified, the soft will use the camera wired as default.  
  - --ip_address IP Address, in format a.b.c.d:port or a.b.c.d. If specified, the soft will try to connect to the IP.
  - --resolution Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
## Features
 - Display bodies bounding boxes by pressing the `b` key.

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/