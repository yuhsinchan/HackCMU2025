# Troubleshooting Guide for Body Tracking Visualizer

## Common Issues and Solutions

### 1. Interactive Viewer Errors

**Error**: `local variable 'ax1' referenced before assignment`
**Solution**: This has been fixed in the latest version. Update your `visualize_data.py` file.

**Error**: Matplotlib not displaying plots
**Solution**: 
```bash
# For headless systems, use Agg backend
export MPLBACKEND=Agg
python3 visualize_data.py your_data.json --mode report
```

### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'seaborn'`
**Solution**:
```bash
pip install seaborn matplotlib
```

**Error**: `ModuleNotFoundError: No module named 'pandas'`
**Solution**:
```bash
pip install pandas numpy
```

### 3. Data Loading Issues

**Error**: `JSON decode error`
**Solution**: Check if your JSON file is valid:
```bash
python3 -c "import json; json.load(open('your_file.json'))"
```

**Error**: `No 3D keypoint data available`
**Solution**: Ensure your data collection included 3D tracking. Check if body_param.enable_tracking was True during collection.

### 4. Animation Issues

**Error**: Animation not saving or playing
**Solution**:
```bash
# Install ffmpeg for MP4 support
sudo apt-get install ffmpeg

# Use GIF format instead
python3 visualize_data.py your_data.json --mode animation --output motion.gif
```

### 5. Performance Issues

**Issue**: Slow visualization with large datasets
**Solutions**:
- Limit frame range: Use only subset of data for visualization
- Reduce animation fps: `--fps 5` instead of default 10
- Use 2D mode: `--view_mode 2d` for animations

### 6. Display Issues

**Issue**: Plots appear too small or crowded
**Solutions**:
- Increase figure size in the code
- Save to files instead of viewing interactively
- Use report mode for best layout

### 7. Data Quality Issues

**Issue**: Keypoints appear invalid or jumping
**Solutions**:
- Check confidence thresholds in data collection
- Use keypoint heatmap to identify problematic joints
- Filter data by confidence scores

## Debug Commands

### Check Data Structure
```bash
python3 -c "
import json
with open('your_data.json') as f:
    data = json.load(f)
print('Session info:', data.get('session_info', {}))
print('Number of frames:', len(data.get('data', [])))
print('First frame keys:', list(data['data'][0].keys()) if data.get('data') else 'None')
"
```

### Validate Keypoint Data
```bash
python3 examples.py --mode analyze --data_file your_data.json
```

### Test Visualization Step by Step
```bash
# Test each mode individually
python3 visualize_data.py your_data.json --mode skeleton
python3 visualize_data.py your_data.json --mode trails  
python3 visualize_data.py your_data.json --mode heatmap
```

## Environment Setup

### For Headless Systems (No Display)
```bash
export DISPLAY=:0  # If using X11 forwarding
# OR
export MPLBACKEND=Agg  # For file output only
```

### For Remote Systems
```bash
# Enable X11 forwarding
ssh -X username@hostname

# OR use VNC/Remote Desktop
```

## Performance Optimization

### For Large Datasets (>1000 frames)
```python
# Modify visualize_data.py to limit frames
max_frames = min(100, len(keypoints_3d))
keypoints_3d = keypoints_3d[:max_frames]
```

### Memory Issues
- Process one body at a time
- Use CSV format for very large datasets
- Clear matplotlib figures after each plot

## Getting Help

1. Check error messages carefully
2. Try with sample data first: `python3 demo_visualization.py`
3. Verify dependencies: `pip list | grep -E "(matplotlib|seaborn|numpy|pandas)"`
4. Test with minimal example:
   ```bash
   python3 -c "
   import matplotlib.pyplot as plt
   import numpy as np
   plt.plot([1,2,3], [1,4,2])
   plt.savefig('test.png')
   print('Basic matplotlib test passed')
   "
   ```

## Version Compatibility

- Python: 3.7+
- matplotlib: 3.3.0+
- numpy: 1.20.0+
- pandas: 1.3.0+
- seaborn: 0.11.0+

## Contact

If you encounter issues not covered here:
1. Check the error message and traceback
2. Verify your data format matches expected structure
3. Test with the demo data first