# TensorFlow Lite Liveness Detection

## Overview
This package provides a complete TensorFlow Lite implementation for liveness detection based on texture and reflection analysis. The model can distinguish between real human faces and spoofing attempts using photos, videos, or screens.

## Features
- Real-time liveness detection
- Configurable thresholds via JSON
- Support for multiple camera formats (NV21, YUV420, RGB)
- Optimized for mobile deployment
- Multiple face detection

## Quick Start

### 1. Generate the model and configuration
```bash
python setup_script.py
```

### 2. Test the model
```bash
python test_model.py
```

### 3. Integrate with Flutter
- Copy `liveness_detection.tflite` to `assets/`
- Copy `flutter_config.json` to `assets/`
- Use the provided Flutter integration code

## Model Specifications

### Input
- **Format**: RGB image
- **Shape**: [1, 480, 640, 3]
- **Type**: uint8
- **Range**: 0-255

### Output
- **Shape**: [1]
- **Type**: int32
- **Values**:
  - `0`: Live person detected
  - `1`: Spoof detected
  - `2`: Multiple faces detected

## Configuration Parameters

Edit `flutter_config.json` to adjust detection sensitivity:

- `texture_threshold`: Minimum texture variance for live skin
- `reflection_threshold`: Maximum reflection score for live person
- `skin_texture_threshold`: Minimum skin texture complexity
- `bright_pixel_threshold`: Brightness threshold for reflection detection

## Performance Tips

1. **Lighting**: Ensure adequate lighting for best results
2. **Distance**: Keep face at appropriate distance from camera
3. **Angle**: Face camera directly for optimal detection
4. **Stability**: Minimize camera shake

## Troubleshooting

### High False Positive Rate
- Increase `texture_threshold`
- Decrease `reflection_threshold`
- Adjust `bright_pixel_threshold` for your lighting conditions

### High False Negative Rate
- Decrease `texture_threshold`
- Increase `reflection_threshold`
- Check camera quality and lighting

## License
This implementation is provided as-is for educational and development purposes.
