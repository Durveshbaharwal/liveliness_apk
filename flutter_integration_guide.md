# Enhanced Liveness Detection Model - Flutter Integration Guide

## Overview
This guide explains how to integrate the enhanced liveness detection model into your Flutter application.

## Model Files
- `enhanced_liveliness.tflite` - The optimized TensorFlow Lite model
- `enhanced_flutter_config.json` - Configuration parameters for Flutter integration

## Flutter Setup

### 1. Add Dependencies
Add the following to your `pubspec.yaml`:

```yaml
dependencies:
  tflite_flutter: ^0.10.1
  camera: ^0.10.0
  image: ^3.0.0
```

### 2. Model Integration

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class LivenessDetector {
  static const String modelPath = 'assets/models/enhanced_liveliness.tflite';
  Interpreter? _interpreter;
  
  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(modelPath);
    print('Model loaded successfully');
  }
  
  Future<LivenessResult> detectLiveness(Uint8List yuvData) async {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }
    
    // Prepare input tensor
    var input = _prepareInput(yuvData);
    var output = List<double>.filled(1, 0);
    
    // Run inference
    _interpreter!.run(input, output);
    
    return _interpretOutput(output[0]);
  }
  
  LivenessResult _interpretOutput(double prediction) {
    int predClass = prediction.round();
    switch (predClass) {
      case 0:
        return LivenessResult(isLive: true, confidence: prediction);
      case 1:
        return LivenessResult(isLive: false, reason: 'Spoof detected');
      case 2:
        return LivenessResult(isLive: false, reason: 'Multiple faces');
      case 3:
        return LivenessResult(isLive: false, reason: 'Low quality');
      default:
        return LivenessResult(isLive: false, reason: 'Unknown');
    }
  }
}

class LivenessResult {
  final bool isLive;
  final double confidence;
  final String reason;
  
  LivenessResult({
    required this.isLive, 
    this.confidence = 0.0, 
    this.reason = ''
  });
}
```

### 3. Camera Integration

```dart
import 'package:camera/camera.dart';

class CameraController {
  CameraController? _controller;
  final LivenessDetector _detector = LivenessDetector();
  
  Future<void> initializeCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
    );
    
    _controller = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    
    await _controller!.initialize();
    await _detector.loadModel();
  }
  
  void startLivenessDetection() {
    _controller!.startImageStream((CameraImage image) {
      _processFrame(image);
    });
  }
  
  void _processFrame(CameraImage image) async {
    if (image.format.group == ImageFormatGroup.yuv420) {
      // Convert CameraImage to Uint8List
      final yuvData = _convertYUV420ToUint8List(image);
      
      // Run liveness detection
      final result = await _detector.detectLiveness(yuvData);
      
      // Handle result
      _handleLivenessResult(result);
    }
  }
}
```

## Performance Optimization

### 1. Frame Processing
- Process every 2nd or 3rd frame to reduce CPU usage
- Use background isolate for inference to avoid UI blocking

### 2. Memory Management
- Dispose of unused camera images immediately
- Implement proper resource cleanup

### 3. Quality Control
- Implement face size validation
- Check lighting conditions
- Ensure proper face positioning

## Configuration Parameters

The model supports the following configuration parameters:

- `confidence_threshold`: 0.75 (minimum confidence for live detection)
- `frame_skip`: 1 (process every nth frame)
- `analysis_window`: 15 (frames to analyze for final decision)
- `quality_threshold`: 0.6 (minimum quality score)

## Error Handling

Implement comprehensive error handling for:
- Model loading failures
- Camera initialization issues
- Low-quality input frames
- Inference timeouts

## Testing

Test the integration with:
- Various lighting conditions
- Different face angles and distances
- Spoof attack scenarios (photos, videos, masks)
- Performance under different device specifications

## Security Considerations

- All processing happens on-device
- No data is transmitted to external servers
- Model integrity verification recommended
- Implement additional security layers as needed

## Troubleshooting

Common issues and solutions:
1. **Model not loading**: Ensure the .tflite file is in the correct assets folder
2. **Poor performance**: Reduce input resolution or increase frame skip
3. **False positives**: Adjust confidence threshold
4. **Camera issues**: Check permissions and camera availability

For more detailed implementation examples and best practices, refer to the included sample Flutter application.
