# ParkVision-AI

An advanced computer vision-based parking management system that uses Mask R-CNN for real-time vehicle detection and parking space monitoring.

## Overview

This project implements an intelligent parking system that can:
- Detect and track vehicles in parking lots using Mask R-CNN
- Monitor parking space occupancy in real-time
- Identify available parking slots
- Process video feeds from parking surveillance cameras
- Provide visual indicators for occupied and available parking spaces

## Features

- **Real-time Vehicle Detection**: Uses Mask R-CNN model pre-trained on COCO dataset for accurate vehicle detection
- **Parking Space Monitoring**: Tracks parking space occupancy using IoU (Intersection over Union) calculations
- **Motion Detection**: Implements motion detection to filter out moving vehicles
- **Visual Feedback**: Provides color-coded visual indicators (green for available, red for occupied spaces)
- **Video Processing**: Supports processing of video feeds from parking lot cameras

## Technical Architecture

### Components

1. **Mask R-CNN Model**
   - Based on Feature Pyramid Network (FPN) and ResNet101 backbone
   - Pre-trained on MS COCO dataset
   - Fine-tuned for vehicle detection

2. **Video Processing Pipeline**
   - Frame extraction and processing
   - Motion detection using frame differencing
   - Vehicle detection and tracking
   - Parking space status updates

3. **Parking Space Analysis**
   - IoU-based occupancy detection
   - Dynamic space status monitoring
   - Real-time availability updates

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- Mask R-CNN dependencies
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ParkVision-AI.git
cd ParkVision-AI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained Mask R-CNN weights:
```bash
# Download mask_rcnn_coco.h5 from the official Mask R-CNN repository
```

## Usage

1. Place your parking lot video in the `Videos` directory

2. Run the main script:
```bash
python Intelligent_parking.py
```

3. The system will:
   - Process the video feed
   - Detect and track vehicles
   - Monitor parking space occupancy
   - Generate output video with visual indicators

## Configuration

The system can be configured through the `Config_Mask_RCNN` class:

- `DETECTION_MIN_CONFIDENCE`: Confidence threshold for vehicle detection (default: 0.6)
- `GPU_COUNT`: Number of GPUs to use (default: 1)
- `IMAGES_PER_GPU`: Batch size per GPU (default: 1)
- `NUM_CLASSES`: Number of classes to detect (default: 81, COCO dataset)

## How It Works

1. **Vehicle Detection**
   - Each frame is processed through the Mask R-CNN model
   - The model identifies vehicles and their precise locations
   - Bounding boxes are generated around detected vehicles

2. **Parking Space Monitoring**
   - Initial frame analysis establishes parking space locations
   - IoU calculations determine space occupancy
   - Threshold-based occupancy determination (IoU < 0.15 considered vacant)

3. **Motion Detection**
   - Frame differencing identifies moving vehicles
   - Prevents false parking space occupancy detection
   - Enhances accuracy of space availability monitoring

## Output

The system generates:
- Processed video with visual indicators
- Real-time parking space status
- Color-coded parking space visualization
  - Green: Available space
  - Red: Occupied space

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mask R-CNN implementation based on Matterport's [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
- COCO dataset for pre-trained model weights 