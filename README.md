# Automated Industrial Parts Picking Robot
![Demo Video](assets/demo.gif)

## Overview
This project implements an automated system for sorting industrial parts (screws) using computer vision and a Delta robot. The system can detect, classify, and sort different types of screws on a moving conveyor belt with high accuracy and efficiency.

## Key Features
- **Real-time Object Detection**: YOLOv8-based detection system for accurate screw identification
- **Precision Sorting**: Delta robot with precise pick-and-place capabilities
- **Conveyor Integration**: Synchronized operation with variable-speed conveyor belt
- **User Interface**: Intuitive GUI for both automated and manual control
- **Calibration System**: Automated camera-robot calibration for accurate positioning

## System Architecture

### Hardware Components
- Delta Robot (COM3)
- Basler Camera
- Conveyor Belt System (COM11)
- Position Encoder (COM6)
- Industrial PC

### Software Stack
- Python 3.8+
- OpenCV
- PyPylon (Basler camera interface)
- Ultralytics YOLOv8
- Custom Robot Control Library

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khoaissleep/Automated_Industrial_Parts_Picking_Robot.git
cd Automated_Industrial_Parts_Picking_Robot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Hardware Setup:
   - Connect Delta Robot to COM3
   - Connect Conveyor to COM11
   - Connect Encoder to COM6
   - Connect Basler camera via USB

## Usage

### Starting the System
```bash
python Robot_control/rbcontrol_demo.py
```

### Operating Modes

#### Automated Mode
1. Launch the application
2. Connect hardware components
3. Click "START" to begin automated operation
4. System will:
   - Detect screws on conveyor
   - Classify by type
   - Sort into appropriate bins
   - Track sorting statistics

#### Manual Mode
- Camera control
- Robot positioning
- Gripper operation
- Conveyor control
- Custom G-code execution

#### Calibration Mode
1. Access Calibration tab
2. Follow on-screen instructions
3. Click calibration points
4. Save calibration parameters

## Performance
- Detection accuracy: >95%
- Sorting speed: Up to 60 parts/minute
- Operating temperature: 0-40°C
- Power consumption: <500W

## Troubleshooting

### Common Issues
1. Camera Connection
   - Check USB connection
   - Verify PyPylon installation
   - Restart application

2. Robot Movement
   - Verify COM port
   - Check power supply
   - Recalibrate if needed

3. Conveyor Issues
   - Check power connection
   - Verify speed settings
   - Inspect belt tension

## Future Improvements
- Multi-camera support
- Enhanced error recovery
- Inventory management integration
- Additional part type support
- Machine learning model improvements

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For support or inquiries, please open an issue in the repository.
