# Restricted Area Breach Detector

Professional real-time surveillance system that detects unauthorized access to protected zones using computer vision.

## Features

- **Multi-Method Detection**: Combines Background Subtraction, Frame Differencing, and Contour Detection
- **Smart Motion Analysis**: Filters false positives with dual verification system
- **Real-time Monitoring**: Live webcam feed with instant breach detection
- **Intuitive Interface**: Mouse-based area selection with visual feedback
- **Auto-dismiss Alerts**: Alarms automatically clear when area is secure
- **Fullscreen Support**: Resizable window with fullscreen toggle
- **Lightweight**: No deep learning required, runs on CPU

## Detection Technology

1. **Background Subtraction (MOG2)**: Learns background patterns over time
2. **Frame Differencing**: Detects motion between consecutive frames
3. **Contour Analysis**: Identifies real objects with minimum size filtering
4. **Dual Verification**: Requires multiple methods to confirm breach

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| Mouse Drag | Select protected area |
| `s` | Save selection |
| `r` | Reset alarm/selection |
| `f` | Toggle fullscreen |
| `q` | Quit |

## Configuration

Key parameters in `main.py`:

- `breach_threshold`: Motion sensitivity (default: 0.15 = 15%)
- `breach_frames_required`: Frames to confirm breach (default: 6)
- `min_contour_area`: Minimum object size in pixels (default: 800)
- `varThreshold`: Background detection sensitivity (default: 6)

**Triple Verification System**: Requires at least 2 of 3 criteria:
1. Valid contour detected + Motion threshold exceeded
2. Frame difference detected + 80% motion threshold
3. Very high motion (150% threshold) for rapid movements

## How It Works

1. Select a protected area by dragging with your mouse
2. Press 's' to activate monitoring
3. System continuously analyzes the area for movement
4. Red alarm triggers when unauthorized entry detected
5. Alarm auto-dismisses when area clears

## Developer

**Huriyeeym** - [GitHub Profile](https://github.com/huriyeeym)

## License

MIT License - Free for personal and commercial use

