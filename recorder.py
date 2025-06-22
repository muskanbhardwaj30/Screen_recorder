import cv2
import numpy as np
import datetime
import sys
from PIL import ImageGrab, ImageDraw, ImageFont
from typing import Tuple


# Context manager for VideoWriter
class VideoWriterContext:
    def __init__(self, filename: str, fourcc: str, fps: float, resolution: Tuple[int, int]):
        self.filename = filename
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.fps = fps
        self.resolution = resolution
        self.writer = None

    def __enter__(self) -> cv2.VideoWriter:
        try:
            self.writer = cv2.VideoWriter(
                self.filename,
                self.fourcc,
                self.fps,
                self.resolution
            )
            if not self.writer.isOpened():
                raise IOError(f"Failed to initialize video writer for {self.filename}")
            return self.writer
        except Exception as e:
            sys.exit(f"VideoWriter Error: {e}")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.writer:
            self.writer.release()
        if exc_type:
            print(f"Exception during recording: {exc_value}")


# Screen region selector
def select_region() -> Tuple[int, int, int, int]:
    try:
        scr = np.array(ImageGrab.grab())
        roi = cv2.selectROI(
            "Select Area (ESC=cancel, ENTER=confirm)",
            cv2.cvtColor(scr, cv2.COLOR_RGB2BGR),
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            print("Using full screen capture")
            return (0, 0, scr.shape[1], scr.shape[0])
        else:
            return (roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
    except Exception as e:
        sys.exit(f"Region selection failed: {e}")


# Load font with fallback
def load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


# Timestamp formatter
format_timestamp = lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def main():
    try:
        print("Starting screen recorder...")
        region = select_region()
        x1, y1, x2, y2 = region
        w, h = x2 - x1, y2 - y1

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"screen_rec_{timestamp}.mp4"

        font = load_font(16)
        print(f"Recording {w}x{h} region | Press 'Q' to stop")

        with VideoWriterContext(fname, 'mp4v', 60.0, (w, h)) as out:
            frame_count = 0
            start_time = datetime.datetime.now()

            while True:
                # Capture screen
                img = ImageGrab.grab(bbox=region)
                draw = ImageDraw.Draw(img)

                # Timestamp at top-left
                draw.text((8, 8), format_timestamp(), font=font, fill=(255, 0, 0))

                # REC indicator at top-right
                rec_circle_x = w - 70
                draw.ellipse((rec_circle_x, 10, rec_circle_x + 12, 22), fill=(255, 0, 0))
                draw.text((rec_circle_x + 16, 8), "REC", font=font, fill=(255, 255, 255))

                # Convert to OpenCV image
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                out.write(frame)
                frame_count += 1

                # FPS calculation
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                fps = frame_count / max(elapsed, 0.001)

                # Preview
                preview = cv2.resize(frame, (426, 240))
                cv2.putText(
                    preview,
                    f"FPS: {fps:.1f} | {w}x{h}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                cv2.imshow('Screen Recorder (Press Q to stop)', preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27 or cv2.getWindowProperty('Screen Recorder (Press Q to stop)', cv2.WND_PROP_VISIBLE) < 1:
                    break

        duration = (datetime.datetime.now() - start_time).total_seconds()
        print(f"Recording complete.\n"
            f"File: {fname}\n"
            f"Duration: {duration:.2f} seconds\n"
            f"Frames: {frame_count}\n"
            f"Average FPS: {frame_count / max(duration, 0.001):.1f}")

    except KeyboardInterrupt:
        print("Recording aborted by user")
    except Exception as e:
        sys.exit(f"Critical error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
