# anti_sleep_alarm.py
# ============================================================
# Anti-Sleep Alarm System using OpenCV Eye Detection
# Author  : Computer Vision Project
# Version : 1.0 (Basic - Haar Cascade)
# ============================================================

# ─── IMPORTS ─────────────────────────────────────────────────
import cv2          # OpenCV for computer vision
import time         # For timing eye closure duration
import numpy as np  # For numerical operations
import sys          # For system operations
import os           # For file path operations

# Try to import pygame for alarm sound
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not found. Trying alternative sound methods...")

# Try winsound for Windows users
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# ─── CONFIGURATION SETTINGS ──────────────────────────────────
class Config:
    """
    Central configuration class.
    Adjust these values to tune the system sensitivity.
    """
    # ── Time Settings ──────────────────────────────────────
    SLEEP_THRESHOLD    = 5.0   # Seconds eyes must be closed to trigger alarm
    DROWSY_THRESHOLD   = 2.0   # Seconds for drowsy warning (before alarm)
    
    # ── Detection Settings ─────────────────────────────────
    FACE_SCALE_FACTOR  = 1.3   # How much image size is reduced at each scale
    FACE_MIN_NEIGHBORS = 5     # Higher = fewer detections but better quality
    FACE_MIN_SIZE      = (80, 80)  # Minimum face size to detect
    
    EYE_SCALE_FACTOR   = 1.1   # Scale factor for eye detection
    EYE_MIN_NEIGHBORS  = 10    # Min neighbors for eye detection
    EYE_MIN_SIZE       = (25, 25)  # Minimum eye size
    
    # ── Display Settings ───────────────────────────────────
    WINDOW_NAME        = "Anti-Sleep Alarm System"
    FRAME_WIDTH        = 640   # Camera frame width
    FRAME_HEIGHT       = 480   # Camera frame height
    
    # ── Color Settings (BGR format) ────────────────────────
    COLOR_GREEN        = (0, 255, 0)    # Awake status
    COLOR_RED          = (0, 0, 255)    # Sleeping status
    COLOR_ORANGE       = (0, 165, 255)  # Drowsy/warning status
    COLOR_BLUE         = (255, 0, 0)    # Face bounding box
    COLOR_YELLOW       = (0, 255, 255)  # Eye bounding box
    COLOR_WHITE        = (255, 255, 255)# Text color
    COLOR_BLACK        = (0, 0, 0)      # Background
    
    # ── Alarm Settings ─────────────────────────────────────
    ALARM_FILE         = "alarm.wav"   # Alarm sound file path
    ALARM_VOLUME       = 0.8           # Volume (0.0 to 1.0)
    
    # ── Font Settings ──────────────────────────────────────
    FONT               = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE   = 1.0
    FONT_SCALE_SMALL   = 0.6
    FONT_THICKNESS     = 2


# ─── SOUND MANAGER ───────────────────────────────────────────
class SoundManager:
    """
    Handles all alarm sound operations.
    Supports pygame, winsound, and silent fallback.
    """
    
    def __init__(self, alarm_file):
        """Initialize sound system with the alarm file."""
        self.alarm_file   = alarm_file
        self.is_playing   = False
        self.sound_system = None
        self._initialize_sound()
    
    def _initialize_sound(self):
        """Initialize the best available sound system."""
        
        # Check if alarm file exists
        if not os.path.exists(self.alarm_file):
            print(f"⚠️  Alarm file '{self.alarm_file}' not found!")
            print("   Run 'run_first_generate_alarm.py' to create it.")
            self.sound_system = "none"
            return
        
        # Try pygame first (best cross-platform support)
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(
                    frequency=44100,  # Audio sample rate
                    size=-16,         # 16-bit audio
                    channels=1,       # Mono
                    buffer=4096       # Buffer size
                )
                pygame.mixer.music.load(self.alarm_file)
                pygame.mixer.music.set_volume(Config.ALARM_VOLUME)
                self.sound_system = "pygame"
                print("✅ Sound system: pygame initialized")
                return
            except Exception as e:
                print(f"⚠️  pygame initialization failed: {e}")
        
        # Try winsound (Windows only)
        if WINSOUND_AVAILABLE:
            self.sound_system = "winsound"
            print("✅ Sound system: winsound initialized")
            return
        
        # No sound available
        self.sound_system = "none"
        print("⚠️  No sound system available. Alarm will be visual only.")
    
    def play_alarm(self):
        """Start playing the alarm sound."""
        if self.is_playing:
            return  # Don't restart if already playing
            
        self.is_playing = True
        print("🚨 ALARM TRIGGERED!")
        
        try:
            if self.sound_system == "pygame":
                # Loop the alarm (-1 means loop forever)
                pygame.mixer.music.play(-1)
                
            elif self.sound_system == "winsound":
                # Play alarm asynchronously on Windows
                winsound.PlaySound(
                    self.alarm_file,
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                )
        except Exception as e:
            print(f"⚠️  Could not play alarm: {e}")
    
    def stop_alarm(self):
        """Stop the alarm sound."""
        if not self.is_playing:
            return
            
        self.is_playing = False
        print("✅ Alarm stopped.")
        
        try:
            if self.sound_system == "pygame":
                pygame.mixer.music.stop()
                
            elif self.sound_system == "winsound":
                winsound.PlaySound(None, winsound.SND_PURGE)
                
        except Exception as e:
            print(f"⚠️  Could not stop alarm: {e}")
    
    def cleanup(self):
        """Clean up sound resources."""
        self.stop_alarm()
        if self.sound_system == "pygame" and PYGAME_AVAILABLE:
            pygame.mixer.quit()


# ─── DETECTOR ────────────────────────────────────────────────
class EyeDetector:
    """
    Handles face and eye detection using Haar Cascade Classifiers.
    Manages eye closure timing and status determination.
    """
    
    def __init__(self):
        """Initialize Haar Cascade classifiers."""
        self._load_cascades()
        
        # ── State Variables ────────────────────────────────
        self.eyes_closed_start = None  # Timestamp when eyes first closed
        self.current_status    = "Awake"  # Current detection status
        self.closure_duration  = 0.0   # How long eyes have been closed
        self.alarm_triggered   = False # Whether alarm is currently active
        self.frame_count       = 0     # Total frames processed
        self.face_detected     = False # Whether face is in frame
    
    def _load_cascades(self):
        """Load Haar Cascade XML files."""
        
        # ── Face Cascade ───────────────────────────────────
        # OpenCV includes these files - find them automatically
        cascade_path = cv2.data.haarcascades
        
        face_cascade_path = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        eye_cascade_path  = os.path.join(cascade_path, 'haarcascade_eye.xml')
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"❌ Cannot load face cascade from: {face_cascade_path}")
        print("✅ Face cascade loaded successfully")
        
        # Load eye detector
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if self.eye_cascade.empty():
            raise IOError(f"❌ Cannot load eye cascade from: {eye_cascade_path}")
        print("✅ Eye cascade loaded successfully")
    
    def detect_faces(self, gray_frame):
        """
        Detect faces in grayscale frame.
        
        Parameters:
            gray_frame : Grayscale image
            
        Returns:
            List of face rectangles (x, y, w, h)
        """
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor  = Config.FACE_SCALE_FACTOR,
            minNeighbors = Config.FACE_MIN_NEIGHBORS,
            minSize      = Config.FACE_MIN_SIZE,
            flags        = cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def detect_eyes(self, face_gray):
        """
        Detect eyes within a face region.
        
        Parameters:
            face_gray : Grayscale region of interest (face area)
            
        Returns:
            List of eye rectangles (x, y, w, h)
        """
        # Use only the upper half of face for eye detection
        # (lower half contains nose and mouth - reduces false detections)
        height = face_gray.shape[0]
        upper_face = face_gray[0:int(height * 0.6), :]  # Top 60% of face
        
        eyes = self.eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor  = Config.EYE_SCALE_FACTOR,
            minNeighbors = Config.EYE_MIN_NEIGHBORS,
            minSize      = Config.EYE_MIN_SIZE,
            flags        = cv2.CASCADE_SCALE_IMAGE
        )
        return eyes
    
    def update_status(self, eyes_detected):
        """
        Update the drowsiness status based on eye detection.
        
        Parameters:
            eyes_detected : Boolean - whether eyes are currently detected
            
        Returns:
            Tuple: (status_string, closure_duration, alarm_should_play)
        """
        current_time = time.time()
        
        if eyes_detected:
            # ── Eyes Open ──────────────────────────────────
            # Reset all timers and status
            self.eyes_closed_start = None
            self.closure_duration  = 0.0
            self.current_status    = "Awake"
            self.alarm_triggered   = False
            
        else:
            # ── Eyes Closed ────────────────────────────────
            if self.eyes_closed_start is None:
                # First frame with eyes closed - start timer
                self.eyes_closed_start = current_time
                print(f"⚠️  Eyes closed detected - starting timer...")
            
            # Calculate how long eyes have been closed
            self.closure_duration = current_time - self.eyes_closed_start
            
            # Determine status based on duration
            if self.closure_duration >= Config.SLEEP_THRESHOLD:
                self.current_status  = "Sleeping"
                self.alarm_triggered = True
                
            elif self.closure_duration >= Config.DROWSY_THRESHOLD:
                self.current_status  = "Drowsy"
                self.alarm_triggered = False
                
            else:
                self.current_status  = "Eyes Closed"
                self.alarm_triggered = False
        
        return self.current_status, self.closure_duration, self.alarm_triggered


# ─── DISPLAY MANAGER ─────────────────────────────────────────
class DisplayManager:
    """
    Handles all visual display operations on the video frame.
    Draws bounding boxes, status text, timer, and overlays.
    """
    
    def __init__(self):
        """Initialize display parameters."""
        self.frame_count = 0
        self.fps_timer   = time.time()
        self.fps         = 0
    
    def calculate_fps(self):
        """Calculate and update FPS counter."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_timer
        
        if elapsed >= 1.0:  # Update FPS every second
            self.fps       = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = time.time()
        
        return self.fps
    
    def draw_face_box(self, frame, x, y, w, h):
        """Draw bounding box around detected face."""
        # Draw rectangle around face (BLUE color)
        cv2.rectangle(
            frame,
            (x, y),           # Top-left corner
            (x + w, y + h),   # Bottom-right corner
            Config.COLOR_BLUE, # Color (BGR)
            2                  # Line thickness
        )
        # Add "Face" label above the box
        cv2.putText(
            frame, "Face",
            (x, y - 10),
            Config.FONT, 0.5,
            Config.COLOR_BLUE, 1
        )
    
    def draw_eye_boxes(self, frame, face_x, face_y, eyes):
        """Draw bounding boxes around detected eyes."""
        for (ex, ey, ew, eh) in eyes:
            # Adjust eye coordinates to full frame coordinates
            eye_x = face_x + ex
            eye_y = face_y + ey
            
            # Draw rectangle around each eye (YELLOW)
            cv2.rectangle(
                frame,
                (eye_x, eye_y),
                (eye_x + ew, eye_y + eh),
                Config.COLOR_YELLOW,
                2
            )
            
            # Draw center point of eye
            center_x = eye_x + ew // 2
            center_y = eye_y + eh // 2
            cv2.circle(frame, (center_x, center_y), 3, Config.COLOR_YELLOW, -1)
    
    def draw_status_panel(self, frame, status, closure_duration, alarm_active, fps):
        """
        Draw the status information panel on the frame.
        
        Shows: Status, Timer, FPS, and Instructions
        """
        frame_h, frame_w = frame.shape[:2]
        
        # ── Background Panel ───────────────────────────────
        # Semi-transparent dark background for text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # ── Status Color Selection ─────────────────────────
        if status == "Awake":
            status_color = Config.COLOR_GREEN
            status_icon  = "✓"
        elif status == "Drowsy":
            status_color = Config.COLOR_ORANGE
            status_icon  = "⚠"
        elif status in ["Sleeping", "Eyes Closed"]:
            status_color = Config.COLOR_RED
            status_icon  = "✗"
        else:
            status_color = Config.COLOR_WHITE
            status_icon  = "?"
        
        # ── Status Text ────────────────────────────────────
        status_text = f"Status: {status}"
        cv2.putText(
            frame, status_text,
            (10, 35),
            Config.FONT,
            Config.FONT_SCALE_LARGE,
            status_color,
            Config.FONT_THICKNESS
        )
        
        # ── Timer Display ──────────────────────────────────
        if closure_duration > 0:
            timer_text  = f"Eyes Closed: {closure_duration:.1f}s / {Config.SLEEP_THRESHOLD:.0f}s"
            timer_color = Config.COLOR_RED if alarm_active else Config.COLOR_ORANGE
        else:
            timer_text  = "Eyes Open"
            timer_color = Config.COLOR_GREEN
        
        cv2.putText(
            frame, timer_text,
            (10, 70),
            Config.FONT,
            Config.FONT_SCALE_SMALL,
            timer_color,
            1
        )
        
        # ── FPS Counter ────────────────────────────────────
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text,
            (frame_w - 120, 25),
            Config.FONT,
            Config.FONT_SCALE_SMALL,
            Config.COLOR_WHITE, 1
        )
        
        # ── Alarm Warning ──────────────────────────────────
        if alarm_active:
            self._draw_alarm_warning(frame)
        
        # ── Progress Bar for Eye Closure ───────────────────
        self._draw_closure_progress_bar(frame, closure_duration)
        
        # ── Instructions ───────────────────────────────────
        self._draw_instructions(frame)
        
        return frame
    
    def _draw_alarm_warning(self, frame):
        """Draw flashing alarm warning on frame."""
        frame_h, frame_w = frame.shape[:2]
        
        # Flash effect based on time
        if int(time.time() * 2) % 2 == 0:  # Flash every 0.5 seconds
            # Red border around entire frame
            border_thickness = 8
            cv2.rectangle(
                frame,
                (0, 0),
                (frame_w - 1, frame_h - 1),
                Config.COLOR_RED,
                border_thickness
            )
            
            # "ALARM!" text in center
            alarm_text = "!!! WAKE UP !!!"
            text_size  = cv2.getTextSize(
                alarm_text, Config.FONT, 1.5, 3
            )[0]
            text_x = (frame_w - text_size[0]) // 2
            text_y = frame_h // 2
            
            # Shadow for readability
            cv2.putText(
                frame, alarm_text,
                (text_x + 2, text_y + 2),
                Config.FONT, 1.5,
                Config.COLOR_BLACK, 4
            )
            cv2.putText(
                frame, alarm_text,
                (text_x, text_y),
                Config.FONT, 1.5,
                Config.COLOR_RED, 3
            )
    
    def _draw_closure_progress_bar(self, frame, closure_duration):
        """Draw a progress bar showing eye closure duration."""
        frame_w = frame.shape[1]
        
        # Bar position and size
        bar_x      = 10
        bar_y      = 85
        bar_width  = frame_w - 20
        bar_height = 10
        
        # Draw background bar (gray)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100), -1
        )
        
        # Calculate fill amount
        if Config.SLEEP_THRESHOLD > 0:
            fill_ratio  = min(closure_duration / Config.SLEEP_THRESHOLD, 1.0)
            fill_width  = int(bar_width * fill_ratio)
            
            # Color changes from green to orange to red
            if fill_ratio < 0.4:
                bar_color = Config.COLOR_GREEN
            elif fill_ratio < 0.7:
                bar_color = Config.COLOR_ORANGE
            else:
                bar_color = Config.COLOR_RED
            
            # Draw filled portion
            if fill_width > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + fill_width, bar_y + bar_height),
                    bar_color, -1
                )
        
        # Border around bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            Config.COLOR_WHITE, 1
        )
    
    def _draw_instructions(self, frame):
        """Draw keyboard instructions at bottom of frame."""
        frame_h, frame_w = frame.shape[:2]
        
        instruction_text = "Press 'Q' to Quit | '+'/'-' to Adjust Sensitivity"
        cv2.putText(
            frame, instruction_text,
            (10, frame_h - 10),
            Config.FONT, 0.45,
            Config.COLOR_WHITE, 1
        )
    
    def draw_no_face_warning(self, frame):
        """Show warning when no face is detected."""
        frame_h, frame_w = frame.shape[:2]
        
        warning_text = "No Face Detected - Position yourself in frame"
        text_size    = cv2.getTextSize(
            warning_text, Config.FONT, 0.6, 1
        )[0]
        text_x = (frame_w - text_size[0]) // 2
        text_y = frame_h - 40
        
        cv2.putText(
            frame, warning_text,
            (text_x, text_y),
            Config.FONT, 0.6,
            Config.COLOR_ORANGE, 1
        )


# ─── MAIN APPLICATION ─────────────────────────────────────────
class AntiSleepAlarmSystem:
    """
    Main application class that orchestrates all components.
    Manages the main processing loop.
    """
    
    def __init__(self):
        """Initialize all system components."""
        print("=" * 55)
        print("  Anti-Sleep Alarm System - Starting Up...")
        print("=" * 55)
        
        # Initialize components
        self.sound_manager   = SoundManager(Config.ALARM_FILE)
        self.eye_detector    = EyeDetector()
        self.display_manager = DisplayManager()
        
        # Initialize camera
        self.camera = self._initialize_camera()
        
        # System state
        self.running        = True
        self.session_start  = time.time()
        
        print("=" * 55)
        print("  System Ready! Press 'Q' to quit.")
        print(f"  Alarm threshold: {Config.SLEEP_THRESHOLD} seconds")
        print(f"  Drowsy warning : {Config.DROWSY_THRESHOLD} seconds")
        print("=" * 55)
    
    def _initialize_camera(self):
        """Initialize webcam with proper settings."""
        # Try to open default camera (index 0)
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            # Try other camera indices
            for idx in range(1, 4):
                camera = cv2.VideoCapture(idx)
                if camera.isOpened():
                    print(f"✅ Camera found at index {idx}")
                    break
            else:
                raise IOError("❌ Cannot access any camera!")
        
        # Set camera resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS
        
        # Read actual resolution (may differ from requested)
        actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized: {actual_w}x{actual_h}")
        
        return camera
    
    def process_frame(self, frame):
        """
        Main frame processing pipeline.
        
        Steps:
        1. Convert to grayscale for detection
        2. Detect faces
        3. Detect eyes within faces
        4. Update status
        5. Draw visualizations
        
        Parameters:
            frame : Raw BGR frame from camera
            
        Returns:
            Processed frame with annotations
        """
        # ── Step 1: Convert to Grayscale ───────────────────
        # Grayscale conversion speeds up detection significantly
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection in poor lighting
        gray = cv2.equalizeHist(gray)
        
        # ── Step 2: Detect Faces ───────────────────────────
        faces         = self.eye_detector.detect_faces(gray)
        face_detected = len(faces) > 0
        eyes_detected = False
        
        # ── Step 3: Process Each Detected Face ─────────────
        for (fx, fy, fw, fh) in faces:
            # Draw face bounding box
            self.display_manager.draw_face_box(frame, fx, fy, fw, fh)
            
            # Extract face region for eye detection
            face_gray = gray[fy:fy + fh, fx:fx + fw]
            
            # ── Step 4: Detect Eyes in Face Region ─────────
            eyes = self.eye_detector.detect_eyes(face_gray)
            
            # Eyes detected if we find at least 1 eye
            # (Sometimes one eye might be partially hidden)
            if len(eyes) >= 1:
                eyes_detected = True
                # Draw eye bounding boxes
                self.display_manager.draw_eye_boxes(frame, fx, fy, eyes)
        
        # ── Step 5: Handle No Face Detected ────────────────
        if not face_detected:
            # If no face found, don't count as sleeping
            # (person may have looked away)
            self.eye_detector.eyes_closed_start = None
            self.eye_detector.closure_duration  = 0.0
            self.eye_detector.current_status    = "No Face"
            self.display_manager.draw_no_face_warning(frame)
            # Stop alarm if it was playing
            self.sound_manager.stop_alarm()
            
            status           = "No Face"
            closure_duration = 0.0
            alarm_active     = False
            
        else:
            # ── Step 6: Update Drowsiness Status ───────────
            status, closure_duration, alarm_active = \
                self.eye_detector.update_status(eyes_detected)
            
            # ── Step 7: Control Alarm Sound ────────────────
            if alarm_active:
                self.sound_manager.play_alarm()
            else:
                self.sound_manager.stop_alarm()
        
        # ── Step 8: Calculate FPS ──────────────────────────
        fps = self.display_manager.calculate_fps()
        
        # ── Step 9: Draw Status Panel ──────────────────────
        frame = self.display_manager.draw_status_panel(
            frame, status, closure_duration,
            alarm_active, fps
        )
        
        return frame
    
    def handle_key_press(self, key):
        """
        Handle keyboard input for sensitivity adjustment and quit.
        
        Keys:
            'q' or 'Q' : Quit application
            '+'        : Increase sleep threshold (less sensitive)
            '-'        : Decrease sleep threshold (more sensitive)
            's'        : Show current settings
        """
        if key == ord('q') or key == ord('Q'):
            # Quit application
            print("\n👋 Quitting application...")
            self.running = False
            
        elif key == ord('+') or key == ord('='):
            # Increase threshold (less sensitive)
            Config.SLEEP_THRESHOLD = min(Config.SLEEP_THRESHOLD + 0.5, 15.0)
            print(f"⚙️  Sleep threshold increased: {Config.SLEEP_THRESHOLD:.1f}s")
            
        elif key == ord('-'):
            # Decrease threshold (more sensitive)
            Config.SLEEP_THRESHOLD = max(Config.SLEEP_THRESHOLD - 0.5, 1.0)
            print(f"⚙️  Sleep threshold decreased: {Config.SLEEP_THRESHOLD:.1f}s")
            
        elif key == ord('s') or key == ord('S'):
            # Show settings
            print(f"\n📊 Current Settings:")
            print(f"   Sleep Threshold : {Config.SLEEP_THRESHOLD:.1f}s")
            print(f"   Drowsy Threshold: {Config.DROWSY_THRESHOLD:.1f}s")
    
    def run(self):
        """Main application loop."""
        print("\n▶️  Starting main loop... (Press 'Q' to quit)")
        
        while self.running:
            # ── Capture Frame ──────────────────────────────
            ret, frame = self.camera.read()
            
            if not ret:
                print("⚠️  Failed to capture frame. Retrying...")
                time.sleep(0.1)
                continue
            
            # ── Flip Frame (Mirror Effect) ─────────────────
            # Makes it feel more natural (like a mirror)
            frame = cv2.flip(frame, 1)
            
            # ── Process Frame ──────────────────────────────
            processed_frame = self.process_frame(frame)
            
            # ── Display Frame ──────────────────────────────
            cv2.imshow(Config.WINDOW_NAME, processed_frame)
            
            # ── Handle Key Press ───────────────────────────
            # waitKey(1) = wait 1ms for key press, needed for display
            key = cv2.waitKey(1) & 0xFF
            self.handle_key_press(key)
            
            # ── Check Window Close Button ──────────────────
            # Handle window close button (X button)
            if cv2.getWindowProperty(
                Config.WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            ) < 1:
                self.running = False
        
        # ── Cleanup ────────────────────────────────────────
        self.cleanup()
    
    def cleanup(self):
        """Release all resources properly."""
        print("\n🧹 Cleaning up resources...")
        
        # Stop alarm
        self.sound_manager.cleanup()
        
        # Release camera
        if self.camera.isOpened():
            self.camera.release()
            print("✅ Camera released")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("✅ Windows closed")
        
        # Show session summary
        session_duration = time.time() - self.session_start
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {session_duration:.1f} seconds")
        print("=" * 55)
        print("  Thank you for using Anti-Sleep Alarm System!")
        print("=" * 55)


# ─── ENTRY POINT ─────────────────────────────────────────────
def main():
    """Application entry point with error handling."""
    try:
        # Create and run the alarm system
        app = AntiSleepAlarmSystem()
        app.run()
        
    except IOError as e:
        print(f"\n❌ Hardware Error: {e}")
        print("   Check your camera connection and try again.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()