<img width="1270" height="991" alt="Screenshot 2026-07-15 120048" src="https://github.com/user-attachments/assets/5db26114-a1b3-4e34-b148-f3df413c90b4" /># 😴 Anti-Sleep Alarm System using Python & OpenCV

## 📌 Project Overview
The Anti-Sleep Alarm System is a real-time computer vision project that detects whether a person is falling asleep in front of a webcam.  

If the system detects that the user's eyes are closed continuously for more than a defined time (e.g., 10 seconds), it triggers an alarm sound to wake them up.

This project is useful for:
- Students during online classes 📚
- People working late nights 💻
- Basic driver drowsiness detection 🚗

---

## 🚀 Features

- 👁️ Real-time face and eye detection
- ⏱️ Sleep detection based on eye closure duration
- 🔊 Alarm system using sound
- 🟢 "Awake" and 🔴 "Sleeping" status display
- 📷 Live webcam feed
- ⌨️ Press 'q' to exit

---

## 🛠️ Technologies Used

- Python 🐍
- OpenCV
- Pygame (for alarm sound)

---

## 📂 Project Structure
AntiSleepProject/
│── venv/ # Virtual environment (not uploaded to GitHub)
│── anti_sleep_alarm.py # Main Python script
│── alarm.wav # Alarm sound file
│── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/anti-sleep-alarm.git
cd anti-sleep-alarm

🧠 How It Works
The webcam captures real-time video.
OpenCV detects the face using Haar Cascade.
Eyes are detected within the face region.
If eyes are not detected:
Timer starts ⏱️
If eyes remain closed for more than 10 seconds:
Alarm is triggered 🔊
If eyes open again:
Timer resets

⚠️ Limitations
Haar Cascade is not 100% accurate
Performance depends on lighting conditions
May fail with glasses or low camera quality

🚀 Future Improvements
Use Eye Aspect Ratio (EAR) for better accuracy
Add GUI interface
Add warning before alarm
Mobile or web-based version

📸 Demo



