import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import supervision as sv
import yt_dlp
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from datetime import datetime, timedelta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

TRACKER = DeepSort(
    max_age=15,
    n_init=2,
    max_cosine_distance=0.4,
    nn_budget=100,
    embedder="mobilenet",
    half=True,
    bgr=True
)

roi_points = []
roi_selected = False

class MainMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Counter - Main Menu")
        self.root.geometry("400x530")
        self.root.configure(bg='#2b2b2b')
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
        self.youtube_url = ""
        self.selected_objects = []
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(padx=50, pady=10, anchor="w")
        
        title_label = tk.Label(title_frame, text="Traffic Counter System", 
                             font=("Arial", 20, "bold"), 
                             fg='white', bg='#2b2b2b', anchor="w")
        title_label.pack()
        
        # YouTube URL input
        url_frame = tk.Frame(self.root, bg='#2b2b2b')
        url_frame.pack(pady=10, padx=20, fill='x')
        
        url_label = tk.Label(url_frame, text="YouTube Live Stream URL:", 
                           font=("Arial", 12), fg='white', bg='#2b2b2b')
        url_label.pack(anchor='w')
        
        self.url_entry = tk.Entry(url_frame, font=("Arial", 10), width=60)
        self.url_entry.pack(pady=5, fill='x')
        self.url_entry.insert(0, "https://www.youtube.com/watch?v=D5kKdEBmrYU")
        
        # Object selection
        objects_frame = tk.Frame(self.root, bg="#2b2b2b")
        objects_frame.pack(pady=10, padx=20, fill='x')
        
        objects_label = tk.Label(objects_frame, text="Select Objects to Count:", 
                               font=("Arial", 12), fg='white', bg='#2b2b2b')
        objects_label.pack(anchor='w')
        
        # Checkboxes for object selection
        checkbox_frame = tk.Frame(objects_frame, bg='#2b2b2b')
        checkbox_frame.pack(pady=10, fill='x')
        
        self.car_var = tk.BooleanVar(value=True)
        self.truck_var = tk.BooleanVar(value=True)
        self.bus_var = tk.BooleanVar(value=True)
        self.person_var = tk.BooleanVar(value=True)
        
        car_cb = tk.Checkbutton(checkbox_frame, text="🚗 Cars", variable=self.car_var,
                              font=("Arial", 10), fg='white', bg='#2b2b2b',
                              selectcolor='#404040', activebackground='#2b2b2b')
        car_cb.grid(row=0, column=0, sticky='w', padx=10)
        
        truck_cb = tk.Checkbutton(checkbox_frame, text="🚚 Trucks", variable=self.truck_var,
                                font=("Arial", 10), fg='white', bg='#2b2b2b',
                                selectcolor='#404040', activebackground='#2b2b2b')
        truck_cb.grid(row=0, column=1, sticky='w', padx=10)
        
        bus_cb = tk.Checkbutton(checkbox_frame, text="🚌 Buses", variable=self.bus_var,
                              font=("Arial", 10), fg='white', bg='#2b2b2b',
                              selectcolor='#404040', activebackground='#2b2b2b')
        bus_cb.grid(row=1, column=0, sticky='w', padx=10, pady=10)
        
        person_cb = tk.Checkbutton(checkbox_frame, text="🚶 People", variable=self.person_var,
                                 font=("Arial", 10), fg='white', bg='#2b2b2b',
                                 selectcolor='#404040', activebackground='#2b2b2b')
        person_cb.grid(row=1, column=1, sticky='w', padx=10, pady=10)
        
        # Instructions
        instructions_frame = tk.Frame(self.root, bg='#2b2b2b')
        instructions_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        instructions_label = tk.Label(instructions_frame, text="Instructions:", 
                                    font=("Arial", 12, "bold"), fg='white', bg='#2b2b2b')
        instructions_label.pack(anchor='w')
        
        instructions_text = """
• Enter a YouTube live stream URL above
• Select which objects you want to count
• Click 'Start Tracking' to begin
• In the video window:
- Click 4 points to define counting area
- Press '4' to reset the area
- Press 's' to toggle detection
- Press 't' to open/close time tracking window
- Press 'q' to quit
"""
        
        instructions_display = tk.Label(instructions_frame, text=instructions_text, 
                                      font=("Arial", 9), fg='#cccccc', bg='#2b2b2b',
                                      justify='left')
        instructions_display.pack(anchor='w')
        
        # Start button
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(padx=40, pady=10, anchor="w")
        
        start_button = tk.Button(button_frame, text="Start Tracking", 
                               font=("Arial", 14, "bold"), 
                               bg='#4CAF50', fg='white',
                               padx=30, pady=10,
                               command=self.start_tracking)
        start_button.pack()
        
    def start_tracking(self):
        self.youtube_url = self.url_entry.get().strip()
        
        if not self.youtube_url:
            messagebox.showerror("Error", "Please enter a YouTube URL!")
            return
        
        # Get selected objects
        self.selected_objects = []
        if self.car_var.get():
            self.selected_objects.append("car")
        if self.truck_var.get():
            self.selected_objects.append("truck")
        if self.bus_var.get():
            self.selected_objects.append("bus")
        if self.person_var.get():
            self.selected_objects.append("person")
        
        if not self.selected_objects:
            messagebox.showerror("Error", "Please select at least one object type to count!")
            return
        
        # Hide main menu and start tracking
        self.root.withdraw()
        
        # Start tracking in a separate thread
        tracking_thread = threading.Thread(target=self.run_tracking_wrapper)
        tracking_thread.daemon = True
        tracking_thread.start()
    
    def run_tracking_wrapper(self):
        try:
            run_tracking_with_counting(self.youtube_url, self.selected_objects)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            # Show main menu again when tracking ends
            self.root.deiconify()
    
    def run(self):
        self.root.mainloop()

class TimeTrackingWindow:
    def __init__(self):
        self.window = None
        self.text_widget = None
        self.is_open = False
        
    def create_window(self):
        if self.window is not None:
            return
            
        self.window = tk.Toplevel()
        self.window.title("Time Tracking - Debug Log")
        self.window.geometry("500x500")
        self.window.configure(bg='#1e1e1e')
        
        # Make window stay on top
        self.window.attributes('-topmost', True)
        
        # Header
        header_frame = tk.Frame(self.window, bg='#1e1e1e')
        header_frame.pack(fill='x', padx=10, pady=5)
        
        header_label = tk.Label(header_frame, text="Real-time Object Tracking in ROI", 
                              font=("Arial", 12, "bold"), fg='white', bg='#1e1e1e')
        header_label.pack()
        
        # Scrolled text widget for log display
        text_frame = tk.Frame(self.window, bg='#1e1e1e')
        text_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.text_widget = scrolledtext.ScrolledText(text_frame, 
                                                   font=("Courier", 9),
                                                   bg='#2b2b2b', fg='#00ff00',
                                                   wrap=tk.WORD,
                                                   state=tk.DISABLED)
        self.text_widget.pack(fill='both', expand=True)
        
        # Clear button
        button_frame = tk.Frame(self.window, bg='#1e1e1e')
        button_frame.pack(fill='x', padx=10, pady=20)
        
        clear_button = tk.Button(button_frame, text="Clear Log", 
                               font=("Arial", 10), 
                               bg='#ff4444', fg='white',
                               command=self.clear_log)
        clear_button.pack(side='right')
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self.is_open = True
        
    def close_window(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.text_widget = None
            self.is_open = False
    
    def toggle(self):
        if self.is_open:
            self.close_window()
        else:
            self.create_window()
    
    def add_log(self, message):
        if self.text_widget and self.is_open:
            try:
                self.text_widget.config(state=tk.NORMAL)
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
                self.text_widget.see(tk.END)  # Auto-scroll to bottom
                self.text_widget.config(state=tk.DISABLED)
            except:
                pass  # Window might be closed
    
    def clear_log(self):
        if self.text_widget:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.config(state=tk.DISABLED)

# Global time tracking window instance
time_tracking_window = TimeTrackingWindow()

def mouse_callback(event, x, y, flags, param):
    global roi_points, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
        print(f"Point {len(roi_points)} selected: {(x, y)}")
        time_tracking_window.add_log(f"ROI Point {len(roi_points)} selected: {(x, y)}")
        if len(roi_points) == 4:
            roi_selected = True
            print("ROI selection complete.")
            time_tracking_window.add_log("ROI selection complete - tracking activated!")

def point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def is_left_of_line(p0, p1, pt):
    return ((p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0])) > 0

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': 'best[height<=480]/best',
        'http_chunk_size': 10485760,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        if 'url' in info:
            return info['url']
        if 'formats' in info:
            for f in sorted(info['formats'], key=lambda x: x.get('height', 0)):
                if f.get('protocol') in ['https', 'http', 'm3u8_native', 'm3u8'] and f.get('height', 0) <= 480:
                    return f.get('url')
        return None

def reset_all_counters():
    global roi_points, roi_selected
    roi_points = []
    roi_selected = False
    time_tracking_window.add_log("🔄 All counters and ROI reset!")
    return {
        'cars_last_positions': {},
        'counted_cars_horizontal': set(),
        'counted_cars_vertical': set(), 
        'counted_people_horizontal': set(),
        'counted_people_vertical': set(),
        'counted_trucks_horizontal': set(),
        'counted_trucks_vertical': set(),
        'counted_buses_horizontal': set(),
        'counted_buses_vertical': set(),
        'car_entry_times': {},
        'truck_entry_times': {},
        'bus_entry_times': {},
        'person_entry_times': {},
        'car_warned_10': set(),
        'car_warned_20': set(),
        'car_warned_30': set(),
        'truck_warned_10': set(),
        'truck_warned_20': set(),
        'truck_warned_30': set(),
        'bus_warned_10': set(),
        'bus_warned_20': set(),
        'bus_warned_30': set(),
        'person_warned_10': set(),
        'person_warned_20': set(),
        'person_warned_30': set(),
        'person_count_horizontal': 0,
        'person_count_vertical': 0,
        'car_count_horizontal': 0,
        'car_count_vertical': 0,
        'truck_count_horizontal': 0,
        'truck_count_vertical': 0,
        'bus_count_horizontal': 0,
        'bus_count_vertical': 0
    }

def run_tracking_with_counting(youtube_url, selected_objects, confidence_threshold=0.6, frame_skip=3, resize_width=480):
    global roi_points, roi_selected

    stream_url = get_youtube_stream_url(youtube_url)
    if not stream_url:
        print("❌ Could not extract stream URL.")
        time_tracking_window.add_log("❌ Could not extract stream URL.")
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Could not open video stream.")
        time_tracking_window.add_log("❌ Could not open video stream.")
        return

    print("Instructions:")
    print(" - Click 4 points on the video to define counting polygon.")
    print(" - Press '4' to redefine ROI anytime.")
    print(" - Press 'q' to quit.")
    print(" - Press 's' to toggle detection skip for higher FPS.")
    print(" - Press 't' to toggle time tracking window.")
    print(f"\nTracking objects: {', '.join(selected_objects)}")
    print("\nCounting Logic:")
    print(" - Horizontal: Counts objects crossing top/bottom edges of polygon")
    print(" - Vertical: Counts objects crossing left/right edges of polygon")
    print("\nOptimizations applied for better real-time performance!")

    time_tracking_window.add_log(f"🎯 Tracking started for: {', '.join(selected_objects)}")
    time_tracking_window.add_log("📋 Instructions: Click 4 points to define ROI, Press 't' to toggle this window")

    detection_enabled = True

    window_name = "RT-DETR + DeepSORT Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.setMouseCallback(window_name, mouse_callback)

    last_detections = []
    frame_count = 0
    last_detection_time = 0
    detection_interval = 0.15
    skip_frames = 0

    tracking_vars = reset_all_counters()

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 25)
    
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    
    print(f"📺 Stream FPS: {fps}")
    time_tracking_window.add_log(f"📺 Stream FPS: {fps}")
    
    target_fps = 20
    frame_time = 1.0 / target_fps
    last_frame_time = time.time()

    while True:
        start_time = time.time()
        
        ret = False
        frame = None
        max_read_attempts = 3
        
        for _ in range(max_read_attempts):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.01)
        
        if not ret:
            print("❌ Failed to grab frame after multiple attempts.")
            for _ in range(5):
                cap.grab()
            continue

        frame_count += 1
        display_frame = frame.copy()

        current_time = time.time()
        if current_time - last_frame_time > 0.5:
            for _ in range(cap.get(cv2.CAP_PROP_BUFFERSIZE) or 3):
                cap.grab()
            last_frame_time = current_time

        # Draw ROI points and polygon
        if len(roi_points) > 0:
            for i, pt in enumerate(roi_points):
                cv2.circle(display_frame, pt, 7, (0, 255, 255), -1)
                cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if len(roi_points) >= 3:
                cv2.polylines(display_frame, [np.array(roi_points)], 
                             isClosed=(len(roi_points)==4), color=(0, 255, 255), thickness=2)

        # Object detection and tracking
        current_time = time.time()
        if detection_enabled and roi_selected and (current_time - last_detection_time) >= detection_interval:
            h, w, _ = frame.shape
            scale_ratio = resize_width / w
            resized_frame = cv2.resize(frame, (resize_width, int(h * scale_ratio)))

            image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(outputs, threshold=confidence_threshold, target_sizes=target_sizes)

            detections = results[0]
            bboxes = []
            confidences = []
            class_ids = []

            for score, label, box in zip(detections['scores'], detections['labels'], detections['boxes']):
                label_name = model.config.id2label[label.item()].lower()
                if label_name in selected_objects:  # Only track selected objects
                    x1, y1, x2, y2 = box.tolist()

                    x1 = x1 / resize_width * frame.shape[1]
                    x2 = x2 / resize_width * frame.shape[1]
                    y1 = y1 / resized_frame.shape[0] * frame.shape[0]
                    y2 = y2 / resized_frame.shape[0] * frame.shape[0]

                    bboxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(score.item())
                    class_ids.append(label_name)

            detection_list = [[box, conf, cls] for box, conf, cls in zip(bboxes, confidences, class_ids)]
            last_detections = TRACKER.update_tracks(detection_list, frame=frame)
            last_detection_time = current_time

        # Counting and time tracking logic
        if len(roi_points) == 4:
            p0, p1, p2, p3 = roi_points

            for track in last_detections:
                if not track.is_confirmed():
                    continue

                if track.time_since_update > 5:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                label = track.get_det_class()

                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                currently_inside = point_in_polygon(centroid, roi_points)
                last_pos = tracking_vars['cars_last_positions'].get(track_id)

                if last_pos is not None:
                    last_inside = point_in_polygon(last_pos, roi_points)
                    if last_inside != currently_inside:
                        # Crossing detection logic (same as original)
                        crossed_horizontal = False
                        last_side_top = is_left_of_line(p0, p1, last_pos)
                        curr_side_top = is_left_of_line(p0, p1, centroid)
                        last_side_bottom = is_left_of_line(p3, p2, last_pos)
                        curr_side_bottom = is_left_of_line(p3, p2, centroid)
                        if last_side_top != curr_side_top or last_side_bottom != curr_side_bottom:
                            crossed_horizontal = True

                        crossed_vertical = False
                        last_side_left = is_left_of_line(p0, p3, last_pos)
                        curr_side_left = is_left_of_line(p0, p3, centroid)
                        last_side_right = is_left_of_line(p1, p2, last_pos)
                        curr_side_right = is_left_of_line(p1, p2, centroid)
                        if last_side_left != curr_side_left or last_side_right != curr_side_right:
                            crossed_vertical = True

                        # Count crossings for selected objects only
                        if label == "car" and "car" in selected_objects:
                            if crossed_horizontal and track_id not in tracking_vars['counted_cars_horizontal']:
                                tracking_vars['car_count_horizontal'] += 1
                                tracking_vars['counted_cars_horizontal'].add(track_id)
                                message = f"🚗 Car ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['car_count_horizontal']})"
                                print(message)
                                time_tracking_window.add_log(message)
                            elif crossed_vertical and track_id not in tracking_vars['counted_cars_vertical']:
                                tracking_vars['car_count_vertical'] += 1
                                tracking_vars['counted_cars_vertical'].add(track_id)
                                message = f"🚗 Car ID {track_id} crossed VERTICAL edge (count: {tracking_vars['car_count_vertical']})"
                                print(message)
                                time_tracking_window.add_log(message)
                        
                        elif label == "truck" and "truck" in selected_objects:
                            if crossed_horizontal and track_id not in tracking_vars['counted_trucks_horizontal']:
                                tracking_vars['truck_count_horizontal'] += 1
                                tracking_vars['counted_trucks_horizontal'].add(track_id)
                                message = f"🚚 Truck ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['truck_count_horizontal']})"
                                print(message)
                                time_tracking_window.add_log(message)
                            elif crossed_vertical and track_id not in tracking_vars['counted_trucks_vertical']:
                                tracking_vars['truck_count_vertical'] += 1
                                tracking_vars['counted_trucks_vertical'].add(track_id)
                                message = f"🚚 Truck ID {track_id} crossed VERTICAL edge (count: {tracking_vars['truck_count_vertical']})"
                                print(message)
                                time_tracking_window.add_log(message)
                        
                        elif label == "bus" and "bus" in selected_objects:
                            if crossed_horizontal and track_id not in tracking_vars['counted_buses_horizontal']:
                                tracking_vars['bus_count_horizontal'] += 1
                                tracking_vars['counted_buses_horizontal'].add(track_id)
                                message = f"🚌 Bus ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['bus_count_horizontal']})"
                                print(message)
                                time_tracking_window.add_log(message)
                            elif crossed_vertical and track_id not in tracking_vars['counted_buses_vertical']:
                                tracking_vars['bus_count_vertical'] += 1
                                tracking_vars['counted_buses_vertical'].add(track_id)
                                message = f"🚌 Bus ID {track_id} crossed VERTICAL edge (count: {tracking_vars['bus_count_vertical']})"
                                print(message)
                                time_tracking_window.add_log(message)
                        
                        elif label == "person" and "person" in selected_objects:
                            if crossed_horizontal and track_id not in tracking_vars['counted_people_horizontal']:
                                tracking_vars['person_count_horizontal'] += 1
                                tracking_vars['counted_people_horizontal'].add(track_id)
                                message = f"🚶 Person ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['person_count_horizontal']})"
                                print(message)
                                time_tracking_window.add_log(message)
                            elif crossed_vertical and track_id not in tracking_vars['counted_people_vertical']:
                                tracking_vars['person_count_vertical'] += 1
                                tracking_vars['counted_people_vertical'].add(track_id)
                                message = f"🚶 Person ID {track_id} crossed VERTICAL edge (count: {tracking_vars['person_count_vertical']})"
                                print(message)
                                time_tracking_window.add_log(message)

                # Time tracking for all selected objects
                if label in selected_objects:
                    entry_times_key = f'{label}_entry_times'
                    warned_10_key = f'{label}_warned_10'
                    warned_20_key = f'{label}_warned_20'
                    warned_30_key = f'{label}_warned_30'
                    
                    if currently_inside:
                        if track_id not in tracking_vars[entry_times_key]:
                            tracking_vars[entry_times_key][track_id] = time.time()
                            time_tracking_window.add_log(f"🔵 {label.capitalize()} ID {track_id} entered ROI")
                        else:
                            elapsed = time.time() - tracking_vars[entry_times_key][track_id]
                            if elapsed > 10 and track_id not in tracking_vars[warned_10_key]:
                                message = f"⚠️ {label.capitalize()} ID {track_id} has been in ROI for 10 seconds."
                                print(message)
                                time_tracking_window.add_log(message)
                                tracking_vars[warned_10_key].add(track_id)
                            if elapsed > 20 and track_id not in tracking_vars[warned_20_key]:
                                message = f"⚠️ {label.capitalize()} ID {track_id} has been in ROI for 20 seconds."
                                print(message)
                                time_tracking_window.add_log(message)
                                tracking_vars[warned_20_key].add(track_id)
                            if elapsed > 30 and track_id not in tracking_vars[warned_30_key]:
                                message = f"⚠️ {label.capitalize()} ID {track_id} has been in ROI for 30 seconds."
                                print(message)
                                time_tracking_window.add_log(message)
                                tracking_vars[warned_30_key].add(track_id)
                    else:
                        if track_id in tracking_vars[entry_times_key]:
                            elapsed = time.time() - tracking_vars[entry_times_key][track_id]
                            message = f"✅ {label.capitalize()} ID {track_id} spent {elapsed:.2f} seconds in ROI."
                            print(message)
                            time_tracking_window.add_log(message)
                            del tracking_vars[entry_times_key][track_id]
                            # Remove from warning sets when object exits
                            tracking_vars[warned_10_key].discard(track_id)
                            tracking_vars[warned_20_key].discard(track_id)
                            tracking_vars[warned_30_key].discard(track_id)

                tracking_vars['cars_last_positions'][track_id] = centroid

        # Draw tracking boxes and IDs for selected objects only
        for track in last_detections:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
                
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class()
            
            # Only draw if object is in selected objects
            if label not in selected_objects:
                continue
            
            # Color coding: cars = green, trucks = red, buses = orange, people = blue
            if label == "car":
                color = (0, 255, 0)  # Green
            elif label == "truck":
                color = (0, 0, 255)  # Red
            elif label == "bus":
                color = (0, 165, 255)  # Orange
            else:  # person
                color = (255, 0, 0)  # Blue
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Display counters - only for selected objects
        counter_height = 40 + (len(selected_objects) * 13)  # Dynamic height based on selected objects
        y_offset = 30
        x_offset = display_frame.shape[1] - 180
        cv2.rectangle(display_frame, (x_offset - 10, 10), (display_frame.shape[1] - 10, counter_height), (0, 0, 0), -1)

        line_offset = 0
        if "car" in selected_objects:
            cv2.putText(display_frame, f"Car H: {tracking_vars['car_count_horizontal']} | V: {tracking_vars['car_count_vertical']}", 
                       (x_offset, y_offset + line_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            line_offset += 20
        
        if "truck" in selected_objects:
            cv2.putText(display_frame, f"Truck H: {tracking_vars['truck_count_horizontal']} | V: {tracking_vars['truck_count_vertical']}", 
                       (x_offset, y_offset + line_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            line_offset += 20
        
        if "bus" in selected_objects:
            cv2.putText(display_frame, f"Bus H: {tracking_vars['bus_count_horizontal']} | V: {tracking_vars['bus_count_vertical']}", 
                       (x_offset, y_offset + line_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            line_offset += 20
        
        if "person" in selected_objects:
            cv2.putText(display_frame, f"Person H: {tracking_vars['person_count_horizontal']} | V: {tracking_vars['person_count_vertical']}", 
                       (x_offset, y_offset + line_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            line_offset += 20
        
        # Show instruction and performance info
        cv2.putText(display_frame, "Press '4' to reset ROI | 's' to toggle detection | 't' for time window", 
                   (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        detection_status = "ON" if detection_enabled else "OFF"
        time_window_status = "OPEN" if time_tracking_window.is_open else "CLOSED"
        cv2.putText(display_frame, f"Detection: {detection_status} | Time Window: {time_window_status}", 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (0, 255, 0) if detection_enabled else (0, 0, 255), 1)

        cv2.imshow(window_name, display_frame)
        
        # Non-blocking key check for better responsiveness
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("🛑 Quitting...")
            time_tracking_window.add_log("🛑 Tracking session ended")
            break
        elif key == ord('4'):
            print("🔁 Resetting ROI selection and all counters...")
            tracking_vars = reset_all_counters()
            last_detections = []
            print("✅ Reset complete. Please select 4 new ROI points.")
        elif key == ord('s'):
            detection_enabled = not detection_enabled
            status = "enabled" if detection_enabled else "disabled"
            message = f"🔄 Detection {status} - {'Higher accuracy' if detection_enabled else 'Higher FPS'}"
            print(message)
            time_tracking_window.add_log(message)
        elif key == ord('t'):
            time_tracking_window.toggle()
            status = "opened" if time_tracking_window.is_open else "closed"
            print(f"🪟 Time tracking window {status}")

        # Improved frame rate control to prevent jumping
        elapsed = time.time() - start_time
        
        if elapsed < frame_time * 0.5:
            sleep_time = frame_time - elapsed
            time.sleep(sleep_time)
        elif elapsed > frame_time * 2:
            skip_frames = min(skip_frames + 1, 3)
            for _ in range(skip_frames):
                cap.grab()
        else:
            skip_frames = max(0, skip_frames - 1)

    cap.release()
    cv2.destroyAllWindows()
    time_tracking_window.close_window()

if __name__ == "__main__":
    # Create and run the main menu
    menu = MainMenu()
    menu.run()