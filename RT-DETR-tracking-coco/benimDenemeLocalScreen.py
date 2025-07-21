import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import supervision as sv
import yt_dlp
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

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

def mouse_callback(event, x, y, flags, param):
    global roi_points, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
        print(f"Point {len(roi_points)} selected: {(x, y)}")
        if len(roi_points) == 4:
            roi_selected = True
            print("ROI selection complete.")

def point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def is_left_of_line(p0, p1, pt):
    # Returns True if pt is left of the directed line from p0 to p1
    return ((p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0])) > 0

def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': 'best[height<=480]/best',  # Even lower resolution for stability
        'http_chunk_size': 10485760,  # 10MB chunks for better streaming
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        if 'url' in info:
            return info['url']
        if 'formats' in info:
            # Prefer formats with lower latency and smaller size
            for f in sorted(info['formats'], key=lambda x: x.get('height', 0)):
                if f.get('protocol') in ['https', 'http', 'm3u8_native', 'm3u8'] and f.get('height', 0) <= 480:
                    return f.get('url')
        return None

def reset_all_counters():
    """Reset all tracking and counting variables"""
    global roi_points, roi_selected
    roi_points = []
    roi_selected = False
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
        'car_warned_10': set(),
        'car_warned_20': set(),
        'car_warned_30': set(),
        'truck_warned_10': set(),
        'truck_warned_20': set(),
        'truck_warned_30': set(),
        'bus_warned_10': set(),
        'bus_warned_20': set(),
        'bus_warned_30': set(),
        'person_count_horizontal': 0,
        'person_count_vertical': 0,
        'car_count_horizontal': 0,
        'car_count_vertical': 0,
        'truck_count_horizontal': 0,
        'truck_count_vertical': 0,
        'bus_count_horizontal': 0,
        'bus_count_vertical': 0
    }

def run_tracking_with_counting(youtube_url, confidence_threshold=0.6, frame_skip=3, resize_width=480):
    global roi_points, roi_selected

    stream_url = get_youtube_stream_url(youtube_url)
    if not stream_url:
        print("❌ Could not extract stream URL.")
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Could not open video stream.")
        return

    print("Instructions:")
    print(" - Click 4 points on the video to define counting polygon.")
    print(" - Press '4' to redefine ROI anytime.")
    print(" - Press 'q' to quit.")
    print(" - Press 's' to toggle detection skip for higher FPS.")
    print("\nCounting Logic:")
    print(" - Horizontal: Counts objects crossing top/bottom edges of polygon")
    print(" - Vertical: Counts objects crossing left/right edges of polygon")
    print("\nOptimizations applied for better real-time performance!")

    detection_enabled = True

    window_name = "RT-DETR + DeepSORT Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.setMouseCallback(window_name, mouse_callback)

    last_detections = []
    frame_count = 0
    last_detection_time = 0
    detection_interval = 0.15  # Slightly slower detection for more stable stream
    skip_frames = 0  # Counter for frame skipping to prevent buffering

    # Initialize all tracking variables
    tracking_vars = reset_all_counters()

    # Set buffer size to reduce latency and prevent frame accumulation
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Additional settings to reduce latency
    cap.set(cv2.CAP_PROP_FPS, 25)  # Lower FPS for more stability
    
    # Try to set fourcc for better compatibility
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    
    print(f"📺 Stream FPS: {fps}")
    
    # More aggressive frame rate control to prevent jumping
    target_fps = 20  # Lower target FPS for smoother experience
    frame_time = 1.0 / target_fps
    
    last_frame_time = time.time()

    while True:
        start_time = time.time()
        
        # Skip frames if we're falling behind to prevent buffering jumps
        ret = False
        frame = None
        max_read_attempts = 3
        
        for _ in range(max_read_attempts):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.01)  # Small delay before retry
        
        if not ret:
            print("❌ Failed to grab frame after multiple attempts.")
            # Try to recover by clearing buffer
            for _ in range(5):
                cap.grab()
            continue

        frame_count += 1
        display_frame = frame.copy()

        # Clear buffer periodically to prevent frame accumulation and jumping
        current_time = time.time()
        if current_time - last_frame_time > 0.5:  # If more than 500ms since last frame
            # Clear any accumulated frames
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

        # Object detection and tracking - time-based instead of frame-based
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
                if label_name in ["car", "person", "truck", "bus"]:
                    x1, y1, x2, y2 = box.tolist()

                    # Scale back to original frame size
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

        # Counting logic based on crossing polygon edges
        if len(roi_points) == 4:
            p0, p1, p2, p3 = roi_points  # polygon points in order

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
                        # Check which edges were crossed:

                        # For Horizontal counting: top (p0-p1) and bottom (p3-p2) edges
                        crossed_horizontal = False
                        last_side_top = is_left_of_line(p0, p1, last_pos)
                        curr_side_top = is_left_of_line(p0, p1, centroid)
                        last_side_bottom = is_left_of_line(p3, p2, last_pos)
                        curr_side_bottom = is_left_of_line(p3, p2, centroid)
                        if last_side_top != curr_side_top or last_side_bottom != curr_side_bottom:
                            crossed_horizontal = True

                        # For Vertical counting: left (p0-p3) and right (p1-p2) edges
                        crossed_vertical = False
                        last_side_left = is_left_of_line(p0, p3, last_pos)
                        curr_side_left = is_left_of_line(p0, p3, centroid)
                        last_side_right = is_left_of_line(p1, p2, last_pos)
                        curr_side_right = is_left_of_line(p1, p2, centroid)
                        if last_side_left != curr_side_left or last_side_right != curr_side_right:
                            crossed_vertical = True

                        if label == "car":
                            if crossed_horizontal and track_id not in tracking_vars['counted_cars_horizontal']:
                                tracking_vars['car_count_horizontal'] += 1
                                tracking_vars['counted_cars_horizontal'].add(track_id)
                                print(f"🚗 Car ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['car_count_horizontal']})")
                            elif crossed_vertical and track_id not in tracking_vars['counted_cars_vertical']:
                                tracking_vars['car_count_vertical'] += 1
                                tracking_vars['counted_cars_vertical'].add(track_id)
                                print(f"🚗 Car ID {track_id} crossed VERTICAL edge (count: {tracking_vars['car_count_vertical']})")
                        elif label == "truck":
                            if crossed_horizontal and track_id not in tracking_vars['counted_trucks_horizontal']:
                                tracking_vars['truck_count_horizontal'] += 1
                                tracking_vars['counted_trucks_horizontal'].add(track_id)
                                print(f"🚚 Truck ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['truck_count_horizontal']})")
                            elif crossed_vertical and track_id not in tracking_vars['counted_trucks_vertical']:
                                tracking_vars['truck_count_vertical'] += 1
                                tracking_vars['counted_trucks_vertical'].add(track_id)
                                print(f"🚚 Truck ID {track_id} crossed VERTICAL edge (count: {tracking_vars['truck_count_vertical']})")
                        elif label == "bus":
                            if crossed_horizontal and track_id not in tracking_vars['counted_buses_horizontal']:
                                tracking_vars['bus_count_horizontal'] += 1
                                tracking_vars['counted_buses_horizontal'].add(track_id)
                                print(f"🚌 Bus ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['bus_count_horizontal']})")
                            elif crossed_vertical and track_id not in tracking_vars['counted_buses_vertical']:
                                tracking_vars['bus_count_vertical'] += 1
                                tracking_vars['counted_buses_vertical'].add(track_id)
                                print(f"🚌 Bus ID {track_id} crossed VERTICAL edge (count: {tracking_vars['bus_count_vertical']})")
                        elif label == "person":
                            if crossed_horizontal and track_id not in tracking_vars['counted_people_horizontal']:
                                tracking_vars['person_count_horizontal'] += 1
                                tracking_vars['counted_people_horizontal'].add(track_id)
                                print(f"🚶 Person ID {track_id} crossed HORIZONTAL edge (count: {tracking_vars['person_count_horizontal']})")
                            elif crossed_vertical and track_id not in tracking_vars['counted_people_vertical']:
                                tracking_vars['person_count_vertical'] += 1
                                tracking_vars['counted_people_vertical'].add(track_id)
                                print(f"🚶 Person ID {track_id} crossed VERTICAL edge (count: {tracking_vars['person_count_vertical']})")

                # Time tracking for vehicles in ROI
                if label in ["car", "truck", "bus"]:
                    entry_times_key = f'{label}_entry_times'
                    warned_10_key = f'{label}_warned_10'
                    warned_20_key = f'{label}_warned_20'
                    warned_30_key = f'{label}_warned_30'
                    
                    if currently_inside:
                        if track_id not in tracking_vars[entry_times_key]:
                            tracking_vars[entry_times_key][track_id] = time.time()
                        else:
                            elapsed = time.time() - tracking_vars[entry_times_key][track_id]
                            if elapsed > 10 and track_id not in tracking_vars[warned_10_key]:
                                print(f"⚠️ {label.capitalize()} ID {track_id} has been in ROI for 10 seconds.")
                                tracking_vars[warned_10_key].add(track_id)
                            if elapsed > 20 and track_id not in tracking_vars[warned_20_key]:
                                print(f"⚠️ {label.capitalize()} ID {track_id} has been in ROI for 20 seconds.")
                                tracking_vars[warned_20_key].add(track_id)
                            if elapsed > 30 and track_id not in tracking_vars[warned_30_key]:
                                print(f"⚠️ {label.capitalize()} ID {track_id} has been in ROI for 30 seconds.")
                                tracking_vars[warned_30_key].add(track_id)
                    else:
                        if track_id in tracking_vars[entry_times_key]:
                            elapsed = time.time() - tracking_vars[entry_times_key][track_id]
                            print(f"✅ {label.capitalize()} ID {track_id} spent {elapsed:.2f} seconds in ROI.")
                            del tracking_vars[entry_times_key][track_id]
                            # Remove from warning sets when vehicle exits
                            tracking_vars[warned_10_key].discard(track_id)
                            tracking_vars[warned_20_key].discard(track_id)
                            tracking_vars[warned_30_key].discard(track_id)

                tracking_vars['cars_last_positions'][track_id] = centroid

        # Draw tracking boxes and IDs
        for track in last_detections:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
                
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class()
            
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

        # Display counters - Fixed size to remove unnecessary black space
        counter_height = 180  # Exact height needed for all counters
        y_offset = 30
        x_offset = display_frame.shape[1] - 150
        cv2.rectangle(display_frame, (x_offset - 10, 10), (display_frame.shape[1] - 10, counter_height), (0, 0, 0), -1)

        cv2.putText(display_frame, f"Car Horizontal: {tracking_vars['car_count_horizontal']}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Car Vertical: {tracking_vars['car_count_vertical']}", (x_offset, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Truck Horizontal: {tracking_vars['truck_count_horizontal']}", (x_offset, y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Truck Vertical: {tracking_vars['truck_count_vertical']}", (x_offset, y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Bus Horizontal: {tracking_vars['bus_count_horizontal']}", (x_offset, y_offset + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Bus Vertical: {tracking_vars['bus_count_vertical']}", (x_offset, y_offset + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Person Horizontal: {tracking_vars['person_count_horizontal']}", (x_offset, y_offset + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Person Vertical: {tracking_vars['person_count_vertical']}", (x_offset, y_offset + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Show instruction and performance info
        cv2.putText(display_frame, "Press '4' to reset ROI | 's' to toggle detection", (10, display_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        detection_status = "ON" if detection_enabled else "OFF"
        cv2.putText(display_frame, f"Detection: {detection_status}", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if detection_enabled else (0, 0, 255), 1)

        cv2.imshow(window_name, display_frame)
        
        # Non-blocking key check for better responsiveness
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("🛑 Quitting...")
            break
        elif key == ord('4'):
            print("🔁 Resetting ROI selection and all counters...")
            tracking_vars = reset_all_counters()
            last_detections = []  # Clear current detections
            print("✅ Reset complete. Please select 4 new ROI points.")
        elif key == ord('s'):
            detection_enabled = not detection_enabled
            status = "enabled" if detection_enabled else "disabled"
            print(f"🔄 Detection {status} - {'Higher accuracy' if detection_enabled else 'Higher FPS'}")

        # Improved frame rate control to prevent jumping
        elapsed = time.time() - start_time
        
        # Only sleep if processing was very fast, otherwise let it run free
        if elapsed < frame_time * 0.5:  # Only if we're running at less than half the time budget
            sleep_time = frame_time - elapsed
            time.sleep(sleep_time)
        
        # Skip multiple frames if we're falling behind (helps prevent jumping)
        elif elapsed > frame_time * 2:  # If processing took more than 2x expected time
            skip_frames = min(skip_frames + 1, 3)
            for _ in range(skip_frames):
                cap.grab()  # Skip frames to catch up
        else:
            skip_frames = max(0, skip_frames - 1)  # Gradually reduce skipping

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=D5kKdEBmrYU"
    run_tracking_with_counting(url, confidence_threshold=0.6)