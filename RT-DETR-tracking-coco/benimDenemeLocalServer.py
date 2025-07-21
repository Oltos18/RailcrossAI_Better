import os
import uuid
import torch
import numpy as np
from PIL import Image
import supervision as sv
import gradio as gr
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import yt_dlp
import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

# Annotators and tracker
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
TRACKER = sv.ByteTrack()

# Get streamable video URL from YouTube
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        "quiet": True,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "simulate": True,
        "forceurl": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

# Set how much of the livestream to process (default: 2 seconds worth of frames)
def calculate_end_frame_index(source_video_path):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return min(video_info.total_frames, video_info.fps * 2)  # ~2 seconds

# Object detection query
def query(image, confidence_threshold):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes
    )
    return results

# Annotate frame
def annotate_image(input_image, detections, labels) -> np.ndarray:
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image

# Main processing function
@spaces.GPU
def process_video(youtube_url, confidence_threshold, progress=gr.Progress(track_tqdm=True)):
    stream_url = get_youtube_stream_url(youtube_url)
    video_info = sv.VideoInfo.from_video_path(stream_url)
    total = calculate_end_frame_index(stream_url)
    
    frame_generator = sv.get_video_frames_generator(
        source_path=stream_url,
        end=total
    )

    result_file_name = f"{uuid.uuid4()}.mp4"
    result_file_path = os.path.join("./", result_file_name)

    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for _ in tqdm(range(total), desc="Processing video.."):
            frame = next(frame_generator)
            results = query(Image.fromarray(frame), confidence_threshold)
            detections = sv.Detections.from_transformers(results[0])
            detections = TRACKER.update_with_detections(detections)
            labels = [model.config.id2label[label] for label in detections.class_id.tolist()]
            annotated_frame = annotate_image(frame, detections, labels)
            sink.write_frame(annotated_frame)

    return result_file_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Real-Time Object Tracking from YouTube Live (RT-DETR)")
    gr.Markdown("Paste a YouTube livestream URL and process the first few seconds with RT-DETR.")
    
    with gr.Row():
        with gr.Column():
            youtube_url_input = gr.Textbox(label='YouTube Livestream URL', placeholder="https://www.youtube.com/watch?v=...")
            conf = gr.Slider(label="Confidence Threshold", minimum=0.1, maximum=1.0, value=0.6, step=0.05)
            submit = gr.Button("Run Tracking")
        with gr.Column():
            output_video = gr.Video(label='Output Video')

    submit.click(
        fn=process_video,
        inputs=[youtube_url_input, conf],
        outputs=output_video
    )

demo.launch(show_error=True)
