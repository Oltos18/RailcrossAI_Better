from typing import List
import os
import numpy as np
import supervision as sv
import uuid
import torch
from tqdm import tqdm
import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)


BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
TRACKER = sv.ByteTrack()


def calculate_end_frame_index(source_video_path):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return min(video_info.total_frames, video_info.fps * 2)
        


def annotate_image(
    input_image,
    detections,
    labels
) -> np.ndarray:
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


@spaces.GPU
def process_video(
    input_video,
    confidence_threshold,
    progress=gr.Progress(track_tqdm=True)
):
    video_info = sv.VideoInfo.from_video_path(input_video)
    total = calculate_end_frame_index(input_video)
    frame_generator = sv.get_video_frames_generator(
        source_path=input_video,
        end=total
    )

    result_file_name = f"{uuid.uuid4()}.mp4"
    result_file_path = os.path.join("./", result_file_name)
    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for _ in tqdm(range(total), desc="Processing video.."):
            frame = next(frame_generator)
            results = query(Image.fromarray(frame), confidence_threshold)
            final_labels = []
            detections = []
            
            detections = sv.Detections.from_transformers(results[0])
            detections = TRACKER.update_with_detections(detections)
            for label in detections.class_id.tolist():
                final_labels.append(model.config.id2label[label])
            frame = annotate_image(
                input_image=frame,
                detections=detections,
                labels=final_labels,
            )
            sink.write_frame(frame)
    return result_file_path
    

def query(image, confidence_threshold):
  inputs = processor(images=image, return_tensors="pt").to(device)
  with torch.no_grad():
    outputs = model(**inputs)
  target_sizes = torch.tensor([image.size[::-1]])
  
  results = processor.post_process_object_detection(outputs=outputs, threshold=confidence_threshold, target_sizes=target_sizes)
  return results

with gr.Blocks() as demo:
  gr.Markdown("## Real Time Object Tracking with RT-DETR")
  gr.Markdown("This is a demo for object tracking using RT-DETR. It runs on ZeroGPU which captures GPU every first time you infer, so the model is actually faster than the inference in this demo.")
  gr.Markdown("Simply upload a video, you can also play with confidence threshold, or try the example below. 👇")
  with gr.Row():
    with gr.Column():
        input_video = gr.Video(
            label='Input Video'
        )
        conf = gr.Slider(label="Confidence Threshold", minimum=0.1, maximum=1.0, value=0.6, step=0.05)
        submit = gr.Button()
    with gr.Column():
        output_video = gr.Video(
            label='Output Video'
        )
  gr.Examples(
      fn=process_video,
      examples=[["./cat.mp4", 0.6], ["./football.mp4", 0.6]],
      inputs=[
          input_video,
          conf
      ],
      outputs=output_video
  )

  submit.click(
      fn=process_video,
      inputs=[input_video, conf],
      outputs=output_video
  )

demo.launch(show_error=True)