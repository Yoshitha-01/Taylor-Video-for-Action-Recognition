import cv2
import os
import torch
import math
import torchvision

def preprocess_tensor(tensor):
    tensor[tensor < 0] = 0
    max_val = tensor.max()
    if max_val > 0:
        tensor = (tensor * 255 / max_val).to(torch.uint8)
    else:
        tensor = tensor.to(torch.uint8)
    return tensor

def update_difference_list(full_difference_list, frame, terms):
    full_difference_list = torch.roll(full_difference_list, -1, 1)
    norm_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).float() / 255.0
    full_difference_list[0, -1, :, :] = norm_frame
    for i in range(1, terms + 3):
        full_difference_list[i, -1 - i, :, :] = (
            full_difference_list[i - 1, -i, :, :] - full_difference_list[i - 1, -1 - i, :, :]
        )
    return full_difference_list

def video_convert(input_path, output_path, terms):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video file {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Length: {frame_count}, FPS: {fps}")

    dynamic_temporal_length = max(terms + 3, math.ceil(fps / 2))
    print(f"Dynamic Temporal Length: {dynamic_temporal_length}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read the first frame.")
    
    gray_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).float() / 255.0
    h, w = gray_frame.shape

    full_difference_list = torch.zeros((terms + 3, dynamic_temporal_length, h, w), dtype=torch.float64)
    full_difference_list[0, 0, :, :] = gray_frame

    for i in range(1, dynamic_temporal_length):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Frame {i} could not be read.")
            break
        gray_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).float() / 255.0
        full_difference_list[0, i, :, :] = gray_frame

    output_video = torch.zeros((frame_count - dynamic_temporal_length + 1, h, w, 3), dtype=torch.uint8)

    for seq in range(frame_count - dynamic_temporal_length + 1):
        if seq != 0:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Frame {seq} could not be read.")
                break
            full_difference_list = update_difference_list(full_difference_list, frame, terms)

        t_sums = [torch.zeros((h, w), dtype=torch.float64) for _ in range(3)]

        dummy = full_difference_list[0, :, :, :].unsqueeze(0).repeat(terms + 3, 1, 1, 1)
        xa_tensor = full_difference_list[0, :, :, :].unsqueeze(0).repeat(terms + 3, 1, 1, 1) - dummy

        for inc_b in range(terms):
            part = torch.pow(xa_tensor[inc_b], inc_b) / math.factorial(inc_b)
            part_sum = torch.sum(part, 0)  
            for i in range(3):
                t_sums[i] += part_sum * full_difference_list[inc_b + i + 1, 0, :, :]

        for i, t_sum in enumerate(t_sums):
            output_video[seq, :, :, i] = preprocess_tensor(t_sum / dynamic_temporal_length)

    cap.release()
    torchvision.io.write_video(filename=output_path, video_array=output_video, fps=fps, video_codec="h264")
    print(f"Video saved to {output_path}")

video_convert("brush.mp4", "brush-taylor-dynamic.mp4", terms=3)
