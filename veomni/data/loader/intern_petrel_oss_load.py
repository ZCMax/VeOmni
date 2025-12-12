# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import random
import re
import time
from typing import Literal

import cv2
import imageio
import numpy as np
from PIL import Image

from veomni.utils.oss_utils import get_oss_backend

from decord import VideoReader
from concurrent.futures import ThreadPoolExecutor

from torchcodec.decoders import VideoDecoder
# import torch




def read_frames_torchcodec(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):

    start_time = time.time()
    oss_read_time = 0

    if "s3://" in video_path:
        assert client is not None, "client must be provided for s3 backend"
        video_bytes = client.get(video_path)  # raw bytes
        oss_read_time = time.time() - start_time
        decoder = VideoDecoder(video_bytes, device="cpu")
        start_time = time.time()
    else:
        try:
            decoder = VideoDecoder(video_path, device="cpu")
        except:
            print(video_path)
        start_time = time.time()


    meta = decoder.metadata
    vlen = meta.num_frames
    fps = meta.average_fps
    # duration = meta.duration      # seconds

    if clip is not None:
        start, end = clip
        start_index = int(start * fps)
        end_index = int(end * fps)
        vlen = end_index - start_index
    else:
        start_index = 0


    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps
    )
    frame_indices = [i + start_index for i in frame_indices]

    start_decode_time = time.time()
    frame_batch = decoder.get_frames_at(frame_indices)   # 返回 FrameBatch(T, C, H, W)
    video_get_batch_time = time.time() - start_decode_time

    frames_tensor = frame_batch.data  # torch.uint8 tensor, (T, C, H, W)

    frames = []
    for i in range(frames_tensor.shape[0]):
        chw = frames_tensor[i]
        hwc = chw.permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray(hwc))

    return frames, oss_read_time, video_get_batch_time, meta.num_frames


def read_single_frame(fp):

    frame = cv2.imread(fp)
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def load_video_fast(video_path, max_workers=8):
    image_list = sort_frames(list(os.listdir(video_path)))
    
    file_paths = [os.path.join(video_path, image) for image in image_list]
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(read_single_frame, file_paths))
        
    read_time = time.time() - start_time
    
    frames = [f for f in frames if f is not None]
    vlen = len(frames)

    return frames, read_time, vlen

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert("RGB")


def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r"_(\d+).jpg$", filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if 0 < max_num_frames < len(frame_indices):
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_folder(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):
    
    if "s3://" in video_path:
        oss_read_time = 0
        assert client is not None, "client should be provided for s3 backend"
        image_list = sort_frames(client.list(video_path))
        image_list = [os.path.join(video_path.split(image.split("/")[0])[0], image) for image in image_list]
        frames = []
        for image in image_list:
            start_time = time.time()
            image_byte = client.get(image)
            oss_read_time += time.time() - start_time
            frame = Image.open(io.BytesIO(image_byte))
            frames.append(frame)
        read_time = oss_read_time
    else:
        frames, read_time, vlen = load_video_fast(video_path)
    vlen = len(frames)

    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    return frames, read_time, vlen


def read_frames_gif(
    video_path, num_frames, sample="rand", fix_start=None, client=None, min_num_frames=4, random_frame_num=None
):
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)

    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):
    decord_video_threads = int(os.getenv("XTUNER_DECORD_VIDEO_THREADS", 2))
    start_time = time.time()
    oss_read_time = 0
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        video_bytes = client.get(video_path)
        oss_read_time = time.time() - start_time
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=decord_video_threads)
        start_time = time.time()
    else:
        video_reader = VideoReader(video_path, num_threads=decord_video_threads)
        start_time = time.time()
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    video_get_batch_time = time.time() - start_time
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames, oss_read_time, video_get_batch_time, vlen


def read_interns1_vl_video(
    path,
    min_num_frames,
    max_num_frames,
    random_frame_num,
    sample="rand",
    clip=None,
    client=None,
    debug=False,
    time_log_thr=10,
):
    start_time = time.time()
    oss_read_time = 0
    vlen = 0
    video_get_batch_time = 0
    if path.endswith("/"):
        frames, oss_read_time, vlen = read_frames_folder(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            random_frame_num=random_frame_num,
        )
    elif path.endswith(".gif"):
        frames = read_frames_gif(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            random_frame_num=random_frame_num,
        )
    elif (
        path.endswith(".mp4")
        or path.endswith(".avi")
        or path.endswith(".mov")
        or path.endswith(".webm")
        or path.endswith(".flv")
        or path.endswith(".wmv")
        or path.endswith(".mkv")
        or path.endswith(".rmvb")
        or path.endswith(".ts")
    ):
        frames, oss_read_time, video_get_batch_time, vlen = read_frames_decord(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            clip=clip,
            random_frame_num=random_frame_num,
        )
    else:
        raise ValueError(f"Unsupported video format: {path}")
    end_time = time.time() - start_time
    if debug and end_time > time_log_thr:
        print(
            f"[Warning] OSS read video {path} cost {end_time} seconds, "
            f"oss_read_time {oss_read_time}, video_get_batch_time {video_get_batch_time}, vlen {vlen}"
        )
    return frames


class InternS1VLOSSLoader:
    # Singleton instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, backend: Literal["petrel"] = "petrel", debug: bool = False, oss_time_log_thr: int = 10, **kwargs
    ):
        self.client = get_oss_backend(backend, **kwargs)
        self.debug = debug
        self.oss_time_log_thr = oss_time_log_thr

    def __call__(
        self,
        path,
        type="image",
        max_num_frames=-1,
        min_num_frames=8,
        sample="rand",
        clip=None,
        random_frame_num=None,
    ):
        if type == "image":
            start_time = time.time()
            img_value_str = self.client.get(path)
            if self.debug:
                end_time = time.time()
                if end_time - start_time > self.oss_time_log_thr:
                    print(f"[Warning] OSS read one image {path} cost {end_time - start_time} seconds")
            img = pil_loader(img_value_str)
            return img

        elif type == "video":
            return read_interns1_vl_video(
                path,
                min_num_frames,
                max_num_frames,
                random_frame_num=random_frame_num,
                sample=sample,
                clip=clip,
                client=self.client,
                debug=self.debug,
                oss_time_log_thr=self.oss_time_log_thr,
            )


class InternS1VLLocalLoader:
    # Singleton instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, debug: bool = False, time_log_thr: int = 10, **kwargs
    ):
        self.debug = debug
        self.time_log_thr = time_log_thr

    def __call__(
        self,
        path,
        type="image",
        max_num_frames=-1,
        min_num_frames=8,
        sample="rand",
        clip=None,
        random_frame_num=None,
    ):
        if type == "image":
            start_time = time.time()

            img = Image.open(path)
            if img.mode == "P":
                img = img.convert("RGBA")
            img = img.convert("RGB")

            if self.debug:
                end_time = time.time()
                if end_time - start_time > self.time_log_thr:
                    print(f"[Warning] OSS read one image {path} cost {end_time - start_time} seconds")
            return img

        elif type == "video":

            return read_interns1_vl_video(
                path,
                min_num_frames,
                max_num_frames,
                random_frame_num=random_frame_num,
                sample=sample,
                clip=clip,
                debug=self.debug,
                time_log_thr=self.time_log_thr,
            )


if __name__ == "__main__":
    video_path = '/mnt/inspurfs/eb3d_t/share/datasets/internvl3_5_tiny_media/P~Video_Video_LongCap~en~llava_video_academic_cap_1_2_m_en_20241008_tiny_final~1.0.0~0.0/multimodal_elements/ego4d/f6907d11-742d-4790-953e-caf327dd480c.mp4'
    read_frames_decord(video_path, 128,)