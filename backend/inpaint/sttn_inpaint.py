import copy
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
import traceback
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import create_mask

# define image preprocessing
_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()
])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        # 1. create InpaintGenerator model instance and load to selected device
        self.model = InpaintGenerator().to(self.device)
        # 2. load pre-trained model weights, load model state dictionary
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location='cpu')['netG'])
        # 3. # set model to evaluation mode
        self.model.eval()
        # 4. model input width and height
        self.model_input_width, self.model_input_height = 640, 120
        # 5. set neighbor frames
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: original frame
        :param mask: mask of subtitle area
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # determine the vertical height of subtitle area
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)

        # initialize frame storage variables
        # high resolution frame storage list
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # store scaled frames
        comps = {}  # store completed frames

        # store final video frames
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []

        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # crop and resize each subtitle area
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)

        # process each subtitle area
        for k in range(len(inpaint_area)):
            try:
                comps[k] = self.inpaint(frames_scaled[k])
            except Exception as e:
                print(f"[ERROR] Exception in self.inpaint(frames_scaled[{k}]): {e}")
                traceback.print_exc()
                raise
        # if there is an area to inpaint
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # get original frame
                # for each subtitle area
                for k in range(len(inpaint_area)):
                    try:
                        comp = cv2.resize(comps[k][j], (W_ori, split_h))
                    except Exception as e:
                        print(f"[ERROR] Exception in cv2.resize: {e}")
                        raise
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # convert color space
                    # get mask area and perform inpainting
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # get mask area
                    # inpaint mask area
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # add final frame to list
                inpainted_frames.append(frame)
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # convert to binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        sample reference frames from the entire video
        """
        # initialize reference frame index list
        ref_index = []
        # iterate over the video length with ref_length
        for i in range(0, length, self.ref_length):
            # if current frame is not in neighbor frames
            if i not in neighbor_ids:
                # add it to reference frame list
                ref_index.append(i)
        # return reference frame index list
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        use STTN to complete inpainting
        """
        frame_length = len(frames)
        # preprocess frames to tensor and normalize
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # transfer feature tensor to selected device (CPU or GPU)
        feats = feats.to(self.device)
        # initialize a list with the same length as the video, for storing processed frames
        comp_frames = [None] * frame_length
            
        # Try to process in smaller batches if we have too many frames
        if frame_length > config.STTN_MAX_LOAD_NUM:
            print(f"[INFO] Processing frames in batches of {config.STTN_MAX_LOAD_NUM}")
            return self._process_in_batches(frames, config.STTN_MAX_LOAD_NUM)
        
        # close gradient calculation, for inference stage to save memory and speed up
        with torch.no_grad():
            # pass processed frames through encoder to generate feature representation
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # get feature dimension information
            _, c, feat_h, feat_w = feats.size()
            # adjust feature shape to match model's expected input
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # get inpainting area
        # iterate over the video with neighbor stride
        for f in range(0, frame_length, self.neighbor_stride):
            # calculate neighbor frame ID
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # get reference frame index
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # close gradient calculation
            with torch.no_grad():
                # pass processed frames through encoder to generate feature representation
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # pass predicted feature through decoder to generate image, apply activation function tanh, then separate tensor
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # rescale result tensor to range 0-255 (image pixel value)
                pred_img = (pred_img + 1) / 2
                # move tensor back to CPU and convert to NumPy array
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # iterate over neighbor frames
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # convert predicted image to unsigned 8-bit integer format
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # if the position is empty, assign the new calculated image
                        comp_frames[idx] = img
                    else:
                        # if the position has an image, mix the new and old images to improve quality
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # return processed frames
        return comp_frames

    def _process_in_batches(self, frames: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        """
        Process frames in smaller batches to avoid memory issues
        """
        print(f"[INFO] Processing {len(frames)} frames in batches of {batch_size}")
        all_processed_frames = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            print(f"[INFO] Processing batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
            try:
                batch_result = self.inpaint(batch_frames)
                all_processed_frames.extend(batch_result)
            except Exception as e:
                print(f"[ERROR] Failed to process batch {i//batch_size + 1}: {e}")
                # If batch processing fails, try with CPU
                print(f"[INFO] Trying with CPU...")
                original_device = self.device
                self.device = torch.device("cpu")
                self.model = self.model.to("cpu")
                try:
                    batch_result = self.inpaint(batch_frames)
                    all_processed_frames.extend(batch_result)
                finally:
                    self.device = original_device
                    self.model = self.model.to(original_device)
        
        return all_processed_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        get subtitle removal area, determine the area and height to fill based on mask
        """
        # store inpainting area list
        inpaint_area = []
        # start from the subtitle position at the bottom of the video, assume subtitle is usually at the bottom
        to_H = from_H = H
        # iterate from bottom to top
        while from_H != 0:
            if to_H - h < 0:
                # if the next segment will exceed the top, start from the top
                from_H = 0
                to_H = h
            else:
                # determine the upper boundary of the segment
                from_H = to_H - h
            # check if the current segment contains mask pixels
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # if not the first segment, move down to ensure no mask area is missed
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # ensure not to exceed the bottom
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                # add the segment to the list
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            # move to the next segment
            to_H -= h
        return inpaint_area  # return inpainting area list

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        # store result list
        inpaint_area = []
        # calculate and store standard interval
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        # check if the last interval reaches the maximum value
        if inpaint_area[-1][1] != ymax:
            # if not, create a new interval, starting from the end of the last interval, ending at the expanded value
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append((inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area  # return inpainting area list


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        # use opencv to read video
        reader = cv2.VideoCapture(self.video_path)
        # get video width, height, frame rate and frame count information and store in frame_info dictionary
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # original width
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # original height
            'fps': reader.get(cv2.CAP_PROP_FPS),  # frame rate
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # total frame count
        }
        # return video reader, frame info and video writer
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None, subtitle_areas=None, frame_intervals=None):
        # initialize STTNInpaint instance
        self.sttn_inpaint = STTNInpaint()
        # video and mask path
        self.video_path = video_path
        self.mask_path = mask_path
        # new: subtitle area and frame interval
        self.subtitle_areas = subtitle_areas
        self.frame_intervals = frame_intervals
        # set output video file path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # set maximum frame count that can be loaded in one processing
        if clip_gap is None:
            self.clip_gap = config.STTN_MAX_LOAD_NUM
        else:
            self.clip_gap = clip_gap

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader = None
        writer = None
        try:
            reader, frame_info = self.read_frame_info_from_video()
            if input_sub_remover is not None:
                writer = input_sub_remover.video_writer
                print(f"[WRITE] Using external writer from remover. target size=({frame_info['W_ori']},{frame_info['H_ori']}), fps={frame_info['fps']}")
            else:
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
                print(f"[WRITE] Created internal writer at {self.video_out_path}. opened={writer.isOpened()} target size=({frame_info['W_ori']},{frame_info['H_ori']}), fps={frame_info['fps']}")

            total_frames = frame_info['len']
            all_frames = []
            for i in range(total_frames):
                success, frame = reader.read()
                if not success:
                    all_frames.append(None)
                else:
                    all_frames.append(frame)

            # Build a list of intervals as (start, end) tuples, 0-based
            intervals_0_based = []
            if self.frame_intervals is not None:
                for interval in self.frame_intervals:
                    s, e = interval
                    s = max(0, int(s) - 1)
                    e = min(total_frames - 1, int(e) - 1)
                    intervals_0_based.append((s, e))

            # Map each frame index to its interval index (if any)
            # This avoids processing frames that are not in any interval (without subtitles)
            frame_to_interval = {}
            if intervals_0_based:
                for idx, (s, e) in enumerate(intervals_0_based):
                    for f in range(s, e + 1):
                        frame_to_interval[f] = idx

            # Prepare inpainting batches for each interval
            interval_batches = [[] for _ in intervals_0_based]
            interval_indices = [[] for _ in intervals_0_based]

            # tqdm bar for overall progress if not provided
            show_tqdm = tbar is None
            if show_tqdm:
                pbar = tqdm(total=total_frames, unit='frame', desc='STTN Subtitle Removal', position=0, file=sys.__stdout__)
            else:
                pbar = None

            # First, collect all frames for inpainting or direct writing
            for i in range(total_frames):
                if all_frames[i] is None:
                    continue
                if i in frame_to_interval:
                    idx = frame_to_interval[i]
                    interval_batches[idx].append(all_frames[i])
                    interval_indices[idx].append(i)

            # Process each interval batch for inpainting and store results in a dict
            inpainted_dict = {}
            frames_processed = 0
            if self.subtitle_areas is not None and self.frame_intervals is not None:
                # Configure explicit progress tracking on the remover (if provided)
                if input_sub_remover is not None:
                    total_to_inpaint = sum(len(frames) for frames in interval_batches)
                    input_sub_remover.total_inpaint_frames = total_to_inpaint
                    input_sub_remover.processed_inpaint_frames = 0
                for idx, (frames_to_inpaint, valid_indices) in enumerate(zip(interval_batches, interval_indices)):
                    if not frames_to_inpaint:
                        continue
                    area = self.subtitle_areas[idx]
                    s, e = intervals_0_based[idx]
                    print(f"[STTN] Start processing frames {s} to {e + 1} (interval {idx+1}/{len(self.frame_intervals)})")
                    mask_size = (frame_info['H_ori'], frame_info['W_ori'])
                    print(f"[STTN] Mask size: {mask_size}, inpainting area: {area}")
                    mask = create_mask(mask_size, [area])

                    # Convert mask to 3-channel format if needed
                    # The 3-color-channel format is required for the inpaint function to color images
                    if mask.ndim == 2:
                        mask = mask[:, :, None]
                    
                    # Process frames in batches to avoid memory issues
                    batch_size = config.STTN_MAX_LOAD_NUM
                    inpainted_frames = []
                    
                    for batch_start in range(0, len(frames_to_inpaint), batch_size):
                        batch_end = min(batch_start + batch_size, len(frames_to_inpaint))
                        batch_frames = frames_to_inpaint[batch_start:batch_end]
                        print(f"[STTN] Processing batch {batch_start//batch_size + 1}/{(len(frames_to_inpaint) + batch_size - 1)//batch_size} ({len(batch_frames)} frames)")
                        
                        try:
                            batch_result = self.sttn_inpaint(batch_frames, mask)
                            inpainted_frames.extend(batch_result)
                        except Exception as e:
                            print(f"[ERROR] Failed to process batch: {e}")
                            print(f"[INFO] Trying with CPU...")
                            # Fallback to CPU if GPU fails
                            original_device = self.sttn_inpaint.device
                            self.sttn_inpaint.device = torch.device("cpu")
                            self.sttn_inpaint.model = self.sttn_inpaint.model.to("cpu")
                            try:
                                batch_result = self.sttn_inpaint(batch_frames, mask)
                                inpainted_frames.extend(batch_result)
                            finally:
                                self.sttn_inpaint.device = original_device
                                self.sttn_inpaint.model = self.sttn_inpaint.model.to(original_device)
                    
                    for j, i_frame in enumerate(valid_indices):
                        if j < len(inpainted_frames):
                            # Store the inpainted frame in the dictionary
                            inpainted_dict[i_frame] = inpainted_frames[j]
                        else:
                            # if frame is not processed, use original frame
                            print(f"[WARNING] Frame {i_frame} not processed - using original frame")
                            inpainted_dict[i_frame] = all_frames[i_frame]
                    frames_in_interval = len(valid_indices)
                    frames_processed += frames_in_interval
                    # Update real progress based on inpainted frames, not final writing
                    if input_sub_remover is not None:
                        input_sub_remover.update_progress(tbar, increment=frames_in_interval)
                    print(f"[STTN] Finished interval {idx+1}/{len(self.frame_intervals)}: processed {frames_processed} frames so far.")

            # Now, write all frames in original order, using inpainted frames where available
            # This ensures the output video has the same frame order as the input video
            for i in range(total_frames):
                if all_frames[i] is None:
                    if show_tqdm:
                        pbar.update(1)
                    continue
                if i in inpainted_dict:
                    frame = inpainted_dict[i]
                else:
                    frame = all_frames[i]
                # Debug: validate frame before writing
                try:
                    if frame is None:
                        print(f"[WRITE][WARN] Frame {i} is None, skipping write")
                        if show_tqdm:
                            pbar.update(1)
                        continue
                    h, w = frame.shape[:2]
                    if (w != frame_info['W_ori']) or (h != frame_info['H_ori']):
                        print(f"[WRITE][WARN] Frame {i} size mismatch. expected=({frame_info['W_ori']},{frame_info['H_ori']}) got=({w},{h})")
                    if not writer or not writer.isOpened():
                        print(f"[WRITE][ERROR] Writer not opened at frame {i}")
                    writer.write(frame)
                except Exception as e:
                    print(f"[WRITE][ERROR] Exception writing frame {i}: {e}")
                    # continue with next frame
                if input_sub_remover is not None:
                    if tbar is not None:
                        input_sub_remover.update_progress(tbar, increment=1)
                    if input_sub_remover.gui_mode:
                        input_sub_remover.preview_frame = cv2.hconcat([all_frames[i], frame])
                if show_tqdm:
                    pbar.update(1)
            if show_tqdm:
                pbar.close()
            print(f"[STTN] All frames processed and written to output video. Total frames: {total_frames}")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
        finally:
            if writer:
                writer.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    
    # record start time
    start = time.time()
    sttn_video_inpaint = STTNVideoInpaint(video_path, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')