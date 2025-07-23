import copy
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import create_mask

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location='cpu')['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 设置相连帧数
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        # 初始化帧存储变量
        # 高分辨率帧存储列表
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # 存放缩放后帧的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (W_ori, split_h))  # 将补全帧缩放回原大小
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转换颜色空间
                    # 获取遮罩区域并进行图像合成
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # 取出遮罩区域
                    # 实现遮罩区域内的图像融合
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # 把特征张量转移到指定的设备（CPU或GPU）
        feats = feats.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # 获取重绘区域
        # 在设定的邻居帧步幅内循环处理视频
        for f in range(0, frame_length, self.neighbor_stride):
            # 计算邻近帧的ID
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # 获取参考帧的索引
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # 同样关闭梯度计算
            with torch.no_grad():
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # 将预测的特征通过解码器生成图片，并应用激活函数tanh，然后分离出张量
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # 将预测的图片转换为无符号8位整数格式
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # 如果该位置为空，则赋值为新计算出的图片
                        comp_frames[idx] = img
                    else:
                        # 如果此位置之前已有图片，则将新旧图片混合以提高质量
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        获取字幕去除区域，根据mask来确定需要填补的区域和高度
        """
        # 存储绘画区域的列表
        inpaint_area = []
        # 从视频底部的字幕位置开始，假设字幕通常位于底部
        to_H = from_H = H
        # 从底部向上遍历遮罩
        while from_H != 0:
            if to_H - h < 0:
                # 如果下一段会超出顶端，则从顶端开始
                from_H = 0
                to_H = h
            else:
                # 确定段的上边界
                from_H = to_H - h
            # 检查当前段落是否包含遮罩像素
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # 如果不是第一个段落，向下移动以确保没遗漏遮罩区域
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # 确保没有越过底部
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                # 将该段落添加到列表中
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            # 移动到下一个段落
            to_H -= h
        return inpaint_area  # 返回绘画区域列表

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        # 存储结果的列表
        inpaint_area = []
        # 计算并存储标准区间
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        # 检查最后一个区间是否达到了最大值
        if inpaint_area[-1][1] != ymax:
            # 如果没有，则创建一个新的区间，开始于最后一个区间的结束，结束于扩大后的值
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append((inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area  # 返回绘画区域列表


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        # 使用opencv读取视频
        reader = cv2.VideoCapture(self.video_path)
        # 获取视频的宽度, 高度, 帧率和帧数信息并存储在frame_info字典中
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # 视频的原始宽度
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # 视频的原始高度
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 视频的帧率
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 视频的总帧数
        }
        # 返回视频读取对象、帧信息和视频写入对象
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None, subtitle_areas=None, frame_intervals=None):
        # STTNInpaint视频修复实例初始化
        self.sttn_inpaint = STTNInpaint()
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        # 新增：字幕区域和帧区间
        self.subtitle_areas = subtitle_areas
        self.frame_intervals = frame_intervals
        # 设置输出视频文件的路径
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # 配置可在一次处理中加载的最大帧数
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
            else:
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))

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
                for idx, (frames_to_inpaint, valid_indices) in enumerate(zip(interval_batches, interval_indices)):
                    if not frames_to_inpaint:
                        continue
                    area = self.subtitle_areas[idx]
                    s, e = intervals_0_based[idx]
                    print(f"[STTN] Start processing frames {s} to {e + 1} (interval {idx+1}/{len(self.frame_intervals)})")
                    mask_size = (frame_info['H_ori'], frame_info['W_ori'])
                    print(f"[STTN] Mask size: {mask_size}, inpainting area: {area}")
                    mask = create_mask(mask_size, [area])
                    if mask.ndim == 2:
                        mask = mask[:, :, None]
                    inpainted_frames = self.sttn_inpaint(frames_to_inpaint, mask)
                    for j, i_frame in enumerate(valid_indices):
                        # Store the inpainted frame in the dictionary
                        inpainted_dict[i_frame] = inpainted_frames[j]
                    frames_processed += len(valid_indices)
                    print(f"[STTN] Finished interval {idx+1}/{len(self.frame_intervals)}: processed {frames_processed} frames so far.")

            # Now, write all frames in original order, using inpainted frames where available
            for i in range(total_frames):
                if all_frames[i] is None:
                    if show_tqdm:
                        pbar.update(1)
                    continue
                if i in inpainted_dict:
                    frame = inpainted_dict[i]
                else:
                    frame = all_frames[i]
                writer.write(frame)
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
    # 记录开始时间
    start = time.time()
    sttn_video_inpaint = STTNVideoInpaint(video_path, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')