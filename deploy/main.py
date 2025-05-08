# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
from tqdm import tqdm
import warnings
import time
import argparse
import logging
from math import ceil

from utils import *
from config import *

warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
def inference_recodings(edf_path, channel_id, model, args):
    start_time = time.time()
    logger = logging.getLogger(__name__)

    # 读取edf文件
    sig_raw = read_raw_edf(edf_path, include=(), verbose=True)
    all_ch_names = sig_raw.ch_names

    sig, ch_id, meas_date, sample_rate, select_ch_names = load_sig(edf_path, channel_id)  # (TN, cn)
    sig = pre_process(sig, sample_rate)  # (N, 3000, cn)

    logger.info("PSG: %s", edf_path)
    logger.info("Start Date: %s, Duration: %d seconds", meas_date, sig.shape[0] * 30)
    logger.info("All Channels: %s", all_ch_names)
    logger.info("Selected Channels: %s, Channel ID: %s", select_ch_names, ch_id)
    logger.info("Sample Rate: %s", sample_rate)
    logger.info("Shape: %s", sig.shape)

    # 准备输入序列，注意维度变化
    seq = sig.transpose(0, 2, 1)  # (N, cn, 3000)
    seq = np.stack([seq[i:i + args.seq_len] for i in range(len(seq) - args.seq_len + 1)], axis=0)  # (seqn, seql, cn, 3000)
    seq = seq.reshape(-1, args.seq_len * len(ch_id), 3000)  # (seqn, seql*cn, 3000)
    batch_num = ceil(len(seq) / args.batch_size)

    model.eval()
    prediction = []
    for batch_idx in tqdm(range(batch_num), desc="Batch Inference"):
        seq_batch = seq[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        batch_size = len(seq_batch)
        # 构造 seq_idx 和 ch_idx
        seq_idx = np.arange(args.seq_len).reshape(1, args.seq_len, 1)  # (1, seql, 1)
        seq_idx = np.tile(seq_idx, (batch_size, 1, len(ch_id)))  # (batch_size, seql, cn)
        seq_idx = np.reshape(seq_idx, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn, )
        ch_idx = ch_id[np.newaxis, np.newaxis, :]  # (1, 1, cn)
        ch_idx = np.tile(ch_idx, (batch_size, args.seq_len, 1))  # (batch_size, seql, cn)
        ch_idx = np.reshape(ch_idx, (batch_size, args.seq_len * len(ch_id)))  # (batch_size, seql*cn, )
        # 转 tensor 并放到 GPU 上
        seq_batch = torch.tensor(seq_batch, dtype=torch.float32).to(args.device)
        seq_idx = torch.tensor(seq_idx, dtype=torch.int64).to(args.device)
        ch_idx = torch.tensor(ch_idx, dtype=torch.int64).to(args.device)
        mask = torch.zeros((batch_size, args.seq_len * len(ch_id)), dtype=torch.int64).bool().to(args.device)
        logits = model(seq_batch, mask, ch_idx, seq_idx)  # (seqn, seql, 5)
        logits = torch.softmax(logits, dim=-1)  # (seqn, seql, 5)
        logits = logits.cpu().numpy()
        prediction.append(logits)

    prediction = np.concatenate(prediction, axis=0)  # (seqn, seql, 5)
    prediction = sequence_voting(prediction, len(sig), args.seq_len)  # (N, )
    logger.info("Inference Time Cost: %.2f seconds", time.time() - start_time)
    return prediction


def inference_main(args):
    """
    根据命令行参数进行推理：
      - 如果输入路径为单个EDF文件，则进行单文件推理；
      - 如果输入路径为一个目录，则递归搜索所有EDF文件，并对每个文件执行推理，保存输出到指定输出目录。
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting inference with arguments: %s", args)

    # 设备设置
    dir_path = os.path.dirname(os.path.abspath(__file__))   # 当前文件所在目录
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.weights = os.path.join(dir_path, 'weights', 'LPSGM_cuda.pt')
    else:
        args.device = torch.device('cpu')
        args.weights = os.path.join(dir_path, 'weights', 'LPSGM_cpu.pt')
    # args.device = torch.device('cpu')
    # args.weights = os.path.join(dir_path, 'weights', 'LPSGM_cpu.pt')
    logger.info("Inference on device: %s", args.device)

    # 加载模型和权重
    logger.info("Loading model from weights: %s", args.weights)
    model = torch.jit.load(args.weights, map_location=args.device).to(args.device)
    logger.info("Model loaded successfully.")

    total_start_time = time.time()
    if os.path.isfile(args.input):
        # 单个文件模式
        logger.info("Running in single-file mode for EDF file: %s", args.input)
        file_start = time.time()
        pred = inference_recodings(args.input, channel_id, model, args)
        file_time = time.time() - file_start
        logger.info("Finished inference for file: %s (Time: %.2f seconds)", args.input, file_time)
        pred_str = ndarray2str(pred)
        # 判断输出路径：如果输出路径为目录，则自动生成输出文件名
        if os.path.isdir(args.output):
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_file = os.path.join(args.output, base_name + ".txt")
        else:
            output_file = args.output
        np.savetxt(output_file, pred_str, fmt='%s')
        logger.info("Output saved to: %s", output_file)
    elif os.path.isdir(args.input):
        # 批量处理模式：递归搜索 EDF 文件
        logger.info("Running in directory mode for EDF files under: %s", args.input)
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
            logger.info("Created output directory: %s", args.output)
        edf_files = []
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(".edf"):
                    edf_files.append(os.path.join(root, file))
        if not edf_files:
            logger.error("No EDF files found under directory: %s", args.input)
            return
        logger.info("Found %d EDF files for inference.", len(edf_files))
        for edf_file in edf_files:
            logger.info("Processing EDF file: %s", edf_file)
            file_start = time.time()
            try:
                pred = inference_recodings(edf_file, channel_id, model, args)
                file_time = time.time() - file_start
                logger.info("Finished inference for file: %s (Time: %.2f seconds)", edf_file, file_time)
                pred_str = ndarray2str(pred)
                # 根据相对于输入目录的相对路径构造输出路径
                relative_path = os.path.relpath(edf_file, args.input)
                output_file = os.path.join(args.output, os.path.splitext(relative_path)[0] + ".txt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.savetxt(output_file, pred_str, fmt='%s')
                logger.info("Output saved to: %s", output_file)
            except Exception as e:
                logger.error("Error processing file %s: %s", edf_file, e)
    else:
        logger.error("The input path is neither a file nor a directory: %s", args.input)
    total_time = time.time() - total_start_time
    logger.info("Total inference time: %.2f seconds", total_time)

    if args.scoredata_xml:
        out_scoredata_file = output_file.replace(".txt", ".scoredata.xml")
        convert_sleep_stages(output_file, out_scoredata_file, args.scoredata_xml)
        logger.info("Scoredata XML saved to: %s", out_scoredata_file)


def setup_logging(log_file=None):
    """配置日志：输出到控制台，并可选写入到指定日志文件。"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 文件 handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleep Stage Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input EDF file or directory containing EDF files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output text file (for single file) or directory (for batch inference)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional log file path to save logs")
    parser.add_argument("--scoredata_xml", type=str, default=None,
                        help="Optional scoredata for alignment with KangDi software")
    
    # 模型参数
    parser.add_argument("--architecture", type=str, default="cat_cls", help="Model architecture")
    parser.add_argument("--epoch_encoder_dropout", type=float, default=0, help="Epoch encoder dropout")
    parser.add_argument("--transformer_num_heads", type=int, default=8, help="Number of transformer heads")
    parser.add_argument("--transformer_dropout", type=float, default=0, help="Transformer dropout")
    parser.add_argument("--transformer_attn_dropout", type=float, default=0, help="Transformer attention dropout")
    parser.add_argument("--ch_num", type=int, default=8, help="Number of channels")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument("--ch_emb_dim", type=int, default=32, help="Channel embedding dimension")
    parser.add_argument("--seq_emb_dim", type=int, default=64, help="Sequence embedding dimension")
    parser.add_argument("--num_transformer_blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--clamp_value", type=float, default=10, help="Clamp value")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    setup_logging(args.log_file)
    inference_main(args)

