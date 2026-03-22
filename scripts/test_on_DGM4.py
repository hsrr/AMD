import argparse
import datetime
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from data import DGM4_Dataset, parse_label_to_answer
from multilabel_metrics import AveragePrecisionMeter

# [B-face_swap, C-face_attribute, D-text_swap, E-text_attribute]
LABEL_NAMES = ['face_swap (B)', 'face_attribute (C)', 'text_swap (D)', 'text_attribute (E)']

LETTER_IDS = []


def extract_closed_set_probs(scores):
    """基于闭集假设提取概率，只在 {A,B,C,D,E} 5 个 token 内做 softmax。

    scores: tuple of tensors, 每个 shape (batch_size, vocab_size), 逐 token 的 logits。

    Returns:
        binary_probs:     (batch_size,) ndarray — P(fake) = 1 - P(A) at first step
        multilabel_probs: (batch_size, 4) ndarray — max P(B/C/D/E) across all steps
    """
    target_tensor = torch.tensor(LETTER_IDS, device=scores[0].device)

    # stack along new dim: (n_steps, batch_size, vocab_size)
    all_logits = torch.stack(scores, dim=0)
    # slice to 5 options: (n_steps, batch_size, 5)
    target_logits = all_logits[:, :, target_tensor]
    # closed-set softmax per step per sample
    closed_probs = F.softmax(target_logits, dim=-1)

    # binary: 1 - P(A) at first step, per sample
    binary_probs = (1.0 - closed_probs[0, :, 0]).cpu().numpy()

    # multilabel: max P(B/C/D/E) across steps, per sample
    multilabel_probs = closed_probs[:, :, 1:].max(dim=0).values.cpu().numpy()

    return binary_probs, multilabel_probs


def parse_prediction_vector(pred_str):
    """解析字母答案 (A/B/C/D/E) 为 4 元素二值数组 [B, C, D, E]。"""
    text = str(pred_str).upper().strip()
    vec = [0, 0, 0, 0]  # [B, C, D, E]

    if re.fullmatch(r'\s*A\s*', text):
        return np.array(vec)

    if re.search(r'(?<![A-Z])B(?![A-Z])', text):
        vec[0] = 1
    if re.search(r'(?<![A-Z])C(?![A-Z])', text):
        vec[1] = 1
    if re.search(r'(?<![A-Z])D(?![A-Z])', text):
        vec[2] = 1
    if re.search(r'(?<![A-Z])E(?![A-Z])', text):
        vec[3] = 1

    if 'A' in text and sum(vec) == 0:
        return np.array([0, 0, 0, 0])

    return np.array(vec)


def predict(
    model,
    processor,
    device,
    input,
    images,
    news_texts,
    max_length,
    top_p,
    temperature,
    history,
    modality_cache,
):
    # 保留与目标版本一致的接口，实际仅使用 text+image 进行 Florence2 生成
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True
    generation_config.top_p = None
    generation_config.top_k = None

    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    image = images[0] if images else None
    if image is not None and image.mode != "RGB":
        image = image.convert("RGB")

    if image is None:
        model_inputs = processor(text=prompt_text, return_tensors="pt")
    else:
        model_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            num_beams=3,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
        )

    input_len = model_inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_len:]
    if generated_ids.numel() == 0:
        generated_ids = outputs.sequences
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    scores = outputs.scores
    if scores is None or len(scores) == 0:
        fallback_logits = torch.zeros((1, model.config.vocab_size), device=device)
        scores = (fallback_logits,)

    return response, scores


def resolve_runtime_args(command_args):
    cfg = {}
    if command_args.config and os.path.exists(command_args.config):
        with open(command_args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    model_id = command_args.model_id or command_args.FKA_Owl_ckpt_path or cfg.get('model_id')
    vals = command_args.vals or cfg.get('vals')
    batch_size = command_args.batch_size or cfg.get('batch_size_val', 1)
    gpu_nu = command_args.GPU_nu if command_args.GPU_nu is not None else cfg.get('GPU_nu', 0)
    output_file = command_args.output_file or cfg.get('output_file')

    if not model_id:
        raise ValueError("Missing model path. Please provide --model_id or --FKA_Owl_ckpt_path.")
    if not vals:
        raise ValueError("Missing validation data. Please provide --vals or set vals in --config.")
    if isinstance(vals, str):
        vals = [vals]

    return model_id, vals, int(batch_size), int(gpu_nu), output_file


def main():
    parser = argparse.ArgumentParser("FKA_Owl", add_help=True)
    parser.add_argument("--FKA_Owl_ckpt_path", default=None, help="Alias of --model_id.")
    parser.add_argument("--config", default=None, help="Optional yaml config for vals/model_id/batch_size_val.")
    parser.add_argument("--model_id", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--vals", nargs="+", default=None, help="Validation JSON files.")
    parser.add_argument("--batch_size", type=int, default=None, help="Validation batch size.")
    parser.add_argument("--GPU_nu", type=int, default=0, help="GPU id.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional log output file.")
    command_args = parser.parse_args()

    time1 = datetime.datetime.now()
    model_id, vals, batch_size, gpu_nu, output_file = resolve_runtime_args(command_args)

    device = torch.device(f"cuda:{gpu_nu}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if device.type == "cuda":
        model = model.half()

    print(f'[!] init the model over ...')

    # 预先获取 A/B/C/D/E 字母的 token id
    token_a_id = processor.tokenizer("A", add_special_tokens=False).input_ids[-1]
    token_b_id = processor.tokenizer("B", add_special_tokens=False).input_ids[-1]
    token_c_id = processor.tokenizer("C", add_special_tokens=False).input_ids[-1]
    token_d_id = processor.tokenizer("D", add_special_tokens=False).input_ids[-1]
    token_e_id = processor.tokenizer("E", add_special_tokens=False).input_ids[-1]

    global LETTER_IDS
    LETTER_IDS = [token_a_id, token_b_id, token_c_id, token_d_id, token_e_id]

    def collate_fn(batch):
        images, questions, answers, fake_image_boxes, vector_answers = zip(*batch)
        return list(images), list(questions), list(answers), list(fake_image_boxes), list(vector_answers)

    def log_print(*args_, **kwargs_):
        print(*args_, **kwargs_)
        if output_file:
            with open(output_file, "a") as flog:
                print(*args_, **kwargs_, file=flog)

    # -------- 收集容器 --------
    all_true = []        # N×4 int
    all_pred = []        # N×4 int  (硬预测)
    all_prob = []        # N×4 float (连续概率 P(B/C/D/E))
    all_binary_prob = [] # N float (P(fake) = 1 - P(A))
    all_harm = []

    ap_meter = AveragePrecisionMeter(difficult_examples=False)
    cls_nums_all = 0

    log_print("\nStarting Evaluation...")
    for val_js in vals:
        with open(val_js, "r") as f:
            val_data = json.load(f)

        val_dataset = DGM4_Dataset(split="validation", data=val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            prefetch_factor=None,
        )

        for batch in tqdm(val_loader):
            images, prompts, _, _, gt_vectors = batch
            current_bsz = len(prompts)

            for b_idx in range(current_bsz):
                single_image = [images[b_idx]] if images else []
                # 对齐目标接口；Florence2 侧未使用 news_text
                single_text = []

                resp, scores = predict(
                    model,
                    processor,
                    device,
                    prompts[b_idx],
                    single_image,
                    single_text,
                    512,
                    0.1,
                    1.0,
                    [],
                    [],
                )

                gt_multilabel = gt_vectors[b_idx].numpy().astype(int)
                pred_multilabel = parse_prediction_vector(resp)
                binary_probs, element_probs = extract_closed_set_probs(scores)

                harm_score = np.mean(pred_multilabel == gt_multilabel)
                all_harm.append(harm_score)

                all_true.append(gt_multilabel)
                all_pred.append(pred_multilabel)
                all_prob.append(element_probs[0])
                all_binary_prob.append(binary_probs[0])

                ap_meter.add(
                    torch.from_numpy(element_probs).float(),
                    torch.from_numpy(gt_multilabel).unsqueeze(0).long(),
                )

                cls_nums_all += 1

    all_true = np.array(all_true)        # (N, 4)
    all_pred = np.array(all_pred)        # (N, 4)
    all_prob = np.array(all_prob)        # (N, 4)
    all_binary_prob = np.array(all_binary_prob)  # (N,)

    # ==================== 指标计算 ====================

    # --- 1. HARM Score (partial credit) ---
    avg_harm = np.mean(all_harm)

    # --- 2. Exact Match ---
    exact_match = np.all(all_true == all_pred, axis=1).sum()
    exact_match_acc = exact_match / cls_nums_all

    # --- 3a. 整体二分类 (real vs fake): ACC & AUC ---
    gt_binary = (all_true.sum(axis=1) > 0).astype(int)
    pred_binary = (all_binary_prob >= 0.5).astype(int)
    binary_acc = np.mean(gt_binary == pred_binary)
    try:
        binary_auc = roc_auc_score(gt_binary, all_binary_prob)
    except ValueError:
        binary_auc = float('nan')

    # --- 3b. 每个 label 的 ACC & AUC (基于连续概率) ---
    per_label_acc = []
    per_label_auc = []
    for j in range(4):
        y_t = all_true[:, j]
        y_p = all_pred[:, j]
        y_prob = all_prob[:, j]
        per_label_acc.append(np.mean(y_t == y_p))
        if y_t.sum() > 0 and len(np.unique(y_t)) > 1:
            per_label_auc.append(roc_auc_score(y_t, y_prob))
        else:
            per_label_auc.append(float('nan'))

    # --- 4. mAP (per-class AP 取均值) ---
    per_class_ap = ap_meter.value()  # (4,) tensor
    if isinstance(per_class_ap, torch.Tensor):
        mAP = per_class_ap.mean().item()
    else:
        per_class_ap = torch.zeros(4)
        mAP = 0.0

    # --- 5. CF1 / OF1 (多标签 P/R/F1) ---
    OP, OR, OF1, CP, CR, CF1 = ap_meter.evaluation(all_prob, all_true)

    # ==================== 打印结果 ====================

    log_print("\n" + "=" * 60)
    log_print("  Multi-label Classification Results (Guardian)")
    log_print("=" * 60)

    log_print(f"\n  HARM Score (partial credit) : {avg_harm:.4f}")
    log_print(f"  Exact Match Accuracy        : {exact_match_acc:.4f} ({exact_match}/{cls_nums_all})")

    binary_auc_str = f"{binary_auc:.4f}" if not np.isnan(binary_auc) else "N/A"
    log_print(f"\n  --- Binary (Real vs Fake) ---")
    log_print(f"  Binary ACC : {binary_acc:.4f}")
    log_print(f"  Binary AUC : {binary_auc_str}")

    log_print(f"\n  --- Multi-label Aggregate ---")
    log_print(f"  mAP  : {mAP:.4f}")
    log_print(f"  OF1  : {OF1:.4f}  (OP={OP:.4f}, OR={OR:.4f})")
    log_print(f"  CF1  : {CF1:.4f}  (CP={CP:.4f}, CR={CR:.4f})")

    log_print(f"\n  --- Per-label Binary (ACC / AUC) ---")
    log_print(f"  {'Label':<25s} {'ACC':>8s} {'AUC':>8s} {'AP':>8s}")
    for j, name in enumerate(LABEL_NAMES):
        auc_str = f"{per_label_auc[j]:.4f}" if not np.isnan(per_label_auc[j]) else "   N/A"
        ap_str = f"{per_class_ap[j].item():.4f}"
        log_print(f"  {name:<25s} {per_label_acc[j]:>8.4f} {auc_str:>8s} {ap_str:>8s}")

    log_print(f"\n  --- Per-label Classification Report ---")
    for j, name in enumerate(LABEL_NAMES):
        y_t = all_true[:, j]
        y_p = all_pred[:, j]
        log_print(f"\n  [{name}]")
        log_print(classification_report(y_t, y_p, target_names=['no', 'yes'], digits=4, zero_division=0))

    time2 = datetime.datetime.now()
    log_print(f"\nTime consumed: {time2 - time1}")


if __name__ == "__main__":
    main()
