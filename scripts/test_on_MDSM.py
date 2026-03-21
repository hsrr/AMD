import argparse
import datetime
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from data import DGM4_Dataset
from multilabel_metrics import AveragePrecisionMeter


OPTION_PREFIX_TO_MULTI = {
    "A": [0, 0, 0, 0],
    "B": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "D": [0, 0, 1, 0],
    "E": [0, 0, 0, 1],
    "F": [1, 0, 1, 0],
    "G": [1, 0, 0, 1],
    "H": [0, 1, 1, 0],
    "I": [0, 1, 0, 1],
}


def load_json_or_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def get_multi_labels(answers, device):
    real_multi = torch.zeros((len(answers), 4), dtype=torch.long, device=device)
    for idx, ans in enumerate(answers):
        prefix = ans.strip()[:1].upper() if isinstance(ans, str) and len(ans.strip()) > 0 else ""
        if prefix in OPTION_PREFIX_TO_MULTI:
            real_multi[idx] = torch.tensor(OPTION_PREFIX_TO_MULTI[prefix], dtype=torch.long, device=device)
    return real_multi


def fuse_multilabel_logits(logits_list):
    valid_logits = []
    for idx in (0, 1, 2):
        if idx < len(logits_list) and logits_list[idx] is not None:
            valid_logits.append(logits_list[idx])
    if not valid_logits:
        raise RuntimeError("classification_logits_list[0/1/2] all None, cannot evaluate")
    return torch.stack(valid_logits, dim=0).mean(dim=0)


def compute_multilabel_scores(pred_multi, real_multi, prob_scores):
    eps = 1e-8
    pred_np = pred_multi.detach().cpu().numpy().astype(np.int64)
    real_np = real_multi.detach().cpu().numpy().astype(np.int64)

    tp = np.sum((pred_np == 1) & (real_np == 1), axis=0).astype(np.float64)
    fp = np.sum((pred_np == 1) & (real_np == 0), axis=0).astype(np.float64)
    fn = np.sum((pred_np == 0) & (real_np == 1), axis=0).astype(np.float64)

    p_cls = tp / (tp + fp + eps)
    r_cls = tp / (tp + fn + eps)
    f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls + eps)

    op = float(tp.sum() / (tp.sum() + fp.sum() + eps))
    orr = float(tp.sum() / (tp.sum() + fn.sum() + eps))
    of1 = float(2 * op * orr / (op + orr + eps))

    cp = float(np.mean(p_cls))
    cr = float(np.mean(r_cls))
    cf1 = float(2 * cp * cr / (cp + cr + eps))

    label_acc = float(np.mean(pred_np == real_np))
    sample_acc = float(np.mean(np.all(pred_np == real_np, axis=1)))

    ap_meter = AveragePrecisionMeter(difficult_examples=False)
    ap_meter.reset()
    ap_meter.add(prob_scores.detach().cpu(), real_multi.detach().cpu())
    ap_values = ap_meter.value()
    map_score = float(ap_values[:4].mean().item()) if torch.is_tensor(ap_values) else 0.0

    return {
        "f1_fs": float(f1_cls[0]),
        "f1_fa": float(f1_cls[1]),
        "f1_ts": float(f1_cls[2]),
        "f1_ta": float(f1_cls[3]),
        "op": op,
        "or": orr,
        "of1": of1,
        "cp": cp,
        "cr": cr,
        "cf1": cf1,
        "map": map_score,
        "label_acc": label_acc,
        "sample_acc": sample_acc,
    }


def evaluate_model(test_loader, model, device):
    all_real_multi = []
    all_pred_multi = []
    all_prob_scores = []

    for inputs, batch_answers in tqdm(test_loader, desc="Evaluating"):
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
        )
        fused_logits = fuse_multilabel_logits(outputs.classification_logits_list)
        prob_scores = torch.sigmoid(fused_logits)
        pred_multi = (prob_scores >= 0.5).long()
        real_multi = get_multi_labels(batch_answers, device)

        all_real_multi.append(real_multi)
        all_pred_multi.append(pred_multi)
        all_prob_scores.append(prob_scores)

    real_multi = torch.cat(all_real_multi, dim=0)
    pred_multi = torch.cat(all_pred_multi, dim=0)
    prob_scores = torch.cat(all_prob_scores, dim=0)
    metrics = compute_multilabel_scores(pred_multi, real_multi, prob_scores)
    metrics["count"] = int(real_multi.size(0))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on DGM4-style data")
    parser.add_argument("--GPU_nu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--model_id", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for DataLoader")
    parser.add_argument("--vals", type=str, nargs="+", required=True, help="List of validation JSON files")
    parser.add_argument("--output_file", type=str, default=None, help="Optional output log file path")
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Root directory for dataset images, joined with ann['image']",
    )
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(f"cuda:{args.GPU_nu}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, ignore_mismatched_sizes=True
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    output_file = args.output_file or os.path.join(args.model_id, f"domain_test_out_{timestamp}.txt")

    logging.basicConfig(level=logging.INFO)

    def log_print(*print_args, **kwargs):
        print(*print_args, **kwargs)
        with open(output_file, "a", encoding="utf-8") as flog:
            print(*print_args, **kwargs, file=flog)

    def collate_fn(batch):
        images, questions, answers = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        return inputs, answers

    log_print(f"Test model_id is: {args.model_id}")
    log_print("evaluate by 4-dim multilabel head (FS/FA/TS/TA)")

    for val_js in args.vals:
        val_data = load_json_or_jsonl(val_js)
        log_print(f"Testing on: {val_js}")
        log_print(f"Validation data size: {len(val_data)}")

        test_dataset = DGM4_Dataset(split="validation", data=val_data, image_root=args.image_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            prefetch_factor=None,
        )

        metrics = evaluate_model(
            test_loader=test_loader,
            model=model,
            device=device,
        )

        log_print("#######<--record-->###########")
        log_print(f"samples: {metrics['count']}")
        log_print(f"F1_FS: {metrics['f1_fs']:.6f}")
        log_print(f"F1_FA: {metrics['f1_fa']:.6f}")
        log_print(f"F1_TS: {metrics['f1_ts']:.6f}")
        log_print(f"F1_TA: {metrics['f1_ta']:.6f}")
        log_print(f"OP: {metrics['op']:.6f}")
        log_print(f"OR: {metrics['or']:.6f}")
        log_print(f"OF1: {metrics['of1']:.6f}")
        log_print(f"CP: {metrics['cp']:.6f}")
        log_print(f"CR: {metrics['cr']:.6f}")
        log_print(f"CF1: {metrics['cf1']:.6f}")
        log_print(f"mAP: {metrics['map']:.6f}")
        log_print(f"Label ACC: {metrics['label_acc']:.6f}")
        log_print(f"Sample ACC: {metrics['sample_acc']:.6f}")
        log_print("########<--record-->#########\n")

    print(f"Logs saved to: {output_file}")


if __name__ == "__main__":
    main()