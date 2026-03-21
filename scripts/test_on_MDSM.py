import argparse
import datetime
import json
import logging
import os
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from data import DGM4_Dataset, describles_answ


FAKE_CLS_TO_MULTI = {
    "orig": [0, 0, 0, 0],
    "face_swap": [1, 0, 0, 0],
    "face_attribute": [0, 1, 0, 0],
    "text_swap": [0, 0, 1, 0],
    "text_attribute": [0, 0, 0, 1],
    "face_swap&text_swap": [1, 0, 1, 0],
    "face_swap&text_attribute": [1, 0, 0, 1],
    "face_attribute&text_swap": [0, 1, 1, 0],
    "face_attribute&text_attribute": [0, 1, 0, 1],
}


def load_json_or_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def build_option_space(device):
    options = list(describles_answ.values())
    option_labels = [
        torch.tensor(FAKE_CLS_TO_MULTI[fake_cls], dtype=torch.long, device=device)
        for fake_cls in describles_answ.keys()
    ]
    option_to_multi = {
        describles_answ[fake_cls]: torch.tensor(
            FAKE_CLS_TO_MULTI[fake_cls], dtype=torch.long, device=device
        )
        for fake_cls in describles_answ.keys()
    }
    return options, option_labels, option_to_multi


def normalize_generated_answer(text):
    text = re.sub(r"<pad>|<s>|</s>", "", text).strip()
    if "Swapped words:" in text:
        text = text.split("Swapped words:")[0].strip()
    if "Manipulated face" in text:
        text = text.split("Manipulated face")[0].strip()
    return text


def find_option_from_text(text, options):
    for option in options:
        if option in text:
            return option
    return options[0]


def get_multi_and_binary_labels(answers, options, option_to_multi, device):
    real_multi = torch.zeros((len(answers), 4), dtype=torch.long, device=device)
    real_binary = torch.zeros(len(answers), dtype=torch.long, device=device)
    for idx, ans in enumerate(answers):
        matched_option = find_option_from_text(ans, options)
        real_multi[idx] = option_to_multi[matched_option]
        if matched_option != options[0]:
            real_binary[idx] = 1
    return real_multi, real_binary


def get_best_option(generated_texts, option_vectors, vectorizer, options, option_labels, device):
    generated_vectors = vectorizer.transform(generated_texts).toarray()
    similarities = cosine_similarity(generated_vectors, option_vectors)
    best_option_indices = similarities.argmax(axis=1)
    best_multi_labels = torch.stack([option_labels[i] for i in best_option_indices], dim=0)

    pred_binary = torch.ones(len(generated_texts), dtype=torch.long, device=device)
    pred_binary[np.array(best_option_indices) == 0] = 0
    fake_scores = similarities[:, 1:].max(axis=1)
    return best_multi_labels, pred_binary, fake_scores


def compute_multilabel_scores(pred_multi, real_multi):
    pred_np = pred_multi.detach().cpu().numpy().astype(np.int64)
    real_np = real_multi.detach().cpu().numpy().astype(np.int64)

    nc = np.sum((pred_np == 1) & (real_np == 1), axis=0).astype(np.float64)
    npred = np.sum(pred_np == 1, axis=0).astype(np.float64)
    ngt = np.sum(real_np == 1, axis=0).astype(np.float64)

    op = float(np.sum(nc) / np.sum(npred)) if np.sum(npred) > 0 else 0.0
    orr = float(np.sum(nc) / np.sum(ngt)) if np.sum(ngt) > 0 else 0.0
    of1 = float((2 * op * orr) / (op + orr)) if (op + orr) > 0 else 0.0

    cp_per_cls = np.divide(nc, npred, out=np.zeros_like(nc), where=npred > 0)
    cr_per_cls = np.divide(nc, ngt, out=np.zeros_like(nc), where=ngt > 0)
    cp = float(np.mean(cp_per_cls))
    cr = float(np.mean(cr_per_cls))
    cf1 = float((2 * cp * cr) / (cp + cr)) if (cp + cr) > 0 else 0.0

    multi_acc = float(np.mean(np.all(pred_np == real_np, axis=1)))
    return cf1, of1, multi_acc


def evaluate_model(test_loader, model, processor, device, option_vectors, vectorizer, options, option_labels, option_to_multi):
    all_real_binary = []
    all_pred_binary = []
    all_fake_scores = []
    all_real_multi = []
    all_pred_multi = []

    for inputs, batch_answers in tqdm(test_loader, desc="Evaluating"):
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        cleaned_answers = [normalize_generated_answer(text) for text in generated_texts]

        real_multi, real_binary = get_multi_and_binary_labels(
            batch_answers, options, option_to_multi, device
        )
        pred_multi, pred_binary, fake_scores = get_best_option(
            cleaned_answers, option_vectors, vectorizer, options, option_labels, device
        )

        all_real_binary.append(real_binary)
        all_pred_binary.append(pred_binary)
        all_fake_scores.append(torch.tensor(fake_scores, dtype=torch.float32, device=device))
        all_real_multi.append(real_multi)
        all_pred_multi.append(pred_multi)

    real_binary = torch.cat(all_real_binary, dim=0)
    pred_binary = torch.cat(all_pred_binary, dim=0)
    fake_scores = torch.cat(all_fake_scores, dim=0).detach().cpu().numpy()
    real_binary_np = real_binary.detach().cpu().numpy()
    pred_binary_np = pred_binary.detach().cpu().numpy()

    binary_acc = float(np.mean(real_binary_np == pred_binary_np))
    binary_err = 1.0 - binary_acc
    try:
        binary_auc = float(roc_auc_score(real_binary_np, fake_scores))
    except ValueError:
        binary_auc = float("nan")

    real_multi = torch.cat(all_real_multi, dim=0)
    pred_multi = torch.cat(all_pred_multi, dim=0)
    multi_cf1, multi_of1, multi_acc = compute_multilabel_scores(pred_multi, real_multi)

    return {
        "binary_auc": binary_auc,
        "binary_err": binary_err,
        "binary_acc": binary_acc,
        "multi_cf1": multi_cf1,
        "multi_of1": multi_of1,
        "multi_acc": multi_acc,
        "count": int(real_binary.size(0)),
    }


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
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    output_file = args.output_file or os.path.join(args.model_id, f"domain_test_out_{timestamp}.txt")

    options, option_labels, option_to_multi = build_option_space(device)
    vectorizer = TfidfVectorizer().fit(options)
    option_vectors = vectorizer.transform(options).toarray()

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
    log_print(f"options from data.py prompt: {len(options)} classes")

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
            processor=processor,
            device=device,
            option_vectors=option_vectors,
            vectorizer=vectorizer,
            options=options,
            option_labels=option_labels,
            option_to_multi=option_to_multi,
        )

        log_print("#######<--record-->###########")
        log_print(f"samples: {metrics['count']}")
        log_print(f"Binary AUC: {metrics['binary_auc']:.6f}")
        log_print(f"Binary ERR: {metrics['binary_err']:.6f}")
        log_print(f"Binary ACC: {metrics['binary_acc']:.6f}")
        log_print(f"Multi CF1: {metrics['multi_cf1']:.6f}")
        log_print(f"Multi OF1: {metrics['multi_of1']:.6f}")
        log_print(f"Multi ACC: {metrics['multi_acc']:.6f}")
        log_print("########<--record-->#########\n")

    print(f"Logs saved to: {output_file}")


if __name__ == "__main__":
    main()