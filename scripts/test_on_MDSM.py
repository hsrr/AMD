import logging
import random
import argparse
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from data import DGM4_Dataset
from sklearn.metrics import roc_auc_score
import numpy as np
import sys, re
from multilabel_metrics import AveragePrecisionMeter
import os,math

import datetime
import time


LETTER_TO_IDX = {'B': 0, 'C': 1, 'D': 2, 'E': 3}


def get_letter_token_ids(tokenizer):
    """Dynamically resolve token IDs for A/B/C/D/E from the tokenizer."""
    a_id = tokenizer.convert_tokens_to_ids('A')
    multi_ids = [tokenizer.convert_tokens_to_ids(c) for c in ['B', 'C', 'D', 'E']]
    return a_id, multi_ids


def extract_scores_from_logits(lm_logits, a_token_id, multi_label_token_ids):
    """Extract all continuous scores from the decoder LM logits (main backbone).

    Returns:
        binary_scores: [N] — P(fake) = 1 - P(A) at position 0.
        multilabel_scores: [N,4] — max P(B/C/D/E) across all positions.
    """
    probs = F.softmax(lm_logits, dim=-1)

    p_a = probs[:, 0, a_token_id]
    binary_scores = 1.0 - p_a

    token_ids = torch.tensor(multi_label_token_ids, device=probs.device)
    letter_probs = probs[:, :, token_ids]
    multilabel_scores, _ = letter_probs.max(dim=1)

    return binary_scores, multilabel_scores

def parse_generated_to_multilabel(generated_texts, device):
    """Parse generated texts (A-E letter format) into multi-label [N,4] and binary labels."""
    multi_label = torch.zeros(len(generated_texts), 4, dtype=torch.float32).to(device)
    pred_label = torch.ones(len(generated_texts), dtype=torch.long).to(device)
    
    for i, text in enumerate(generated_texts):
        clean = text.split('Manipulated')[0].split('Swapped')[0].strip()
        clean = re.sub(r'[^A-E,\s]', '', clean).strip()
        letters = [l.strip() for l in clean.split(',') if l.strip()]
        
        if not letters or letters == ['A']:
            pred_label[i] = 0
            continue
        
        has_valid = False
        for letter in letters:
            if letter in LETTER_TO_IDX:
                multi_label[i, LETTER_TO_IDX[letter]] = 1.0
                has_valid = True
        if not has_valid:
            pred_label[i] = 0
    
    return multi_label, pred_label

def get_multi_label_from_vectors(vector_answers, device):
    """Build multi_label [N,4] from pre-computed vector_answers."""
    multi_label = torch.stack(list(vector_answers), dim=0).long().to(device)
    real_label_pos = [i for i in range(len(vector_answers)) if vector_answers[i].sum().item() == 0]
    return multi_label, real_label_pos

def get_multi_label_from_text(answers, device):
    """Parse answer text (new A-E format) into multi_label [N,4]."""
    multi_label = torch.zeros([len(answers), 4], dtype=torch.long).to(device)
    real_label_pos = []
    for i, ans in enumerate(answers):
        clean = ans.split('Manipulated')[0].split('Swapped')[0].strip()
        letters = [l.strip() for l in clean.split(',')]
        if letters == ['A'] or clean == 'A':
            real_label_pos.append(i)
            continue
        for letter in letters:
            if letter in LETTER_TO_IDX:
                multi_label[i, LETTER_TO_IDX[letter]] = 1
    return multi_label, real_label_pos







def run_batch(inputs,model,processor):
    # 调用 modeling_florence2.py 中的generate
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts




def box_iou(boxes1, boxes2, test=False):
    '''
    计算两个边界框集合的 IoU（Intersection over Union），
    并返回每个边界框对的 IoU 值和并集面积。
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    # union = area1[:, None] + area2 - inter
    union = area1 + area2 - inter

    iou = inter / union

    if test:
        zero_lines = boxes2==torch.zeros_like(boxes2)
        zero_lines_idx = torch.where(zero_lines[:,0]==True)[0]

        for idx in zero_lines_idx:
            if all(boxes1[idx,:] < 1e-4):
                iou[idx]=1

    return iou, union

def parse_coordinates(text):
    # 使用正则表达式匹配坐标
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    match = re.search(pattern, text)
    # print(f'input text is {text}')
    
    if match:
        # 将匹配到的坐标转换为整数
        loc_x1 = int(match.group(1))
        loc_y1 = int(match.group(2))
        loc_x2 = int(match.group(3))
        loc_y2 = int(match.group(4))
        # print('解析到的坐标是：')
        # print(loc_x1, loc_y1, loc_x2, loc_y2)
        return torch.tensor([[loc_x1, loc_y1, loc_x2, loc_y2]])
    else:
        # print('没有match')
        return torch.tensor([[0, 0, 0, 0]])

def evaluate_model(test_loader, model, processer, device, a_token_id, multi_label_token_ids):

    cls_nums_all = 0
    cls_acc_all = 0   
    all_binary_gt = []
    all_binary_scores = []
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for inputs, batch_answers, vector_answers in tqdm(test_loader, desc="Evaluating"):

        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)

        labels = processer.tokenizer(
            text=batch_answers,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=800,
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )

        lm_logits = outputs.logits
        binary_score, multilabel_scores = extract_scores_from_logits(lm_logits, a_token_id, multi_label_token_ids)

        generated_texts = run_batch(inputs, model, processer)
        task_answers = []
        
        for i, (generated_text, answers) in enumerate(zip(generated_texts, batch_answers)):
            full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
            if '<loc_' in full_answer:
                task_answers.append(full_answer.split('Manipulated face')[0])
            else:
                task_answers.append(full_answer)
                
        real_multi_label, real_label_pos = get_multi_label_from_vectors(vector_answers, device)
        real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
        real_label[real_label_pos] = 0
        
        pred_multi_label, pred_label = parse_generated_to_multilabel(task_answers, device)

        cls_nums_all += len(generated_texts)
        cls_acc_all += torch.sum(real_label == pred_label).item()
        
        all_binary_gt.extend(real_label.cpu().tolist())
        all_binary_scores.extend(binary_score.cpu().tolist())

        multi_label_meter.add(multilabel_scores, real_multi_label)

    ACC_cls = cls_acc_all / cls_nums_all if cls_nums_all > 0 else 0.0
    
    ap_values = multi_label_meter.value()
    MAP = ap_values.mean() if isinstance(ap_values, torch.Tensor) and ap_values.numel() > 0 else 0.0
    
    AUC = 0.0
    try:
        if len(set(all_binary_gt)) > 1:
            AUC = roc_auc_score(all_binary_gt, all_binary_scores)
    except Exception:
        AUC = 0.0

    OP, OR, OF1, CP, CR, CF1 = 0, 0, 0, 0, 0, 0
    try:
        OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    except Exception:
        pass

    return ACC_cls, cls_acc_all, cls_nums_all, MAP, OP, OR, OF1, CP, CR, CF1, AUC


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on multiple validation datasets")
    parser.add_argument('--GPU_nu', type=int, default=0, help='GPU index to use')
    parser.add_argument('--model_id', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for DataLoader')
    parser.add_argument('--vals', type=str, nargs='+', required=True, help='List of validation JSON files')
    parser.add_argument('--output_file', type=str, default=None, help='Optional: output log file path')
    
    args = parser.parse_args()

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set device
    device = torch.device(f"cuda:{args.GPU_nu}" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    a_token_id, multi_label_token_ids = get_letter_token_ids(processor.tokenizer)

    # Prepare output file path
    output_file = args.output_file or os.path.join(args.model_id, f"domain_test_out_{timestamp}.txt")

    def log_print(*print_args, **kwargs):
        print(*print_args, **kwargs)
        with open(output_file, "a") as flog:
            print(*print_args, **kwargs, file=flog)
            
    def collate_fn(batch):
        images, questions, answers, fake_image_box, vector_answers = zip(*batch)
        
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        return inputs, answers, vector_answers

    # Print test configuration
    log_print(f'Test model_id is: {args.model_id}')

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # DataLoader config
    num_workers = 0
    prefetch_factor = None

    # Process each validation file
    for val_js in args.vals:
        with open(val_js, "r") as f:
            val_data = json.load(f)

        log_print(f"Testing on: {val_js}")
        log_print(f"Validation data size: {len(val_data)}")

        test_dataset = DGM4_Dataset(split="validation", data=val_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        # Evaluate
        ACC_cls, cls_acc_all, cls_nums_all, MAP, OP, OR, OF1, CP, CR, CF1, AUC = evaluate_model(test_loader, model, processor, device, a_token_id, multi_label_token_ids)

        log_print('#######<--record-->###########')
        log_print(f"binary_acc={ACC_cls*100:.2f}% (cls_acc_all: {cls_acc_all}, cls_nums_all: {cls_nums_all})")
        log_print(f"binary_auc={AUC:.4f}")
        log_print(f"mAP={MAP:.4f}")
        log_print(f"CF1={CF1:.4f}")
        log_print(f"OP={OP:.4f}, OR={OR:.4f}, OF1={OF1:.4f}")
        log_print(f"CP={CP:.4f}, CR={CR:.4f}")
        log_print('########<--record-->#########\n')

    print(f"Logs saved to: {output_file}")
    
if __name__ == "__main__":
    main()