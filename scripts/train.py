import argparse
import os
import json
import re
from functools import partial
from datetime import datetime
import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
import torch.nn.functional as F
import box_ops
import math

import random
import wandb
from data import DocVQADataset, TheCauldronDataset, VQAInstructDataset,DGM4_Dataset
from peft import LoraConfig, get_peft_model
import numpy as np
import sys
from multilabel_metrics import AveragePrecisionMeter
from torchvision.ops.boxes import box_area



def load_json_or_jsonl(filepath):
    with open(filepath, "r") as f:
        content = f.read().strip()
    if content.startswith('['):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def set_seed(seed, rank=0):

    seed = seed + rank  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def get_bbox_loss(output_coord, target_bbox, is_image=None):
    """
    Bounding Box Loss: L1 & GIoU

    Args:
        image_embeds: encoding full images
    """
    loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

    boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
    boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
    if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
        # early check of degenerated boxes
        # print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
        loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
    else:
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
        loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

    if is_image is None:
        num_boxes = target_bbox.size(0)
    else:
        num_boxes = torch.sum(1 - is_image)
        loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
        loss_giou = loss_giou * (1 - is_image)

    return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes


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

def collate_fn(batch, processor, device):

    #### DGM4的定义：
    images, questions, answers = zip(*batch)
    
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers


def create_data_loaders(
    train_dataset,
    val_datasets,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_loaders = {}
    for name, val_dataset in val_datasets.items():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size//2,
            collate_fn=partial(collate_fn, processor=processor, device=device),
            num_workers=num_workers,
            sampler=val_sampler,
        )
        val_loaders[name] = val_loader

    return train_loader, val_loaders

OPTION_PREFIX_TO_MULTI = {
    "A": [1, 0, 0, 0, 0],  # No
    "B": [0, 1, 0, 0, 0],  # FS
    "C": [0, 0, 1, 0, 0],  # FA
    "D": [0, 0, 0, 1, 0],  # TS
    "E": [0, 0, 0, 0, 1],  # TA
    "F": [0, 1, 0, 1, 0],  # FS + TS
    "G": [0, 1, 0, 0, 1],  # FS + TA
    "H": [0, 0, 1, 1, 0],  # FA + TS
    "I": [0, 0, 1, 0, 1],  # FA + TA
}


def get_multi_label(answers, device):
    multi_label = torch.zeros((len(answers), 5), dtype=torch.float32, device=device)
    for idx, answer in enumerate(answers):
        prefix = answer.strip()[:1].upper() if isinstance(answer, str) and len(answer.strip()) > 0 else ""
        if prefix in OPTION_PREFIX_TO_MULTI:
            multi_label[idx] = torch.tensor(OPTION_PREFIX_TO_MULTI[prefix], dtype=torch.float32, device=device)
    return multi_label


def fuse_multilabel_logits(logits_list):
    valid_logits = []
    for idx in (0, 1, 2):
        if idx < len(logits_list) and logits_list[idx] is not None:
            valid_logits.append(logits_list[idx])
    if not valid_logits:
        raise RuntimeError("classification_logits_list[0/1/2] all None, cannot compute multilabel logits")
    return torch.stack(valid_logits, dim=0).mean(dim=0)


def compute_multilabel_metrics(pred_labels, gt_labels, prob_scores):
    eps = 1e-8
    # DGM4指标按4类篡改类型计算：FS/FA/TS/TA，不包含"No"维度
    pred_np = pred_labels[:, 1:].detach().cpu().numpy().astype(np.int64)
    gt_np = gt_labels[:, 1:].detach().cpu().numpy().astype(np.int64)
    prob_np = prob_scores[:, 1:]

    tp = np.sum((pred_np == 1) & (gt_np == 1), axis=0).astype(np.float64)
    fp = np.sum((pred_np == 1) & (gt_np == 0), axis=0).astype(np.float64)
    fn = np.sum((pred_np == 0) & (gt_np == 1), axis=0).astype(np.float64)

    p_cls = tp / (tp + fp + eps)
    r_cls = tp / (tp + fn + eps)
    f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls + eps)

    op = float(tp.sum() / (tp.sum() + fp.sum() + eps))
    orr = float(tp.sum() / (tp.sum() + fn.sum() + eps))
    of1 = float(2 * op * orr / (op + orr + eps))

    cp = float(np.mean(p_cls))
    cr = float(np.mean(r_cls))
    cf1 = float(2 * cp * cr / (cp + cr + eps))

    label_acc = float(np.mean(pred_np == gt_np))
    sample_acc = float(np.mean(np.all(pred_np == gt_np, axis=1)))

    ap_meter = AveragePrecisionMeter(difficult_examples=False)
    ap_meter.reset()
    ap_meter.add(prob_np.detach().cpu(), gt_labels[:, 1:].detach().cpu().long())
    ap_values = ap_meter.value()
    if torch.is_tensor(ap_values):
        map_score = float(ap_values.mean().item())
    else:
        map_score = 0.0

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

def synchronize_metrics(metric_tensor, world_size):
    """
    使用reduce操作同步指标。聚合各个进程的指标，计算全局值。
    """
    # 将指标结果归约到 rank 0 进程
    dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)
    # 在 rank 0 进程计算均值
    if dist.get_rank() == 0:
        metric_tensor /= world_size
    return metric_tensor


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
        print('没有match')
        return torch.tensor([[0, 0, 0, 0]])


def evaluate_model(rank, world_size, model, val_loaders, device, global_step, max_val_item_count):

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            val_item_count = 0
            all_probs = []
            all_preds = []
            all_targets = []
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                inputs, batch_answers = batch
                batch_size = inputs["input_ids"].size(0)
                val_item_count += batch_size

                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                )
                fused_logits = fuse_multilabel_logits(outputs.classification_logits_list)
                prob_scores = torch.sigmoid(fused_logits)
                pred_labels = (prob_scores >= 0.5).long()
                gt_labels = get_multi_label(batch_answers, device).long()

                all_probs.append(prob_scores)
                all_preds.append(pred_labels)
                all_targets.append(gt_labels)

                if val_item_count > max_val_item_count:
                    break

            if len(all_targets) == 0:
                continue

            probs = torch.cat(all_probs, dim=0)
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            local_metrics = compute_multilabel_metrics(preds, targets, probs)

            synced_metrics = {}
            for metric_name, metric_value in local_metrics.items():
                metric_tensor = torch.tensor(metric_value, dtype=torch.float32, device=device)
                synced_metrics[metric_name] = synchronize_metrics(metric_tensor, world_size)

            if dist.get_rank() == 0:
                print(
                    f"Rank {rank} - Step {global_step} - {val_name} "
                    f"CF1={synced_metrics['cf1'].item():.4f} OF1={synced_metrics['of1'].item():.4f} "
                    f"mAP={synced_metrics['map'].item():.4f} label_acc={synced_metrics['label_acc'].item():.4f}"
                )
                wandb.log({
                    f"{val_name}_F1_FS": synced_metrics["f1_fs"].item(),
                    f"{val_name}_F1_FA": synced_metrics["f1_fa"].item(),
                    f"{val_name}_F1_TS": synced_metrics["f1_ts"].item(),
                    f"{val_name}_F1_TA": synced_metrics["f1_ta"].item(),
                    f"{val_name}_OP": synced_metrics["op"].item(),
                    f"{val_name}_OR": synced_metrics["or"].item(),
                    f"{val_name}_OF1": synced_metrics["of1"].item(),
                    f"{val_name}_CP": synced_metrics["cp"].item(),
                    f"{val_name}_CR": synced_metrics["cr"].item(),
                    f"{val_name}_CF1": synced_metrics["cf1"].item(),
                    f"{val_name}_MAP": synced_metrics["map"].item(),
                    f"{val_name}_label_acc": synced_metrics["label_acc"].item(),
                    f"{val_name}_sample_acc": synced_metrics["sample_acc"].item(),
                    "step": global_step,
                })
            
    model.train()



def train_model(rank, AMD_init_pth, train_js, val_js, world_size, dataset_name, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000, regular_weight=0.07, train_domain='NYT',random_seed=12, image_root=''):
    setup(rank, world_size)
    set_seed(random_seed, rank)
    device = torch.device(f"cuda:{rank}")
    train_data=[]
    val_data=[]
    
    logged_task_name = f'AMD_test_{train_domain}' 
    train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_train_name = f'{logged_task_name}_{train_time}' 
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # Initialize wandb
    if rank == 0:  # Only initialize wandb in the main process
        wandb.init(project= logged_task_name, name=run_name)
        wandb.config.update({
            "dataset": dataset_name,
            "batch_size": batch_size,
            "use_lora": use_lora,
            
            "epochs": epochs,
            "learning_rate": lr,
            "eval_steps": eval_steps,
            "world_size": world_size,
        })

    # Load the dataset based on the dataset_name argument
    if dataset_name == "docvqa":
        train_dataset = DocVQADataset(split='train')
        val_datasets = {"docvqa": DocVQADataset(split='validation')}
    elif dataset_name == "cauldron":
        train_dataset = TheCauldronDataset(split='train')
        val_datasets = {
            "cauldron": TheCauldronDataset(split='validation'), 
            "docvqa": DocVQADataset(split='validation')
        }
    elif dataset_name == 'vqainstruct':
        train_dataset = VQAInstructDataset(split='train')
        val_datasets = {
            "vqainstruct": VQAInstructDataset(split='validation'), 
            "docvqa": DocVQADataset(split='validation')
        }
    elif dataset_name == 'DGM4':
        train_data = load_json_or_jsonl(train_js)
        val_data = load_json_or_jsonl(val_js)
            
        train_dataset = DGM4_Dataset(split='train', data=train_data, image_root=image_root)
        val_datasets = {"DGM4": DGM4_Dataset(split='validation', data=val_data, image_root=image_root)}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        AMD_init_pth, trust_remote_code=True, local_files_only=True, ignore_mismatched_sizes=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        AMD_init_pth, trust_remote_code=True, local_files_only=True
    )
    

    if use_lora:
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Create DataLoaders
    num_workers = 0
    train_loader, val_loaders = create_data_loaders(
        train_dataset,
        val_datasets,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)  #lr=1e-6
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        LLM_loss = 0
        image_loss = 0
        text_loss = 0
        LT_loss = 0
        regular_loss_total = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)
            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            if outputs.loss is None:
                raise RuntimeError("模型未返回LM loss，无法按原始训练方式执行")
            total_loss = outputs.loss
            multi_labels = get_multi_label(answers, device)
            temp_loss0 = torch.tensor(0.0, device=device)
            temp_loss1 = torch.tensor(0.0, device=device)
            temp_loss2 = torch.tensor(0.0, device=device)
            reg_loss = torch.tensor(0.0, device=device)

            logits_list = outputs.classification_logits_list
            ### logits = [image_classification, text_classification,learnable_token_logits,output_coord,loss_regular]
            
            for i,logits in enumerate(logits_list):
                if logits is not None:
                    if i == 0:
                        temp_loss0 = criterion(logits, multi_labels)
                        if torch.isnan(temp_loss0):
                            raise RuntimeError(f"❌ logits_list[{i}] 多标签损失 temp_loss0 为 NaN")
                        total_loss += 0.1*temp_loss0 
                    if i == 1:
                        temp_loss1 = criterion(logits, multi_labels)
                        if torch.isnan(temp_loss1):
                            raise RuntimeError(f"❌ logits_list[{i}]  temp_loss1 = NaN")
                        total_loss += 0.1*temp_loss1 
                    if i == 2:
                        temp_loss2 = criterion(logits, multi_labels)
                        if torch.isnan(temp_loss2):
                            raise RuntimeError(f"❌ logits_list[{i}]  temp_loss2 = NaN")
                        total_loss += 0.1*temp_loss2 
                    
                    if i == 4: 
                        reg_loss = logits.to(device)
                        if torch.isnan(reg_loss):
                            raise RuntimeError(f"❌ logits_list[{i}] loss_regular = NaN")
                        reg_loss = regular_weight * reg_loss
                        total_loss += reg_loss

    
            total_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += total_loss.item()
            LLM_loss += outputs.loss.item()
            image_loss += temp_loss0.item()
            text_loss += temp_loss1.item()
            LT_loss += temp_loss2.item()
            regular_loss_total += reg_loss.item()
            
            
            
            if rank == 0:
                if (global_step + 1) % 100 == 0:
                    print(f"  [Step {global_step+1}] "
                          f"total={total_loss.item():.4f} LLM={outputs.loss.item():.4f} "
                          f"img={temp_loss0.item():.4f} txt={temp_loss1.item():.4f} "
                          f"LT={temp_loss2.item():.4f} reg={reg_loss.item():.4f}")
                wandb.log({"step": global_step + 1, "step_train_loss": total_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_LLM_loss": outputs.loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_image_loss": temp_loss0.item()})
                wandb.log({"step": global_step + 1, "step_avg_text_loss": temp_loss1.item()})
                wandb.log({"step": global_step + 1, "step_avg_LearnableToken_loss": temp_loss2.item()})
                wandb.log({"step": global_step + 1, "step_avg_regular_loss": reg_loss.item()})

            global_step += 1

            if global_step % eval_steps == 0:
                evaluate_model(rank, world_size, model, val_loaders, device, global_step, max_val_item_count)

        evaluate_model(rank, world_size, model, val_loaders, device, global_step, max_val_item_count)

        # Log training loss to wandb
        avg_train_loss = train_loss / len(train_loader)
        avg_LLM_loss = LLM_loss / len(train_loader)
        avg_image_loss = image_loss / len(train_loader)
        avg_text_loss = text_loss / len(train_loader)
        avg_LT_loss = LT_loss / len(train_loader)
        avg_regular_loss = regular_loss_total / len(train_loader)
    
        
        if rank == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"total_loss={avg_train_loss:.4f} LLM={avg_LLM_loss:.4f} "
                  f"image_cls={avg_image_loss:.4f} text_cls={avg_text_loss:.4f} "
                  f"LT_cls={avg_LT_loss:.4f} regular={avg_regular_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_LLM_loss": avg_LLM_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_image_loss": avg_image_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_text_loss": avg_text_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_LearnableToken_loss": avg_LT_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_regular_loss": avg_regular_loss})


        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            out_put_prefix = './AMD_log'
            output_dir = os.path.join(out_put_prefix,f"./train_{train_time}/epoch_{epoch+1}")
            
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    # Finish the wandb run
    if rank == 0:
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train AMD model on specified dataset")
    parser.add_argument("--AMD-init-pth", type=str, help="AMD model dir")
    parser.add_argument("--dataset-type", type=str, default="DGM4", choices=["docvqa", "cauldron", "vqainstruct","DGM4"], help="Dataset to train on")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for training") 
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=13, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=2000, help="Number of steps between evaluations") 
    parser.add_argument("--run-name", type=str, default='test', help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=2000, help="Maximum number of items to evaluate on during validation")
    parser.add_argument("--regular-weight", type=float, default=0.07, help="loss weight of L_TRP")
    parser.add_argument("--train-js", type=str, default='./train.json', help="json file for train")
    parser.add_argument("--val-js", type=str, default='./val.json', help="json file for val")
    parser.add_argument("--train-domain", type=str, default='NYT', help="News domain of train data")
    parser.add_argument("--seed", type=int, default=12, help="random seed, small is better")
    parser.add_argument("--image-root", type=str, default='/data1/yaxiong/dataset', help="root directory for dataset images, joined with ann['image']")
    
    
    
    
    
    
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(args.AMD_init_pth, args.train_js, args.val_js, world_size, args.dataset_type, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count, args.regular_weight, args.train_domain, args.seed, args.image_root),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    
    main()