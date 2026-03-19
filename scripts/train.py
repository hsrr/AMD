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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from multilabel_metrics import AveragePrecisionMeter
from torchvision.ops.boxes import box_area

OPTION_TEXTS = [
    "A. No.",
    "B. Only face swap.",
    "C. Only face attribute.",
    "D. Only text swap.",
    "E. Face swap and text swap.",
    "F. Face attribute and text swap.",
]

OPTION_MATRIX_VALUES = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -0.33, -0.33, -0.33],
    [-0.33, 1.0, -0.33, -0.33],
    [-0.33, -0.33, 1.0, -0.33],
    [0.5, -0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5, -0.5],
]

OPTION_PREFIX_TO_INDEX = {
    "A.": 0,
    "B.": 1,
    "C.": 2,
    "D.": 3,
    "E.": 4,
    "F.": 5,
}

MULTI_LABEL_TARGETS = torch.tensor(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
    ],
    dtype=torch.long,
)



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
    images, questions, answers,fake_image_box = zip(*batch)
    
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers,fake_image_box


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

def extract_option_index(answer):
    answer = answer.strip()
    for prefix, idx in OPTION_PREFIX_TO_INDEX.items():
        if answer.startswith(prefix):
            return idx

    match = re.search(r"([A-F])\.", answer)
    if match:
        return OPTION_PREFIX_TO_INDEX.get(f"{match.group(1)}.", 0)
    return 0

def get_option_class_labels(answers, device):
    indices = [extract_option_index(ans) for ans in answers]
    return torch.tensor(indices, dtype=torch.long, device=device)

def get_option_matrix(device):
    return torch.tensor(OPTION_MATRIX_VALUES, dtype=torch.float, device=device)

def build_multiclass_logits(logits_list, option_matrix):
    cls_logits = logits_list[:3]
    if any(logits is None for logits in cls_logits):
        return None

    # Reuse the three existing binary heads and decode to 6-way class scores via the provided matrix.
    branch_scores = torch.stack(
        [logits[:, 0] - logits[:, 1] for logits in cls_logits],
        dim=1,
    )
    zero_column = torch.zeros(
        (branch_scores.size(0), 1),
        device=branch_scores.device,
        dtype=branch_scores.dtype,
    )
    branch_scores = torch.cat([branch_scores, zero_column], dim=1)
    return torch.matmul(branch_scores, option_matrix.t())

def get_multi_label(answers,device):
    class_indices = [extract_option_index(ans) for ans in answers]
    multi_label = MULTI_LABEL_TARGETS[class_indices].to(device)
    real_label_pos = [i for i, idx in enumerate(class_indices) if idx == 0]
    return multi_label, real_label_pos

def get_best_option(generated_texts, option_vectors,vectorizer,options,option_labels,device):
    '''批量计算模型的输出对应哪一个选项
    输入是生成的多个文本，和固定选项的向量表示
    '''
    # 将生成文本批量转换为向量
    generated_vectors = vectorizer.transform(generated_texts).toarray()

    # 计算相似度
    similarities = cosine_similarity(generated_vectors, option_vectors)

    # 获取每个生成文本的相似度最高的选项
    best_option_indices = similarities.argmax(axis=1)

    # 返回选项、相似度和对应的01标签
    best_options = [options[i] for i in best_option_indices]
    best_similarities = [similarities[i, best_option_indices[i]] for i in range(len(generated_texts))]

    best_multi_labels = torch.stack([option_labels[i] for i in best_option_indices], dim=0)
    # 对 best_multi_labels 进行归一化
    # best_multi_labels_prob = F.softmax(best_multi_labels.float(), dim=1)
    
    #ori_pos，构造模型输出对应的单分类标签
    pred_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
    real_label_pos = np.where(np.array(best_options) == 'A. No.')[0].tolist()
    # 是A. No.的地方设置为 0 --代表real图文
    pred_label[real_label_pos] = 0
    
    return best_options, best_similarities, best_multi_labels,pred_label

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


def evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count,option_vectors,vectorizer,options,option_labels, classification_only=False):

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            val_item_count = 0
            cls_nums_all = 0
            cls_acc_all = 0 
            IOU_pred = []
            multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
            multi_label_meter.reset()
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                inputs, batch_answers, fake_image_box = batch
                val_item_count += len(inputs)
                generated_ids = model.module.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                
                task_answers = []
                if not classification_only:
                    output_coords = torch.zeros((len(generated_texts), 4)).to(device)
                    true_coords = torch.zeros((len(generated_texts), 4)).to(device)
                
                for i, (generated_text, answers) in enumerate(zip(generated_texts, batch_answers)):

                    full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
                    
                    if '<loc_' in full_answer:
                        task_answers.append(full_answer.split('Manipulated face')[0])
                        if not classification_only:
                            output_coords[i] = parse_coordinates(full_answer).to(device)
                            true_coords[i] = parse_coordinates(answers).to(device)
    
                    else:
                        task_answers.append(full_answer)
                        if not classification_only:
                            true_coords[i] = parse_coordinates(answers).to(device)
                
                
                real_multi_label, real_label_pos = get_multi_label(batch_answers,device)
                real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
                real_label[real_label_pos] = 0
                best_options, _ ,best_multi_labels,pred_label = get_best_option(task_answers, option_vectors,vectorizer,options,option_labels,device)
                
                ##--reeal/fake---##
                cls_nums_all = val_item_count
                cls_acc_all += torch.sum(real_label == pred_label).item()
                
                ##-IoU--##
                if not classification_only:
                    IOU, _ = box_iou(output_coords, true_coords.to(device), test=True)
                    for iou_value in IOU.cpu().tolist():
                        if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                            IOU_pred.append(iou_value)
                        else:
                            IOU_pred.append(0.0)
                ######################################
                            
                ##-multi--##
                multi_label_meter.add(best_multi_labels, real_multi_label)
                
                local_ACC_cls = cls_acc_all / cls_nums_all
                local_IOU_score = sum(IOU_pred)/len(IOU_pred) if len(IOU_pred) > 0 else 0.0
                local_MAP = multi_label_meter.value()[:3].mean().item()


                if val_item_count > max_val_item_count:
                    break
        local_ACC_cls_tensor = torch.tensor(local_ACC_cls, device=device)
        local_IoU_score_tensor = torch.tensor(local_IOU_score, device=device)
        local_MAP_tensor = torch.tensor(local_MAP, device=device)


        ACC_cls = synchronize_metrics(local_ACC_cls_tensor, world_size)
        IoUscore = synchronize_metrics(local_IoU_score_tensor, world_size)
        MAP = synchronize_metrics(local_MAP_tensor, world_size)

        if dist.get_rank() == 0:
            print(f"Rank {rank} - Step {global_step} - ACC perform ({val_name}): {ACC_cls.item()}")
            wandb_payload = {
                f"{val_name}_ACC_cls": ACC_cls.item(),
                f"{val_name}_MAP": MAP.item(),
                "step": global_step
            }
            if not classification_only:
                wandb_payload[f"{val_name}_IoUscore"] = IoUscore.item()
            wandb.log(wandb_payload)
            
    model.train()



def train_model(rank, AMD_init_pth, train_js, val_js, world_size, dataset_name, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000, regular_weight=0.07, train_domain='NYT',random_seed=12, classification_only=False):
    setup(rank, world_size)
    set_seed(random_seed, rank)
    device = torch.device(f"cuda:{rank}")
    train_data=[]
    val_data=[]
    
    logged_task_name = f'AMD_test_{train_domain}' 
    train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_train_name = f'{logged_task_name}_{train_time}' 
    
    criterion = torch.nn.CrossEntropyLoss() 
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    option_matrix = get_option_matrix(device)
    option_labels = [option_matrix[i] for i in range(option_matrix.size(0))]
    options = OPTION_TEXTS
    
    vectorizer = TfidfVectorizer().fit(options)
    option_vectors = vectorizer.transform(options).toarray()
    
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
        with open(train_js, "r") as f:
            train_data = json.load(f)
        with open(val_js, "r") as f:
            val_data = json.load(f)
            
        train_dataset = DGM4_Dataset(split='train',data=train_data)
        val_datasets = {"DGM4": DGM4_Dataset(split='validation',data=val_data)}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        AMD_init_pth, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        AMD_init_pth, trust_remote_code=True
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

    model = DDP(model, device_ids=[rank])

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
        multiclass_loss = 0
        loss_bbox = 0
        loss_giou = 0
        loss_regular = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers,fake_image_box = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            answer_supervision = [ans.split('Manipulated face')[0] if classification_only else ans for ans in answers]

            labels = processor.tokenizer(
                text=answer_supervision,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            total_loss = outputs.loss
            logits_list = outputs.classification_logits_list
            ### logits = [image_classification, text_classification,learnable_token_logits,output_coord,loss_regular]
            step_multiclass_loss = torch.tensor(0.0, device=device)
            step_bbox_loss = torch.tensor(0.0, device=device)
            step_giou_loss = torch.tensor(0.0, device=device)
            step_regular_loss = torch.tensor(0.0, device=device)

            class_labels = get_option_class_labels(answer_supervision, device)
            multiclass_logits = build_multiclass_logits(logits_list, option_matrix)
            if multiclass_logits is None:
                raise RuntimeError("❌ classification logits 不完整，无法计算矩阵多分类损失")
            step_multiclass_loss = criterion(multiclass_logits, class_labels)
            if torch.isnan(step_multiclass_loss):
                raise RuntimeError("❌ 多分类损失 step_multiclass_loss = NaN")
            total_loss += 0.3 * step_multiclass_loss

            for i,logits in enumerate(logits_list):
                if logits is not None:
                    if i == 3 and not classification_only: ## output_coord
                        output_coords = logits.to(device)
                        tensor_fake_image_box = torch.cat(fake_image_box, dim=0).reshape(len(fake_image_box), -1).to(device)
                        step_bbox_loss, step_giou_loss = get_bbox_loss(output_coords, tensor_fake_image_box) 
                        if torch.isnan(step_bbox_loss):
                            raise RuntimeError(f"❌ logits_list[{i}] loss_bbox = NaN")
                        total_loss += 0.1*(step_bbox_loss + step_giou_loss)
                    
                    if i == 4: 
                        step_regular_loss = logits.to(device)
                        if torch.isnan(step_regular_loss):
                            raise RuntimeError(f"❌ logits_list[{i}] loss_regular = NaN")
                        step_regular_loss = regular_weight * step_regular_loss
                        total_loss += step_regular_loss

    
            total_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += total_loss.item()
            LLM_loss += outputs.loss.item()
            multiclass_loss += step_multiclass_loss.item()
            if not classification_only:
                loss_bbox += step_bbox_loss.item()
                loss_giou += step_giou_loss.item()
            loss_regular += step_regular_loss.item()
            
            
            
            if rank == 0:
                wandb.log({"step": global_step + 1, "step_train_loss": total_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_LLM_loss": outputs.loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_multiclass_loss": step_multiclass_loss.item()})
                if not classification_only:
                    wandb.log({"step": global_step + 1, "step_avg_bbox_loss": step_bbox_loss.item()})
                    wandb.log({"step": global_step + 1, "step_avg_giou_loss": step_giou_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_regular_loss": step_regular_loss.item()})
                
            global_step += 1

            if global_step % eval_steps == 0:
                evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count,option_vectors,vectorizer,options,option_labels, classification_only=classification_only)

        evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count,option_vectors,vectorizer,options,option_labels, classification_only=classification_only)

        # Log training loss to wandb
        avg_train_loss = train_loss / len(train_loader)
        avg_LLM_loss = LLM_loss / len(train_loader)
        avg_multiclass_loss = multiclass_loss / len(train_loader)
        avg_bbox_loss = loss_bbox / len(train_loader)
        avg_giou_loss = loss_giou / len(train_loader)
        avg_regular_loss = loss_regular / len(train_loader)
    
        
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_LLM_loss": avg_LLM_loss})
            wandb.log({"epoch": epoch + 1, "epoch_avg_multiclass_loss": avg_multiclass_loss})
            if not classification_only:
                wandb.log({"epoch": epoch + 1, "epoch_avg_bbox_loss": avg_bbox_loss})
                wandb.log({"epoch": epoch + 1, "epoch_avg_giou_loss": avg_giou_loss})
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
    parser.add_argument("--regular-weight", type=int, default=2000, help="loss weight of L_TRP")
    parser.add_argument("--train-js", type=str, default='./train.json', help="json file for train")
    parser.add_argument("--val-js", type=str, default='./val.json', help="json file for val")
    parser.add_argument("--train-domain", type=str, default='NYT', help="News domain of train data")
    parser.add_argument("--seed", type=int, default=12, help="random seed, small is better")
    parser.add_argument("--classification-only", action='store_true', help="Only keep classification-related supervision (disable bbox GT supervision).")
    
    
    
    
    
    
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(args.AMD_init_pth, args.train_js, args.val_js, world_size, args.dataset_type, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count, args.regular_weight, args.train_domain, args.seed, args.classification_only),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    
    main()