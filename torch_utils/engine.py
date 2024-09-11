import math
import sys
import time

import numpy as np
import torch
import torchvision.models.detection.mask_rcnn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from utils.general import save_validation_results

"""Computes the mean average precision at different intersection over union (IoU) thresholds, 
specifically at 0.5."""
mean_average_precision_50 = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=[0.5],
    extended_summary=True,
)

"""Computes the mean average precision at different intersection over union (IoU) thresholds,
specifically averaging over the IoU thresholds from 0.5 to 0.95, with steps of 0.05."""
mean_average_precision_50_95 = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    rec_thresholds=np.linspace(0, 1, 1001).tolist(),
    extended_summary=True,
)

def train_one_epoch(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step_counter += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]


        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(data_loader)))

    return (
        metric_logger, 
        batch_loss_list, 
        batch_loss_cls_list, 
        batch_loss_box_reg_list, 
        batch_loss_objectness_list, 
        batch_loss_rpn_list
    )

def validate_one_epoch(
    model,
    val_dataloader,
    device,
    epoch,
    val_loss_hist,
    print_freq
):
    model.train()  # Set the model to train mode to get losses
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Validation Epoch: [{epoch}]"

    # Lists to store validation losses
    val_loss_list = []
    val_loss_cls_list = []
    val_loss_box_reg_list = []
    val_loss_objectness_list = []
    val_loss_rpn_list = []
    precision = []
    recall = []

    with torch.inference_mode():  # Disable gradient computation
        for images, targets in metric_logger.log_every(val_dataloader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)

            # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Validation Loss is {loss_value}, stopping validation")
                print(loss_dict_reduced)
                break

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

            # Append losses to respective lists
            val_loss_list.append(loss_value)
            val_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
            val_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
            val_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
            val_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
            val_loss_hist.send(loss_value)

            # Now, compute the predictions (instead of the loss, because we are in eval mode in this
            # section, then save precision and recall values
            model.eval()
            predictions = model(images, targets)
            precision_recall = compute_precision_recall_metrics(predictions, targets)
            precision.append(precision_recall["precision"])
            recall.append(precision_recall["recall"])
            model.train()

    # Compute sum of all batch losses
    sum_loss = sum(val_loss_list)
    sum_loss_cls = sum(val_loss_cls_list)
    sum_loss_box_reg = sum(val_loss_box_reg_list)
    sum_loss_objectness = sum(val_loss_objectness_list)
    sum_loss_rpn = sum(val_loss_rpn_list)
    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)

    return (
        metric_logger,
        sum_loss,
        sum_loss_cls,
        sum_loss_box_reg,
        sum_loss_objectness,
        sum_loss_rpn,
        avg_precision,
        avg_recall
    )

def get_final_precision_recall_f1_tables(model, val_dataloader, device, print_freq) -> None:
    """Logs Precision-Recall, Precision-Confidence, Recall-Confidence, and F1-Confidence tables
    to wandb. It uses the `model` to predict the given `val_dataloader`."""
    with torch.inference_mode():  # Disable gradient computation
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Table Metrics "

        all_predictions = []
        all_labels = []
        for images, targets in metric_logger.log_every(val_dataloader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

            # Forward pass
            predictions = model(images, targets)
            predictions = utils.preds_or_target_to_tensor(predictions)

            all_predictions.extend(predictions)
            all_labels.extend(targets)

    # Get the precision, recall, and scores arrays
    pr_precision, pr_recall, f1, pr_scores = compute_precision_recall_metrics(
        all_predictions, all_labels, return_curves=True
    )["curves"]

    # Now use these processed arrays to create wandb tables
    precision_recall_data = [["Guitar-necks", p, r] for p, r in zip(pr_precision, pr_recall)]
    precision_confidence_data = [["Guitar-necks", p, c] for p, c in zip(pr_precision, pr_scores)]
    recall_confidence_data = [["Guitar-necks", r, c] for r, c in zip(pr_recall, pr_scores)]
    f1_confidence_data = [["Guitar-necks", f, c] for f, c in zip(f1, pr_scores)]

    # Return the data for the tables with the names that will be used to log them to wandb
    return {
        "curves/Precision-Recall(B)_table": precision_recall_data,
        "curves/Precision-Confidence(B)_table": precision_confidence_data,
        "curves/Recall-Confidence(B)_table": recall_confidence_data,
        "curves/F1-Confidence(B)_table": f1_confidence_data,
    }

def compute_precision_recall_metrics(
    predictions: list[dict[str, torch.Tensor]], 
    targets: list[dict[str, torch.Tensor]], 
    return_curves=False
) -> dict[str, float]:
    """Returns a `dict` containing the precision, and the recall of objects using
    the `mean_average_precision_50_95` metric from `torchmetrics.detection.mean_ap`.
    See: https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html"""
    predictions = utils.preds_or_target_to_tensor(predictions)
    targets = utils.preds_or_target_to_tensor(targets)

    ap50_95 = mean_average_precision_50_95(predictions, targets)

    precision = ap50_95["map"]
    if precision is None or precision < 0:
        precision = ap50_95["precision"]
        precision = precision[precision >= 0].mean().item()

    recall, items = 0.0, 0.0
    for recall_key in ["mar_1", "mar_10", "mar_100"]:
        recall_key = ap50_95[recall_key]
        if recall_key is not None and recall_key >= 0:
            recall += recall_key
            items += 1
    if items == 0:
        recall = ap50_95["recall"]
        recall = recall[recall >= 0].mean().item()
    else:
        recall = recall / items

    if return_curves:
        precision_curve, recall_curve, f1_curve, scores = precision_recall_curves(ap50_95)
        return {
            "precision": precision,
            "recall": recall,
            "curves": (precision_curve, recall_curve, f1_curve, scores)
        }

    return {
        "precision": precision,
        "recall": recall,
    }


def precision_recall_curves(metrics: dict) -> tuple:
    """Processes a metric in a value of the `dict` returned by the `compute_metrics()` function with
    regards to "mAP50" and "mAP50-95", and returns the precision, recall, f1 score, and scores,
    needed for plotting the Precision-Recall curve, Recall-Confidence curve, Precision-Confidence
    curve, and F1-confidence curve."""
    precision = metrics["precision"].detach().clone()  # (T, R, K, A, M)
    scores = metrics["scores"].detach().clone()  # (T, R, K, A, M)

    # Process Precision-Recall curve data
    precision[precision < 0] = torch.nan
    precision = precision.nanmean(axis=(0, 2, 3, 4))  # (R,)

    precision = utils.smooth(precision, 0.05)

    # The recall values are implicitly [0, 0.01, 0.02, ..., 0.99, 1]
    recall = np.linspace(0, 1, 1001)

    recall = utils.smooth(recall, 0.05)

    # Process Recall-Confidence curve data
    scores[scores < 0] = torch.nan
    scores = scores.nanmean(axis=(0, 2, 3, 4)).flatten()

    scores = utils.smooth(scores, 0.05)

    # If the last score associated with the last precision is not 1, we need to add a point at
    # (1, precision[-1]) to the precision curve. Same with recall.
    if scores[-1] != 1:
        precision = np.append(precision, precision.max())
        recall = np.append(recall, recall.min())
        scores = np.append(scores, 1)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1, scores


def model_performance_analysis(model, verbose=True) -> dict:
    """Analyzes the model's performance in terms of the number of parameters, the number of
    GigaFLOPs, and the inference speed in milliseconds.

    Returns
    -------
    dict
        A dictionary with keys being "model/parameters", "model/GFLOPs", and
        "model/speed_PyTorch(ms)".
    """
    # Put model in eval mode for analysis
    model.eval()

    # Perform model performance analysis
    num_params = utils.count_parameters(model)
    gflops = utils.estimate_gflops(model)
    inference_speed = utils.measure_inference_speed(model)

    if verbose:
        print(f"Model Performance:")
        print(f"Parameters: {num_params}")
        print(f"GFLOPs: {gflops:.2f}")
        print(f"Inference Speed: {inference_speed:.2f} ms")

    return {
        "model/parameters": num_params,
        "model/GFLOPs": gflops,
        "model/speed_PyTorch(ms)": inference_speed,
    }


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model, 
    data_loader, 
    device, 
    save_valid_preds=False,
    out_dir=None,
    classes=None,
    colors=None
):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if save_valid_preds and counter == 1:
            # The validation prediction image which is saved to disk
            # is returned here which is again returned at the end of the
            # function for WandB logging.
            val_saved_image = save_validation_results(
                images, outputs, counter, out_dir, classes, colors
            )
        elif save_valid_preds == False and counter == 1:
            val_saved_image = np.ones((1, 64, 64, 3))
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats, val_saved_image
