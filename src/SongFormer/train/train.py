import argparse
import copy
import importlib
import os
import traceback

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.local_sgd import LocalSGD
from accelerate.utils import LoggerType, set_seed

# from ema import LitEma
from ema_pytorch import EMA
from encodec.balancer import Balancer
from loguru import logger
from omegaconf import OmegaConf
from eval_infer_results_used_in_train import eval_infer_results
from vis_infer_chunk_class_used_in_train import vis_infer_chunk
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from utils.check_nan import NanInfError, check_model_param
from utils.timer import TrainTimer

# for lance
torch.multiprocessing.set_start_method("spawn", force=True)


def save_checkpoint(
    checkpoint_dir,
    model,
    model_ema,
    optimizer,
    scheduler,
    step,
    accelerator,
    wait_for_everyone=True,
):
    if wait_for_everyone:
        accelerator.wait_for_everyone()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt-{}.pt".format(step))
    if accelerator.is_main_process:
        accelerator.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": accelerator.unwrap_model(optimizer).state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "model_ema": model_ema.state_dict(),
                "global_step": step,
            },
            checkpoint_path,
        )

        print("Saved checkpoint: {}".format(checkpoint_path))

        with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
            f.write("model.ckpt-{}.pt".format(step))
    return checkpoint_path


def attempt_to_restore(
    model,
    model_ema,
    optimizer,
    scheduler,
    checkpoint_dir,
    device,
    accelerator,
    keep_training,
    strict=True,
):
    accelerator.wait_for_everyone()

    checkpoint_list = os.path.join(checkpoint_dir, "checkpoint")

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, device)
        if strict:
            accelerator.unwrap_model(model).load_state_dict(checkpoint["model"], True)
            accelerator.unwrap_model(optimizer).load_state_dict(checkpoint["optimizer"])
            if scheduler:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if model_ema and accelerator.is_main_process:
                model_ema.load_state_dict(checkpoint["model_ema"])
        else:
            accelerator.unwrap_model(model).load_state_dict(
                checkpoint["model"], False
            )  # false to change
        if keep_training:
            global_step = checkpoint["global_step"]
        else:
            global_step = 0
        del checkpoint
    else:
        global_step = 0

    return global_step


def load_checkpoint(checkpoint_path, device=None):
    if device:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return checkpoint


def evaluate(model, eval_data_loader, accelerator, global_step):
    model.eval()
    results_by_dataset = {}
    evaluate_num = 0

    with torch.no_grad():
        with TrainTimer(
            step=global_step, name="time/eval_time", accelerator=accelerator
        ):
            for batch in tqdm(eval_data_loader, desc="Evaluating"):
                try:
                    if batch is None:
                        continue
                    batch = {
                        key: (
                            val.to(accelerator.device)
                            if isinstance(val, torch.Tensor)
                            else val
                        )
                        for key, val in batch.items()
                    }

                    assert len(batch["data_ids"]) == 1
                    dataset_id = batch["dataset_ids"].item()

                    # If multi-GPU training is used, the logic here may need to be modified.

                    # if accelerator.num_processes > 1:
                    #     result = model.module.infer_with_metrics(batch, prefix="valid_")
                    # else:
                    #     result = model.infer_with_metrics(batch, prefix="valid_")
                    result = model.ema_model.infer_with_metrics(batch, prefix="valid_")

                    if dataset_id not in results_by_dataset:
                        results_by_dataset[dataset_id] = []

                    results_by_dataset[dataset_id].append(result)
                    evaluate_num += 1

                except Exception as e:
                    logger.error(f"Error in evaluate {dataset_id}: {e}")
                    continue

    flat_result = {}

    # Average per dataset
    for dataset_id, result_list in results_by_dataset.items():
        df = pd.DataFrame(result_list)
        avg_metrics = df.mean().to_dict()
        for k, v in avg_metrics.items():
            flat_result[f"dataset_{dataset_id}_{k}"] = v

    # Overall average
    all_results = [res for results in results_by_dataset.values() for res in results]
    overall_df = pd.DataFrame(all_results)
    overall_metrics = overall_df.mean().to_dict()
    for k, v in overall_metrics.items():
        flat_result[f"overall_{k}"] = v

    return flat_result


def test_metrics(accelerator, model, hparams, ckpt_path, infer_dicts):
    with torch.no_grad():

        def add_dict_prefix(d, prefix: str):
            if not prefix:
                return d
            return {prefix + key: value for key, value in d.items()}

        total_results = {}

        for item in infer_dicts:
            infer_dir = ckpt_path.replace("output/", "infer_output/").replace(".pt", "")
            os.makedirs(infer_dir, exist_ok=True)
            os.makedirs(os.path.join(infer_dir, "visualisation"), exist_ok=True)

            vis_infer_chunk(
                device=accelerator.device,
                model=model,
                embedding_dir=item["embedding_dir"],
                segments_dir=item["ann_dir"],
                eval_id_scp_path=item["eval_id_scp_path"],
                visual_id_list_path=item["visual_id_list_path"],
                dataset_label=item["dataset_label"],
                dataset_ids=item["dataset_ids"],
                hparams=hparams,
                output_dir=infer_dir,
            )

            for result_type in ["normal", "prechorus2chorus", "prechorus2verse"]:
                result_dir = infer_dir.replace(
                    "infer_output", "infer_results/mannual_cbhao/" + result_type
                )
                result_type2prechorus2what = dict(
                    normal=None, prechorus2chorus="chorus", prechorus2verse="verse"
                )
                tmp_result = eval_infer_results(
                    ann_dir=item["ann_dir"],
                    est_dir=infer_dir,
                    output_dir=result_dir,
                    eval_lists_file_path=item["eval_id_scp_path"],
                    prechorus2what=result_type2prechorus2what[result_type],
                )
                tmp_result = tmp_result.to_dict(orient="records")
                assert len(tmp_result) == 1, "There should be only one record"
                tmp_result = tmp_result[0]
                tmp_result = add_dict_prefix(
                    d=tmp_result, prefix=f"{item['infer_name']}_{result_type}/"
                )
                total_results.update(tmp_result)

        total_results = add_dict_prefix(d=total_results, prefix="result_")
        return total_results


def prefix_dict(d, prefix: str):
    if prefix:
        return d
    return {prefix + key: value for key, value in d.items()}


def main(args, hparams):
    assert hasattr(args, "init_seed"), "hparams should have seed attribute"
    set_seed(args.init_seed)
    accelerator = Accelerator(
        log_with=["wandb", LoggerType.MLFLOW],
        project_dir=os.path.join(args.checkpoint_dir, "tracker"),
        gradient_accumulation_steps=hparams.accumulation_steps,
    )

    device = accelerator.device
    rank = accelerator.process_index
    local_rank = accelerator.local_process_index

    tags = []
    if args.tags:
        for tag in args.tags.split("/"):
            tags.append(tag)
    init_kwargs = {
        "wandb": {
            "resume": "allow",
            "name": args.run_name,
        },
        "mlflow": {
            "run_name": args.run_name,
        },
    }

    accelerator.init_trackers(
        "SongFormer",
        config={
            **prefix_dict(vars(copy.deepcopy(args)), "a_"),
            **prefix_dict(copy.deepcopy(hparams), "h_"),
        },
        init_kwargs=init_kwargs,
    )

    def print_rank_0(msg):
        accelerator.print(msg)

    module = importlib.import_module("models." + args.model_name)
    Model = getattr(module, "Model")
    model = Model(hparams)
    params = model.parameters()

    if accelerator.is_main_process:
        model_ema = EMA(model, include_online_model=False, **hparams.ema_kwargs)
        model_ema.to(accelerator.device)

    num_params = 0
    for param in params:
        num_params += torch.prod(torch.tensor(param.size()))

    train_dataset = hydra.utils.instantiate(hparams.train_dataset)
    eval_dataset = hydra.utils.instantiate(hparams.eval_dataset)

    data_loader = DataLoader(
        train_dataset, **hparams.train_dataloader, collate_fn=train_dataset.collate_fn
    )
    eval_data_loader = DataLoader(
        eval_dataset, **hparams.eval_dataloader, collate_fn=eval_dataset.collate_fn
    )

    warmup_steps = hparams.warmup_steps
    total_steps = hparams.total_steps

    balancer = Balancer(
        weights={"loss_section": 1, "loss_function": 1},
        rescale_grads=True,
        monitor=True,
    )

    optimizer = optim.Adam(
        model.parameters(),
        **hparams.optimizer,
    )

    print(f"warmup_steps: {warmup_steps}, total_steps: {total_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,  # warmup steps
        num_training_steps=total_steps,  # total training steps
    )

    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )

    global_step = attempt_to_restore(
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        accelerator=accelerator,
        keep_training=True,
        strict=True,
    )
    print_rank_0(
        f"-------------------------Parameters: {num_params}-----------------------"
    )

    # early stop
    best_hr5 = -float("inf")
    best_acc = -float("inf")
    no_improve_steps = 0
    early_stop_patience = (
        hparams.early_stopping_step
    )  # Stop if there is no improvement in 5 consecutive evaluations

    ran_since_savepoint_loaded = False
    max_steps = args.max_steps
    LOCAL_SGD_STEPS = 8

    eval_ckpt_lists = []
    seek_results_ckpt_lists = []

    for epoch in range(args.max_epochs):
        if no_improve_steps >= early_stop_patience:
            break

        if global_step >= max_steps:
            break
        model.train()
        if rank == 0:
            progress_bar = tqdm(range(len(data_loader)))
        with LocalSGD(
            accelerator=accelerator,
            model=model,
            local_sgd_steps=LOCAL_SGD_STEPS,
            enabled=False,
        ) as local_sgd:
            print("***data_loader length:", len(data_loader))
            for step, batch in enumerate(data_loader):
                if global_step >= max_steps:
                    break
                if batch is None:
                    continue
                with accelerator.accumulate(model):
                    model.train()

                    try:
                        optimizer.zero_grad()

                        with TrainTimer(global_step, "time/forward_time", accelerator):
                            logits, loss, losses = model(batch)

                        with TrainTimer(global_step, "time/backward_time", accelerator):
                            loss_sum = balancer.cal_mix_loss(
                                {
                                    "loss_section": losses["loss_section"],
                                    "loss_function": losses["loss_function"],
                                },
                                list(model.parameters()),
                                accelerator=accelerator,
                            )

                            accelerator.backward(loss_sum)

                        with TrainTimer(global_step, "time/optimize_time", accelerator):
                            if accelerator.sync_gradients:
                                optimizer.step()
                                scheduler.step()
                                local_sgd.step()

                                if accelerator.is_main_process:
                                    model_ema.update()

                        check_model_param(model, step=global_step)

                        if rank == 0 and global_step % args.log_interval == 0:
                            progress_bar.update(args.log_interval)
                            progress_bar.set_description(
                                f"epoch: {epoch:03}, step: {global_step:06}, loss_awl: {loss_sum.item():.2f}"
                            )

                        if global_step % args.log_interval == 0 and rank == 0:
                            accelerator.log(
                                {
                                    **balancer.metrics,
                                    "training/epoch": epoch,
                                    # "training/loss": loss.item(),
                                    "training/loss_awl": loss_sum.item(),
                                    "training/loss_function": losses[
                                        "loss_function"
                                    ].item(),
                                    "training/loss_section": losses[
                                        "loss_section"
                                    ].item(),
                                    "training/learning_rate": scheduler.get_lr()[0],
                                    "training/batch_size": int(
                                        hparams.train_dataloader.batch_size
                                    ),
                                    "training/local_sgd_steps": LOCAL_SGD_STEPS,
                                    "training/num_of_gpu": accelerator.num_processes,
                                },
                                step=global_step,
                            )
                        if (
                            accelerator.sync_gradients
                            and global_step % args.eval_interval == 0
                        ):
                            accelerator.wait_for_everyone()
                            if rank == 0:
                                model.eval()

                                eval_res = evaluate(
                                    model=model_ema,
                                    eval_data_loader=eval_data_loader,
                                    accelerator=accelerator,
                                    global_step=global_step,
                                )

                                eval_res = prefix_dict(d=eval_res, prefix="eval/")
                                accelerator.log(
                                    eval_res,
                                    step=global_step,
                                )

                                hr5 = eval_res.get("dataset_5_HitRate_0.5F", 0)
                                acc = eval_res.get("dataset_5_acc", 0)

                                if hr5 > best_hr5 or acc > best_acc:
                                    print(
                                        f"Eval improved: dataset_5_HitRate_0.5F {hr5:.4f} (prev {best_hr5:.4f}), dataset_5_acc {acc:.4f} (prev {best_acc:.4f})"
                                    )
                                    best_hr5 = max(best_hr5, hr5)
                                    best_acc = max(best_acc, acc)
                                    no_improve_steps = 0

                                    tmp_ckpt_path = save_checkpoint(
                                        checkpoint_dir=args.checkpoint_dir,
                                        model=model,
                                        model_ema=model_ema,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        step=global_step,
                                        accelerator=accelerator,
                                        wait_for_everyone=False,
                                    )
                                    eval_ckpt_lists.append(
                                        (tmp_ckpt_path, float(acc), global_step)
                                    )
                                    with open(
                                        os.path.join(
                                            args.checkpoint_dir, "early_stop.log"
                                        ),
                                        "a",
                                    ) as f:
                                        f.write(
                                            f"model.ckpt-{global_step}.pt hr5: {hr5}, acc: {acc} \n"
                                        )
                                    print("Saved best checkpoint at step", global_step)
                                else:
                                    logger.warning("write a ckpt not improved!")
                                    save_checkpoint(
                                        checkpoint_dir=args.checkpoint_dir,
                                        model=model,
                                        model_ema=model_ema,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        step=global_step,
                                        accelerator=accelerator,
                                        wait_for_everyone=False,
                                    )
                                    no_improve_steps += 1
                                    print(
                                        f"No improvement for {no_improve_steps} eval steps."
                                    )
                                    if no_improve_steps >= early_stop_patience:
                                        print("Early stopping triggered.")
                                        break

                        accelerator.wait_for_everyone()
                        if (
                            accelerator.sync_gradients
                            and global_step % args.save_interval == 0
                        ):
                            print("save checkpoint", global_step)
                            checkpoint_path = save_checkpoint(
                                checkpoint_dir=args.checkpoint_dir,
                                model=model,
                                model_ema=model_ema,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=global_step,
                                accelerator=accelerator,
                                wait_for_everyone=False,
                            )
                            if (
                                not hasattr(hparams, "infer_dicts")
                                or hparams.infer_dicts is None
                                or len(hparams.infer_dicts) == 0
                            ):
                                logger.error(
                                    "No infer_dicts provided, skipping test_metrics for ckpt",
                                )
                            else:
                                test_metrics_ret = test_metrics(
                                    accelerator=accelerator,
                                    model=model_ema.ema_model,
                                    hparams=hparams,
                                    ckpt_path=checkpoint_path,
                                    infer_dicts=hparams.infer_dicts,
                                )

                                accelerator.log(
                                    test_metrics_ret,
                                    step=global_step,
                                )
                            seek_results_ckpt_lists.append(checkpoint_path)

                        if accelerator.sync_gradients:
                            global_step += 1

                        ran_since_savepoint_loaded = True
                    except NanInfError as e:
                        print(e)
                        exit(-1)
                    except Exception:
                        traceback.print_exc()
                        # print(features["filenames"], features["seq_lens"], features["seqs"].shape)
                        optimizer.zero_grad()

    eval_ckpt_lists.sort(key=lambda x: x[1], reverse=True)
    seek_results_ckpt_lists = set(seek_results_ckpt_lists)

    for x in eval_ckpt_lists[:2]:
        print(f"Eval ckpt: {x[0]}, acc: {x[1]}")
        if x[0] not in seek_results_ckpt_lists:
            if (
                not hasattr(hparams, "infer_dicts")
                or hparams.infer_dicts is None
                or len(hparams.infer_dicts) == 0
            ):
                logger.error(
                    "No infer_dicts provided, skipping test_metrics for ckpt",
                    x[0],
                )
                continue
            ckpt = load_checkpoint(x[0], device=device)
            model_ema.load_state_dict(ckpt["model_ema"])

            test_metrics_ret = test_metrics(
                accelerator=accelerator,
                model=model_ema.ema_model,
                hparams=hparams,
                ckpt_path=x[0],
                infer_dicts=hparams.infer_dicts,
            )

            accelerator.log(
                test_metrics_ret,
                step=x[2],
            )

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script arguments")

    # ---------------- Must be specified via command line ----------------
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--init_seed", type=int, required=True, help="Random seed for initialization"
    )

    # ---------------- Optional parameters (can override configuration file) ----------------
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (overrides config if provided)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (overrides config if provided)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--tags", type=str, default=None, help="Optional tags for experiment tracking"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Interval (in steps) to save checkpoints",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="Interval (in steps) to run evaluation",
    )

    # ---------------- Training process control parameters ----------------
    parser.add_argument(
        "--not_keep_step",
        action="store_true",
        help="If set, do not keep the training step",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval (in steps)"
    )

    args = parser.parse_args()

    # Read configuration file
    hp = OmegaConf.load(args.config)

    # Ensure init_seed exists
    assert hasattr(args, "init_seed"), "args should have an init_seed attribute"

    if args.run_name is None:
        args.run_name = str(hp.args.run_name) + "_" + str(args.init_seed)
    if args.model_name is None:
        args.model_name = hp.args.model_name
    if args.save_interval is None:
        args.save_interval = hp.args.save_interval
    if args.eval_interval is None:
        args.eval_interval = hp.args.eval_interval
    if args.checkpoint_dir is None:
        args.checkpoint_dir = hp.args.checkpoint_dir + "_" + str(args.init_seed)
    if args.max_epochs is None:
        args.max_epochs = hp.args.max_epochs
    if args.max_steps is None:
        args.max_steps = hp.args.max_steps
    if args.tags is None:
        args.tags = hp.args.tags
    args.keep_step = not args.not_keep_step

    main(args, hp)
