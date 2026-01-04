import os
import sys
import re
import math
import time
import gc
import pickle
import importlib
from glob import glob
from pathlib import Path
from contextlib import nullcontext
import multiprocessing as mp

import numpy as np
from scipy import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import sklearn.preprocessing
from sklearn.utils import class_weight

import psutil
import utils
import models


# -------------------------
# Utility: format seconds
# -------------------------
def _fmt_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _pick_device() -> torch.device:
    # Priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _best_num_workers(default=4) -> int:
    """
    On Mac, too many workers can cause jitter/resource contention:
    default to 4 (you can override with environment variable ISIC_NUM_WORKERS)
    """
    env = os.environ.get("ISIC_NUM_WORKERS", "").strip()
    if env != "":
        try:
            return max(0, int(env))
        except:
            pass

    # Physical core count
    try:
        phys = psutil.cpu_count(logical=False) or 0
    except:
        phys = 0
    if phys <= 0:
        phys = os.cpu_count() or 4
    return int(min(default, max(0, phys)))


def main():
    # Multiprocessing DataLoader compatibility (fixes the bootstrapping error you hit)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if len(sys.argv) < 4:
        print("Usage: python train.py <pc_cfg> <model_cfg> <gpu/cpu> [cv_subset]")
        sys.exit(1)


    # Config loading (keep original logic)
    mdlParams = {}

    pc_cfg = importlib.import_module("pc_cfgs." + sys.argv[1])
    mdlParams.update(pc_cfg.mdlParams)

    model_cfg = importlib.import_module("cfgs." + sys.argv[2])
    mdlParams_model = model_cfg.init(mdlParams)
    mdlParams.update(mdlParams_model)

    mdlParams["trainSetState"] = "train"
    # Per-cfg separate save directory
    mdlParams["saveDirBase"] = os.path.join(mdlParams["saveDir"], sys.argv[2])

    # Keep arguments like "gpu0" as-is (but on Mac it doesn't affect MPS)
    mdlParams["numGPUs"] = [0]  
    if "gpu" in sys.argv[3]:
        gpu_id = [int(s) for s in re.findall(r"\d+", sys.argv[3])][-1]
        mdlParams["numGPUs"] = [gpu_id]

        cuda_str = ""
        for i in range(len(mdlParams["numGPUs"])):
            cuda_str = cuda_str + str(mdlParams["numGPUs"][i])
            if i != len(mdlParams["numGPUs"]) - 1:  
                cuda_str = cuda_str + ","
        print("Devices to use:", cuda_str)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

    if len(sys.argv) > 4:
        mdlParams["cv_subset"] = [int(s) for s in re.findall(r"\d+", sys.argv[4])]
        print("Training validation sets", mdlParams["cv_subset"])

    # Whether a validation set exists
    if "valIndCV" in mdlParams or "valInd" in mdlParams:
        eval_set = "valInd"
        print("Evaluating on validation set during training.")
    else:
        eval_set = "trainInd"
        print("No validation set, evaluating on training set during training.")

    # -------------------------
    # Load CV.pkl
    # -------------------------
    prevFile = Path(os.path.join(mdlParams["saveDirBase"], "CV.pkl"))
    if prevFile.exists():
        print("Part of CV already done")
        with open(mdlParams["saveDirBase"] + "/CV.pkl", "rb") as f:
            allData = pickle.load(f)
    else:
        allData = {
            "f1Best": {},
            "sensBest": {},
            "specBest": {},
            "accBest": {},
            "waccBest": {},
            "aucBest": {},
            "convergeTime": {},
            "bestPred": {},
            "targets": {},
        }

    # CV folds
    if mdlParams.get("cv_subset", None) is not None:
        cv_set = mdlParams["cv_subset"]
    else:
        cv_set = range(mdlParams["numCV"])

    # -------------------------
    # Device selection (key: use MPS, no longer stuck on CPU)
    # -------------------------
    device = _pick_device()
    print("Torch:", torch.__version__)
    if hasattr(torch.backends, "mps"):
        print("mps_built", torch.backends.mps.is_built())
        print("mps_available", torch.backends.mps.is_available())
    print("cuda", torch.cuda.is_available())
    print("Using device:", device)

    # Some performance config for MPS / CUDA
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # DataLoader performance params
    num_workers = _best_num_workers(default=4)
    # pin_memory is meaningless on MPS and will warn, so disable it
    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)
    prefetch_factor = 2 if num_workers > 0 else None

    # Batch-level logging frequency (override via env var ISIC_BATCH_LOG_EVERY)
    env_log_every = os.environ.get("ISIC_BATCH_LOG_EVERY", "").strip()
    if env_log_every:
        try:
            batch_log_every = max(1, int(env_log_every))
        except:
            batch_log_every = 50
    else:
        batch_log_every = 50

    # -------------------------
    # Train each fold
    # -------------------------
    for cv in cv_set:
        already_trained = False

        if "valIndCV" in mdlParams:
            mdlParams["saveDir"] = os.path.join(mdlParams["saveDirBase"], "CVSet" + str(cv))
            if os.path.isdir(mdlParams["saveDirBase"]) and os.path.isdir(mdlParams["saveDir"]):
                all_max_iter = []
                for name in os.listdir(mdlParams["saveDir"]):
                    int_list = [int(s) for s in re.findall(r"\d+", name)]
                    if len(int_list) > 0:
                        all_max_iter.append(int_list[-1])
                all_max_iter = np.array(all_max_iter)
                if len(all_max_iter) > 0 and np.max(all_max_iter) >= mdlParams["training_steps"]:
                    print(f"Fold {cv} already fully trained with {int(np.max(all_max_iter))} iterations")
                    already_trained = True

        if already_trained:
            continue

        print("CV set", cv)

        importlib.reload(models)

        modelVars = {}
        modelVars["device"] = device

        # Indices for the current fold
        mdlParams["trainInd"] = mdlParams["trainIndCV"][cv]
        if "valIndCV" in mdlParams:
            mdlParams["valInd"] = mdlParams["valIndCV"][cv]

        if "valIndCV" in mdlParams:
            mdlParams["saveDir"] = os.path.join(mdlParams["saveDirBase"], "CVSet" + str(cv))
        else:
            mdlParams["saveDir"] = mdlParams["saveDirBase"]

        os.makedirs(mdlParams["saveDirBase"], exist_ok=True)

        # Whether to load checkpoint (only restore if checkpoint files actually exist in the directory)
        os.makedirs(mdlParams["saveDir"], exist_ok=True)

        ckpts = sorted(glob(os.path.join(mdlParams["saveDir"], "checkpoint-*.pt")))
        ckpts = [p for p in ckpts if "checkpoint_best" not in os.path.basename(p)]
        bests = sorted(glob(os.path.join(mdlParams["saveDir"], "checkpoint_best-*.pt")))

        load_old = 1 if (len(ckpts) > 0 or len(bests) > 0) else 0
        if load_old:
            print("Loading old model")

        # Track training process (keep original structure)
        save_dict = {
            "acc": [],
            "loss": [],
            "wacc": [],
            "auc": [],
            "sens": [],
            "spec": [],
            "f1": [],
            "step_num": [],
        }
        if mdlParams["print_trainerr"]:
            save_dict_train = {
                "acc": [],
                "loss": [],
                "wacc": [],
                "auc": [],
                "sens": [],
                "spec": [],
                "f1": [],
                "step_num": [],
            }

        # subtract_set_mean
        if mdlParams["subtract_set_mean"] == 1:
            mdlParams["setMean"] = np.mean(mdlParams["images_means"][mdlParams["trainInd"], :], (0))
            print("Set Mean", mdlParams["setMean"])

        # -------------------------
        # class_weights
        # -------------------------
        class_weights_arr = np.ones((mdlParams["numClasses"],), dtype=np.float32)

        if mdlParams["balance_classes"] < 3 or mdlParams["balance_classes"] in [7, 11]:
            class_weights_arr = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(np.argmax(mdlParams["labels_array"][mdlParams["trainInd"], :], 1)),
                y=np.argmax(mdlParams["labels_array"][mdlParams["trainInd"], :], 1),
            ).astype(np.float32)
            print("Current class weights", class_weights_arr)
            class_weights_arr = class_weights_arr * mdlParams["extra_fac"]
            print("Current class weights with extra", class_weights_arr)

        elif mdlParams["balance_classes"] in [3, 4]:
            not_one_hot = np.argmax(mdlParams["labels_array"], 1)
            mdlParams["class_indices"] = []
            for i in range(mdlParams["numClasses"]):
                mdlParams["class_indices"].append(np.where(not_one_hot == i)[0])
                mdlParams["class_indices"][i] = np.setdiff1d(mdlParams["class_indices"][i], mdlParams["valInd"])

        elif mdlParams["balance_classes"] in [5, 6, 13]:
            class_weights_arr = (1.0 / np.mean(mdlParams["labels_array"][mdlParams["trainInd"], :], axis=0)).astype(
                np.float32
            )
            print("Current class weights", class_weights_arr)
            if isinstance(mdlParams["extra_fac"], float):
                class_weights_arr = np.power(class_weights_arr, mdlParams["extra_fac"]).astype(np.float32)
            else:
                class_weights_arr = (class_weights_arr * mdlParams["extra_fac"]).astype(np.float32)
            print("Current class weights with extra", class_weights_arr)

        elif mdlParams["balance_classes"] == 9:
            print("Balance 9")
            indices_ham = mdlParams["trainInd"][mdlParams["trainInd"] < 25331]
            if mdlParams["numClasses"] == 9:
                cw_ = (1.0 / np.mean(mdlParams["labels_array"][indices_ham, :8], axis=0)).astype(np.float32)
                class_weights_arr = np.zeros([mdlParams["numClasses"]], dtype=np.float32)
                class_weights_arr[:8] = cw_
                class_weights_arr[-1] = float(np.max(cw_))
            else:
                class_weights_arr = (1.0 / np.mean(mdlParams["labels_array"][indices_ham, :], axis=0)).astype(np.float32)

            print("Current class weights", class_weights_arr)
            if isinstance(mdlParams["extra_fac"], float):
                class_weights_arr = np.power(class_weights_arr, mdlParams["extra_fac"]).astype(np.float32)
            else:
                class_weights_arr = (class_weights_arr * mdlParams["extra_fac"]).astype(np.float32)
            print("Current class weights with extra", class_weights_arr)

        # Meta scaler
        if mdlParams.get("meta_features", None) is not None and mdlParams["scale_features"]:
            mdlParams["feature_scaler_meta"] = sklearn.preprocessing.StandardScaler().fit(
                mdlParams["meta_array"][mdlParams["trainInd"], :]
            )
            print(
                "scaler mean",
                mdlParams["feature_scaler_meta"].mean_,
                "var",
                mdlParams["feature_scaler_meta"].var_,
            )

        # -------------------------
        # DataLoaders
        # -------------------------
        dataset_train = utils.ISICDataset(mdlParams, "trainInd")

        val_split_name = "valInd" if ("valInd" in mdlParams) else "trainInd"
        dataset_val = utils.ISICDataset(mdlParams, val_split_name)

        dl_kwargs = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

        if mdlParams["multiCropEval"] > 0:
            modelVars["dataloader_valInd"] = DataLoader(
                dataset_val,
                batch_size=mdlParams["multiCropEval"],
                shuffle=False,
                **dl_kwargs,
            )
        else:
            modelVars["dataloader_valInd"] = DataLoader(
                dataset_val,
                batch_size=mdlParams["batchSize"],
                shuffle=False,
                **dl_kwargs,
            )

        if mdlParams["balance_classes"] in [12, 13]:
            strat_sampler = utils.StratifiedSampler(mdlParams)
            modelVars["dataloader_trainInd"] = DataLoader(
                dataset_train,
                batch_size=mdlParams["batchSize"],
                sampler=strat_sampler,
                drop_last=True,
                **dl_kwargs,
            )
        else:
            modelVars["dataloader_trainInd"] = DataLoader(
                dataset_train,
                batch_size=mdlParams["batchSize"],
                shuffle=True,
                drop_last=True,
                **dl_kwargs,
            )

        print(f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")

        # -------------------------
        # Model
        # -------------------------
        modelVars["model"] = models.getModel(mdlParams)()

        # Load trained model (meta case)
        if mdlParams.get("meta_features", None) is not None:
            files = glob(mdlParams["model_load_path"] + "/CVSet" + str(cv) + "/*")
            global_steps = np.zeros([len(files)])
            for i in range(len(files)):
                if "best" not in files[i]:
                    continue
                if "checkpoint" not in files[i]:
                    continue
                nums = [int(s) for s in re.findall(r"\d+", files[i])]
                global_steps[i] = nums[-1]

            chkPath = (
                mdlParams["model_load_path"]
                + "/CVSet"
                + str(cv)
                + "/checkpoint_best-"
                + str(int(np.max(global_steps)))
                + ".pt"
            )
            print("Restoring lesion-trained CNN for meta data training:", chkPath)
            state = torch.load(chkPath, map_location="cpu", weights_only=False)

            curr_model_dict = modelVars["model"].state_dict()
            for name, param in state["state_dict"].items():
                if isinstance(param, nn.Parameter):
                    param = param.data
                if name in curr_model_dict and curr_model_dict[name].shape == param.shape:
                    curr_model_dict[name].copy_(param)
                else:
                    print("not restored", name, getattr(param, "shape", None))

        if "Dense" in mdlParams["model_type"]:
            if mdlParams["input_size"][0] != 224:
                modelVars["model"] = utils.modify_densenet_avg_pool(modelVars["model"])
            num_ftrs = modelVars["model"].classifier.in_features
            modelVars["model"].classifier = nn.Linear(num_ftrs, mdlParams["numClasses"])
        elif "dpn" in mdlParams["model_type"]:
            num_ftrs = modelVars["model"].classifier.in_channels
            modelVars["model"].classifier = nn.Conv2d(num_ftrs, mdlParams["numClasses"], [1, 1])
        elif "efficient" in mdlParams["model_type"]:
            num_ftrs = modelVars["model"]._fc.in_features
            modelVars["model"]._fc = nn.Linear(num_ftrs, mdlParams["numClasses"])
        elif "wsl" in mdlParams["model_type"]:
            num_ftrs = modelVars["model"].fc.in_features
            modelVars["model"].fc = nn.Linear(num_ftrs, mdlParams["numClasses"])
        else:
            num_ftrs = modelVars["model"].last_linear.in_features
            modelVars["model"].last_linear = nn.Linear(num_ftrs, mdlParams["numClasses"])

        # meta case
        if mdlParams.get("meta_features", None) is not None:
            if mdlParams["freeze_cnn"]:
                for param in modelVars["model"].parameters():
                    param.requires_grad = False
                if "efficient" in mdlParams["model_type"]:
                    for param in modelVars["model"]._fc.parameters():
                        param.requires_grad = True
                elif "wsl" in mdlParams["model_type"]:
                    for param in modelVars["model"].fc.parameters():
                        param.requires_grad = True
                else:
                    for param in modelVars["model"].last_linear.parameters():
                        param.requires_grad = True
            else:
                for param in modelVars["model"].parameters():
                    param.is_cnn_param = True
                if hasattr(modelVars["model"], "_fc"):
                    for param in modelVars["model"]._fc.parameters():
                        param.is_cnn_param = False

            modelVars["model"] = models.modify_meta(mdlParams, modelVars["model"])
            for param in modelVars["model"].parameters():
                if not hasattr(param, "is_cnn_param"):
                    param.is_cnn_param = False

        # to device
        modelVars["model"] = modelVars["model"].to(modelVars["device"])

        # channels_last
        if modelVars["device"].type in ["cuda", "mps"]:
            try:
                modelVars["model"] = modelVars["model"].to(memory_format=torch.channels_last)
            except Exception:
                pass

        # DataParallel only for multi-GPU CUDA
        if len(mdlParams["numGPUs"]) > 1 and modelVars["device"].type == "cuda":
            modelVars["model"] = nn.DataParallel(modelVars["model"])

        # -------------------------
        # criterion
        # -------------------------
        weight_tensor = torch.tensor(class_weights_arr, dtype=torch.float32, device=modelVars["device"])

        if mdlParams.get("focal_loss", False):
            modelVars["criterion"] = utils.FocalLoss(alpha=class_weights_arr.tolist())
        elif mdlParams["balance_classes"] in [0, 3, 12]:
            modelVars["criterion"] = nn.CrossEntropyLoss()
        elif mdlParams["balance_classes"] == 8:
            modelVars["criterion"] = nn.CrossEntropyLoss(reduction="none")
        elif mdlParams["balance_classes"] in [6, 7]:
            modelVars["criterion"] = nn.CrossEntropyLoss(weight=weight_tensor, reduction="none")
        elif mdlParams["balance_classes"] == 10:
            modelVars["criterion"] = utils.FocalLoss(mdlParams["numClasses"])
        elif mdlParams["balance_classes"] == 11:
            modelVars["criterion"] = utils.FocalLoss(mdlParams["numClasses"], alpha=weight_tensor)
        else:
            modelVars["criterion"] = nn.CrossEntropyLoss(weight=weight_tensor)

        # -------------------------
        # optimizer / scheduler
        # -------------------------
        if mdlParams.get("meta_features", None) is not None:
            if mdlParams["freeze_cnn"]:
                modelVars["optimizer"] = optim.Adam(
                    filter(lambda p: p.requires_grad, modelVars["model"].parameters()),
                    lr=mdlParams["learning_rate_meta"],
                )
            else:
                modelVars["optimizer"] = optim.Adam(
                    [
                        {
                            "params": filter(lambda p: not p.is_cnn_param, modelVars["model"].parameters()),
                            "lr": mdlParams["learning_rate_meta"],
                        },
                        {
                            "params": filter(lambda p: p.is_cnn_param, modelVars["model"].parameters()),
                            "lr": mdlParams["learning_rate"],
                        },
                    ],
                    lr=mdlParams["learning_rate"],
                )
        else:
            modelVars["optimizer"] = optim.Adam(modelVars["model"].parameters(), lr=mdlParams["learning_rate"])

        modelVars["scheduler"] = lr_scheduler.StepLR(
            modelVars["optimizer"],
            step_size=mdlParams["lowerLRAfter"],
            gamma=1 / np.float32(mdlParams["LRstep"]),
        )

        modelVars["softmax"] = nn.Softmax(dim=1)

        # -------------------------
        # Restore from checkpoint
        # -------------------------
        if load_old:
            # ✅ FIX: Prefer restoring checkpoint-*.pt; if only best exists, restore checkpoint_best-*.pt
            ckpts = sorted(glob(os.path.join(mdlParams["saveDir"], "checkpoint-*.pt")))
            ckpts = [p for p in ckpts if "checkpoint_best" not in p]
            if len(ckpts) > 0:
                chkPath = ckpts[-1]
            else:
                bests = sorted(glob(os.path.join(mdlParams["saveDir"], "checkpoint_best-*.pt")))
                if len(bests) == 0:
                    raise FileNotFoundError(f"No checkpoint found in {mdlParams['saveDir']}")
                chkPath = bests[-1]

            print("Restoring:", chkPath)
            state = torch.load(chkPath, map_location="cpu", weights_only=False)
            modelVars["model"].load_state_dict(state["state_dict"])
            modelVars["optimizer"].load_state_dict(state["optimizer"])
            start_epoch = state["epoch"] + 1
            mdlParams["valBest"] = state.get("valBest", 1000)
            mdlParams["lastBestInd"] = state.get("lastBestInd", -1)
        else:
            start_epoch = 1
            mdlParams["lastBestInd"] = -1
            mdlParams["valBest"] = 1000

        numBatchesTrain = len(modelVars["dataloader_trainInd"])
        print("Train batches", numBatchesTrain)

        # -------------------------
        # Training loop + ETA
        # -------------------------
        train_start_time = time.time()
        print("Start training...")

        for step in range(start_epoch, mdlParams["training_steps"] + 1):
            epoch_start = time.time()

            # LR schedule step
            if step >= mdlParams["lowerLRat"] - mdlParams["lowerLRAfter"]:
                modelVars["scheduler"].step()

            modelVars["model"].train()

            running_loss = 0.0
            batch_times = []

            for j, (inputs, labels, indices) in enumerate(modelVars["dataloader_trainInd"]):
                t0 = time.time()

                # move data
                if mdlParams.get("meta_features", None) is not None:
                    inputs[0] = inputs[0].to(modelVars["device"])
                    inputs[1] = inputs[1].to(modelVars["device"])
                else:
                    inputs = inputs.to(modelVars["device"])
                    # channels_last
                    if isinstance(inputs, torch.Tensor) and inputs.ndim == 4 and modelVars["device"].type in ["cuda", "mps"]:
                        try:
                            inputs = inputs.contiguous(memory_format=torch.channels_last)
                        except Exception:
                            pass

                labels = labels.to(modelVars["device"])

                modelVars["optimizer"].zero_grad()

                with torch.set_grad_enabled(True):
                    if mdlParams.get("aux_classifier", False):
                        outputs, outputs_aux = modelVars["model"](inputs)
                        loss1 = modelVars["criterion"](outputs, labels)
                        labels_aux = labels.repeat(mdlParams["multiCropTrain"])
                        loss2 = modelVars["criterion"](outputs_aux, labels_aux)
                        loss = loss1 + mdlParams["aux_classifier_loss_fac"] * loss2
                    else:
                        outputs = modelVars["model"](inputs)
                        loss = modelVars["criterion"](outputs, labels)

                    if mdlParams["balance_classes"] in [6, 7, 8]:
                        indices_np = indices.cpu().numpy()
                        weights = torch.tensor(
                            mdlParams["loss_fac_per_example"][indices_np].astype(np.float32),
                            dtype=torch.float32,
                            device=modelVars["device"],
                        )
                        loss = loss * weights
                        loss = loss.mean()

                    loss.backward()
                    modelVars["optimizer"].step()

                loss_item = float(loss.detach().item())
                running_loss += loss_item

                bt = time.time() - t0
                batch_times.append(bt)

                # Batch-level progress + ETA
                if (j + 1) % batch_log_every == 0 or (j + 1) == numBatchesTrain:
                    avg_bt = float(np.mean(batch_times[-batch_log_every:])) if len(batch_times) >= 1 else bt
                    remain_batches = numBatchesTrain - (j + 1)
                    eta_epoch = remain_batches * avg_bt

                    # Total ETA (estimate using average time of completed epochs)
                    elapsed_total = time.time() - train_start_time
                    done_epochs = (step - start_epoch) + (j + 1) / max(1, numBatchesTrain)
                    avg_epoch_time = elapsed_total / max(1e-6, done_epochs)
                    remain_epochs = (mdlParams["training_steps"] - step) + remain_batches / max(1, numBatchesTrain)
                    eta_total = remain_epochs * avg_epoch_time

                    lr_now = modelVars["optimizer"].param_groups[0]["lr"]
                    print(
                        f"[Fold {cv}] Epoch {step}/{mdlParams['training_steps']} "
                        f"Batch {j+1}/{numBatchesTrain} "
                        f"loss={loss_item:.4f} lr={lr_now:.2e} "
                        f"ETA(epoch)={_fmt_seconds(eta_epoch)} ETA(total)={_fmt_seconds(eta_total)}",
                        flush=True,
                    )

            # Also print a summary at the end of each epoch
            epoch_time = time.time() - epoch_start
            avg_train_loss = running_loss / max(1, numBatchesTrain)
            print(
                f"[Fold {cv}] Epoch {step} done. avg_train_loss={avg_train_loss:.4f} time={_fmt_seconds(epoch_time)}",
                flush=True,
            )

            # -------------------------
            # display_step: evaluate + save
            # -------------------------
            if step % mdlParams["display_step"] == 0 or step == 1:
                if mdlParams["classification"]:
                    modelVars["model"].eval()
                    loss_eval, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = (
                        utils.getErrClassification_mgpu(mdlParams, eval_set, modelVars)
                    )

                    save_dict["loss"].append(loss_eval)
                    save_dict["acc"].append(accuracy)
                    save_dict["wacc"].append(waccuracy)
                    save_dict["auc"].append(auc)
                    save_dict["sens"].append(sensitivity)
                    save_dict["spec"].append(specificity)
                    save_dict["f1"].append(f1)
                    save_dict["step_num"].append(step)

                    prog_path = mdlParams["saveDir"] + f"/progression_{eval_set}.mat"
                    if os.path.isfile(prog_path):
                        os.remove(prog_path)
                    io.savemat(prog_path, save_dict)

                eval_metric = -np.mean(waccuracy)

                # best checkpoint
                if eval_metric < mdlParams["valBest"]:
                    mdlParams["valBest"] = eval_metric
                    if mdlParams["classification"]:
                        allData["f1Best"][cv] = f1
                        allData["sensBest"][cv] = sensitivity
                        allData["specBest"][cv] = specificity
                        allData["accBest"][cv] = accuracy
                        allData["waccBest"][cv] = waccuracy
                        allData["aucBest"][cv] = auc

                    oldBestInd = mdlParams["lastBestInd"]
                    mdlParams["lastBestInd"] = step
                    allData["convergeTime"][cv] = step
                    allData["bestPred"][cv] = predictions
                    allData["targets"][cv] = targets

                    with open(mdlParams["saveDirBase"] + "/CV.pkl", "wb") as f:
                        pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)

                    old_best_path = mdlParams["saveDir"] + f"/checkpoint_best-{oldBestInd}.pt"
                    if os.path.isfile(old_best_path):
                        os.remove(old_best_path)

                    state = {
                        "epoch": step,
                        "valBest": mdlParams["valBest"],
                        "lastBestInd": mdlParams["lastBestInd"],
                        "state_dict": modelVars["model"].state_dict(),
                        "optimizer": modelVars["optimizer"].state_dict(),
                    }
                    torch.save(state, mdlParams["saveDir"] + f"/checkpoint_best-{step}.pt")

                # current checkpoint
                state = {
                    "epoch": step,
                    "valBest": mdlParams["valBest"],
                    "lastBestInd": mdlParams["lastBestInd"],
                    "state_dict": modelVars["model"].state_dict(),
                    "optimizer": modelVars["optimizer"].state_dict(),
                }
                torch.save(state, mdlParams["saveDir"] + f"/checkpoint-{step}.pt")

                # delete older checkpoint
                if step == mdlParams["display_step"]:
                    lastInd = 1
                else:
                    lastInd = step - mdlParams["display_step"]

                if lastInd >= 1:
                    old_ckpt = mdlParams["saveDir"] + f"/checkpoint-{lastInd}.pt"
                    if os.path.isfile(old_ckpt):
                        os.remove(old_ckpt)

                duration = time.time() - train_start_time
                if mdlParams["classification"]:
                    print("\n")
                    print("Config:", sys.argv[2])
                    print(
                        f"Fold: {cv} Epoch: {step}/{mdlParams['training_steps']} "
                        f"elapsed={_fmt_seconds(duration)} "
                        + time.strftime("%d.%m.-%H:%M:%S", time.localtime())
                    )
                    print(
                        f"Loss on {eval_set}: {loss_eval}  Acc: {accuracy}  F1: {f1}  "
                        f"(best WACC: {-mdlParams['valBest']} at Epoch {mdlParams['lastBestInd']})"
                    )
                    print("AUC", auc, "Mean AUC", np.mean(auc))
                    print("Per Class Acc", waccuracy, "Weighted Accuracy", np.mean(waccuracy))
                    print("Sensitivity:", sensitivity, "Specificity", specificity)
                    print("Confusion Matrix")
                    print(conf_matrix)

                    if mdlParams["peak_at_testerr"]:
                        loss_t, acc_t, sens_t, spec_t, _, f1_t, _, _, _, _, _ = utils.getErrClassification_mgpu(
                            mdlParams, "testInd", modelVars
                        )
                        print("Test loss:", loss_t, "Acc:", acc_t, "F1:", f1_t)
                        print("Sensitivity:", sens_t, "Specificity:", spec_t)

                    if mdlParams["print_trainerr"] and "train" not in eval_set:
                        loss_tr, acc_tr, sens_tr, spec_tr, conf_tr, f1_tr, auc_tr, wacc_tr, pred_tr, targ_tr, _ = (
                            utils.getErrClassification_mgpu(mdlParams, "trainInd", modelVars)
                        )
                        save_dict_train["loss"].append(loss_tr)
                        save_dict_train["acc"].append(acc_tr)
                        save_dict_train["wacc"].append(wacc_tr)
                        save_dict_train["auc"].append(auc_tr)
                        save_dict_train["sens"].append(sens_tr)
                        save_dict_train["spec"].append(spec_tr)
                        save_dict_train["f1"].append(f1_tr)
                        save_dict_train["step_num"].append(step)

                        train_prog = mdlParams["saveDir"] + "/progression_trainInd.mat"
                        if os.path.isfile(train_prog):
                            os.remove(train_prog)
                        io.savemat(train_prog, save_dict_train)

                        print("Train loss:", loss_tr, "Acc:", acc_tr, "F1:", f1_tr)
                        print("Sensitivity:", sens_tr, "Specificity:", spec_tr)

        modelVars.clear()
        gc.collect()

        print("Best F1:", allData["f1Best"].get(cv, None))
        print("Best Sens:", allData["sensBest"].get(cv, None))
        print("Best Spec:", allData["specBest"].get(cv, None))
        print("Best Acc:", allData["accBest"].get(cv, None))
        print("Best Per Class Accuracy:", allData["waccBest"].get(cv, None))
        if cv in allData["waccBest"]:
            print("Best Weighted Acc:", np.mean(allData["waccBest"][cv]))
        print("Best AUC:", allData["aucBest"].get(cv, None))
        if cv in allData["aucBest"]:
            print("Best Mean AUC:", np.mean(allData["aucBest"][cv]))
        print("Convergence Steps:", allData["convergeTime"].get(cv, None))


if __name__ == "__main__":
    mp.freeze_support()
    main()
