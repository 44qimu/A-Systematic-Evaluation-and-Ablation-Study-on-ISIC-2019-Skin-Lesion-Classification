import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import models
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import gc
import importlib
import time
import csv
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import imagesize

# -----------------------------
# Helpers
# -----------------------------
def get_best_device():
    """Return best available device: CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patch_isic_csv_reader_skip_header():
    if getattr(csv, "_isic_reader_patched", False):
        return

    _orig_reader = csv.reader

    def _norm(x):
        return str(x).strip().lower().lstrip("\ufeff")

    def _is_header_row(row):
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            return False

        first0 = _norm(row[0])
        if first0 in ("image", "image_id", "isic_id"):
            return True

        header_tokens = {
            "mel", "nv", "bcc", "akiec", "bkl", "df", "vasc", "scc", "unk"
        }
        for c in row:
            if _norm(c) in header_tokens:
                return True
        return False

    def _patched_reader(f, *args, **kwargs):
        it = _orig_reader(f, *args, **kwargs)
        try:
            first = next(it)
        except StopIteration:
            return iter(())

        if _is_header_row(first):
            return it  # skip header

        def gen():
            yield first
            for row in it:
                yield row
        return gen()

    csv._orig_reader = _orig_reader
    csv.reader = _patched_reader
    csv._isic_reader_patched = True


def parse_ckpt_and_gpu_arg(arg_str: str):
    """
    Parse ckpt_name and GPU index from sys.argv[6] (or default).
    Convention in original code:
      - contains 'last'  -> ckpt_name = 'checkpoint-'
      - otherwise       -> ckpt_name = 'checkpoint_best-'
      - contains 'first' -> use_first = True (choose an earlier checkpoint)
      - GPU index extracted from digits in arg_str, e.g. 'bestgpu0' -> 0
    """
    if arg_str is None:
        arg_str = "last"

    if "last" in arg_str:
        ckpt_name = "checkpoint-"
    else:
        ckpt_name = "checkpoint_best-"

    use_first = True if "first" in arg_str else False

    digits = re.findall(r"\d+", arg_str)
    gpu_ids = []
    if len(digits) > 0:
        gpu_ids = [int(digits[-1])]
    return ckpt_name, use_first, gpu_ids


def safe_num_workers():
    """
    Mac + (MPS/CPU) often hangs with multi-worker DataLoader.
    Keep your original 8 workers on CUDA; otherwise default to 0 on macOS.
    """
    if sys.platform == "darwin" and (not torch.cuda.is_available()):
        return 0
    return 8


def find_latest_checkpoint(save_dir: str, ckpt_name: str, use_first: bool):
    """
    Scan save_dir for checkpoint files and pick the latest (or earlier if use_first).
    Original behavior: if use_first -> pick second-latest ([-2]) else latest.
    Here: robust to small count.
    """
    files = glob(os.path.join(save_dir, "*"))
    steps = []

    for f in files:
        base = os.path.basename(f)
        if "checkpoint" not in base:
            continue
        if ckpt_name not in base:
            continue
        nums = [int(s) for s in re.findall(r"\d+", base)]
        if len(nums) == 0:
            continue
        steps.append(nums[-1])

    if len(steps) == 0:
        raise FileNotFoundError(
            f"[EVAL] No checkpoint found in: {save_dir}\n"
            f"Expected filenames containing '{ckpt_name}'."
        )

    steps = sorted(list(set(steps)))
    if use_first and len(steps) >= 2:
        chosen = steps[-2]  # earlier one (2nd latest)
    else:
        chosen = steps[-1]  # latest

    chk_path = os.path.join(save_dir, f"{ckpt_name}{int(chosen)}.pt")
    return chk_path, int(chosen)


def to_device_weight(class_weights, device):
    return torch.tensor(class_weights.astype(np.float32), dtype=torch.float32, device=device)


def _extract_state_dict(state_obj):
    """Support checkpoints that are either {'state_dict':...} or raw state_dict."""
    if isinstance(state_obj, dict) and ("state_dict" in state_obj):
        return state_obj["state_dict"]
    return state_obj


def align_meta_array_to_ckpt(mdlParams, state_dict):
    """
    Make mdlParams['meta_array'] dim match checkpoint meta dim.
    - If current meta_dim > ckpt_meta_dim -> crop.
    - If current meta_dim < ckpt_meta_dim -> zero-pad (rare, but makes it run).
    Returns ckpt_meta_dim if meta exists in ckpt; otherwise None.
    """
    if mdlParams.get("meta_features", None) is None:
        return None
    if not isinstance(state_dict, dict):
        return None
    if "meta_before.0.weight" not in state_dict:
        return None
    if "meta_array" not in mdlParams:
        return None
    if not isinstance(mdlParams["meta_array"], np.ndarray):
        return None

    ckpt_meta_dim = int(state_dict["meta_before.0.weight"].shape[1])
    cur_meta_dim = int(mdlParams["meta_array"].shape[1])

    if cur_meta_dim != ckpt_meta_dim:
        print(f"[EVAL] Meta dim mismatch: current={cur_meta_dim} ckpt={ckpt_meta_dim}. Aligning meta_array...")
        if cur_meta_dim > ckpt_meta_dim:
            mdlParams["meta_array"] = mdlParams["meta_array"][:, :ckpt_meta_dim]
            print(f"[EVAL] Cropped meta_array to shape {mdlParams['meta_array'].shape}")
        else:
            pad = ckpt_meta_dim - cur_meta_dim
            mdlParams["meta_array"] = np.pad(mdlParams["meta_array"], ((0, 0), (0, pad)), mode="constant")
            print(f"[EVAL] Padded meta_array to shape {mdlParams['meta_array'].shape}")

    # Store for downstream usage/debug
    mdlParams["ckpt_meta_dim"] = ckpt_meta_dim
    return ckpt_meta_dim


def enforce_model_meta_input_dim(model, ckpt_meta_dim):
    """
    Some modify_meta() implementations may still create meta_before with wrong in_features.
    This function force-replaces meta_before[0] Linear to match ckpt_meta_dim (keeping out_features).
    """
    if ckpt_meta_dim is None:
        return

    if not hasattr(model, "meta_before"):
        return

    mb = getattr(model, "meta_before")
    if isinstance(mb, nn.Sequential) and len(mb) > 0 and isinstance(mb[0], nn.Linear):
        lin0 = mb[0]
        if int(lin0.in_features) != int(ckpt_meta_dim):
            print(f"[EVAL] Fixing model.meta_before[0] in_features: {lin0.in_features} -> {ckpt_meta_dim}")
            new_lin = nn.Linear(int(ckpt_meta_dim), int(lin0.out_features), bias=(lin0.bias is not None))
            mb[0] = new_lin
    # If it's not Sequential/Linear, do nothing (unknown implementation).


# -----------------------------
# Main
# -----------------------------

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Basic argument safety
# Expected args (as used in your original script):
#   sys.argv[1] : pc_cfg module name under pc_cfgs/
#   sys.argv[2] : model cfg module name under cfgs/
#   sys.argv[3] : multi-crop spec or single
#   sys.argv[4] : voting scheme
#   sys.argv[5] : saveDirBase override or 'NONE'
#   sys.argv[6] : checkpoint spec + (optional) gpu index, e.g. 'lastgpu0' / 'bestgpu0' / 'firstbestgpu0'
#   sys.argv[7] : optional 'HAMONLY'
#   sys.argv[8] : optional external set path

if len(sys.argv) < 6:
    raise ValueError(
        "Usage (minimum): python eval.py <pc_cfg> <cfg> <multiSpec|single> <votingScheme> <saveDirBase|NONE> [ckptSpec] [HAMONLY] [ext_path]"
    )

pc_cfg_name = sys.argv[1]
cfg_name = sys.argv[2]
multi_spec = sys.argv[3]
voting_scheme = sys.argv[4]
save_dir_arg = sys.argv[5]
ckpt_arg = sys.argv[6] if len(sys.argv) > 6 else "last"  # default if not provided

# Import machine config
pc_cfg = importlib.import_module("pc_cfgs." + pc_cfg_name)
mdlParams.update(pc_cfg.mdlParams)

# Parse checkpoint arg + gpu
mdlParams["ckpt_name"], use_first, gpu_ids = parse_ckpt_and_gpu_arg(ckpt_arg)
if use_first:
    mdlParams["use_first"] = True

# Set visible devices ONLY if CUDA available + you provided a GPU index
if torch.cuda.is_available() and len(gpu_ids) > 0:
    mdlParams["numGPUs"] = gpu_ids
    cuda_str = ",".join([str(x) for x in mdlParams["numGPUs"]])
    print("Devices to use:", cuda_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str
else:
    # On MPS/CPU we still want "one device" semantics for code paths that divide by len(numGPUs)
    mdlParams["numGPUs"] = [0]
    print("CUDA_VISIBLE_DEVICES not set (CUDA unavailable or no GPU index provided). Using single-device eval.")

# Optional HAMONLY
if len(sys.argv) > 7:
    if "HAMONLY" in sys.argv[7]:
        mdlParams["eval_on_ham_only"] = True

# Patch csv.reader to skip ISIC GroundTruth header rows (handles BOM) before cfg.init()
patch_isic_csv_reader_skip_header()

# Import model config
model_cfg = importlib.import_module("cfgs." + cfg_name)
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# Safety: if config overwrote numGPUs to empty (e.g., on MPS/CPU), force single-device semantics
if ("numGPUs" not in mdlParams) or (isinstance(mdlParams["numGPUs"], (list, tuple)) and len(mdlParams["numGPUs"]) == 0):
    mdlParams["numGPUs"] = [0]

# Path name where model is saved is the fifth argument
if "NONE" in save_dir_arg:
    mdlParams["saveDirBase"] = mdlParams["saveDir"] + cfg_name
else:
    mdlParams["saveDirBase"] = save_dir_arg

# Third is multi crop yes no
if "multi" in multi_spec:
    if "rand" in multi_spec:
        mdlParams["numRandValSeq"] = [int(s) for s in re.findall(r"\d+", multi_spec)][0]
        print("Random sequence number", mdlParams["numRandValSeq"])
    else:
        mdlParams["numRandValSeq"] = 0

    mdlParams["multiCropEval"] = [int(s) for s in re.findall(r"\d+", multi_spec)][-1]
    mdlParams["voting_scheme"] = voting_scheme

    if "scale" in multi_spec:
        print(
            "Multi Crop and Scale Eval with crop number:",
            mdlParams["multiCropEval"],
            " Voting scheme: ",
            mdlParams["voting_scheme"],
        )
        mdlParams["orderedCrop"] = False
        mdlParams["scale_min"] = [int(s) for s in re.findall(r"\d+", multi_spec)][-2] / 100.0

    elif "determ" in multi_spec:
        # Example application: multideterm5sc3f2
        mdlParams["deterministic_eval"] = True
        mdlParams["numCropPositions"] = [int(s) for s in re.findall(r"\d+", multi_spec)][-3]
        num_scales = [int(s) for s in re.findall(r"\d+", multi_spec)][-2]
        all_scales = [1.0, 0.5, 0.75, 0.25, 0.9, 0.6, 0.4]
        mdlParams["cropScales"] = all_scales[:num_scales]
        mdlParams["cropFlipping"] = [int(s) for s in re.findall(r"\d+", multi_spec)][-1]
        print(
            "deterministic eval with crops number",
            mdlParams["numCropPositions"],
            "scales",
            mdlParams["cropScales"],
            "flipping",
            mdlParams["cropFlipping"],
        )
        mdlParams["multiCropEval"] = mdlParams["numCropPositions"] * len(mdlParams["cropScales"]) * mdlParams["cropFlipping"]
        mdlParams["offset_crop"] = 0.2

    elif "order" in multi_spec:
        mdlParams["orderedCrop"] = True
        if mdlParams.get("var_im_size", False):
            # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
            mdlParams["cropPositions"] = np.zeros([len(mdlParams["im_paths"]), mdlParams["multiCropEval"], 2], dtype=np.int64)

            for u in range(len(mdlParams["im_paths"])):
                height, width = imagesize.get(mdlParams["im_paths"][u])
                if width < mdlParams["input_size"][0]:
                    height = int(mdlParams["input_size"][0] / float(width)) * height
                    width = mdlParams["input_size"][0]
                if height < mdlParams["input_size"][0]:
                    width = int(mdlParams["input_size"][0] / float(height)) * width
                    height = mdlParams["input_size"][0]
                if mdlParams.get("resize_large_ones") is not None:
                    if width == mdlParams["large_size"] and height == mdlParams["large_size"]:
                        width, height = (mdlParams["resize_large_ones"], mdlParams["resize_large_ones"])

                ind = 0
                for i in range(np.int32(np.sqrt(mdlParams["multiCropEval"]))):
                    for j in range(np.int32(np.sqrt(mdlParams["multiCropEval"]))):
                        mdlParams["cropPositions"][u, ind, 0] = (
                            mdlParams["input_size"][0] / 2
                            + i * ((width - mdlParams["input_size"][1]) / (np.sqrt(mdlParams["multiCropEval"]) - 1))
                        )
                        mdlParams["cropPositions"][u, ind, 1] = (
                            mdlParams["input_size"][1] / 2
                            + j * ((height - mdlParams["input_size"][0]) / (np.sqrt(mdlParams["multiCropEval"]) - 1))
                        )
                        ind += 1

            # Sanity checks (kept from original logic)
            height = mdlParams["input_size"][0]
            width = mdlParams["input_size"][1]
            for u in range(len(mdlParams["im_paths"])):
                height_test, width_test = imagesize.get(mdlParams["im_paths"][u])
                if width_test < mdlParams["input_size"][0]:
                    height_test = int(mdlParams["input_size"][0] / float(width_test)) * height_test
                    width_test = mdlParams["input_size"][0]
                if height_test < mdlParams["input_size"][0]:
                    width_test = int(mdlParams["input_size"][0] / float(height_test)) * width_test
                    height_test = mdlParams["input_size"][0]
                if mdlParams.get("resize_large_ones") is not None:
                    if width_test == mdlParams["large_size"] and height_test == mdlParams["large_size"]:
                        width_test, height_test = (mdlParams["resize_large_ones"], mdlParams["resize_large_ones"])
                test_im = np.zeros([width_test, height_test])
                for i in range(mdlParams["multiCropEval"]):
                    im_crop = test_im[
                        np.int32(mdlParams["cropPositions"][u, i, 0] - height / 2) : np.int32(mdlParams["cropPositions"][u, i, 0] - height / 2) + height,
                        np.int32(mdlParams["cropPositions"][u, i, 1] - width / 2) : np.int32(mdlParams["cropPositions"][u, i, 1] - width / 2) + width,
                    ]
                    if im_crop.shape[0] != mdlParams["input_size"][0]:
                        print("Wrong shape", im_crop.shape[0], mdlParams["im_paths"][u])
                    if im_crop.shape[1] != mdlParams["input_size"][1]:
                        print("Wrong shape", im_crop.shape[1], mdlParams["im_paths"][u])

        else:
            # Fixed-size case
            mdlParams["cropPositions"] = np.zeros([mdlParams["multiCropEval"], 2], dtype=np.int64)
            if mdlParams["multiCropEval"] == 5:
                numCrops = 4
            elif mdlParams["multiCropEval"] == 7:
                numCrops = 9
                mdlParams["cropPositions"] = np.zeros([9, 2], dtype=np.int64)
            else:
                numCrops = mdlParams["multiCropEval"]

            ind = 0
            for i in range(np.int32(np.sqrt(numCrops))):
                for j in range(np.int32(np.sqrt(numCrops))):
                    mdlParams["cropPositions"][ind, 0] = mdlParams["input_size"][0] / 2 + i * (
                        (mdlParams["input_size_load"][0] - mdlParams["input_size"][0]) / (np.sqrt(numCrops) - 1)
                    )
                    mdlParams["cropPositions"][ind, 1] = mdlParams["input_size"][1] / 2 + j * (
                        (mdlParams["input_size_load"][1] - mdlParams["input_size"][1]) / (np.sqrt(numCrops) - 1)
                    )
                    ind += 1

            # Add center crop
            if mdlParams["multiCropEval"] == 5:
                mdlParams["cropPositions"][4, 0] = mdlParams["input_size_load"][0] / 2
                mdlParams["cropPositions"][4, 1] = mdlParams["input_size_load"][1] / 2
            if mdlParams["multiCropEval"] == 7:
                mdlParams["cropPositions"] = np.delete(mdlParams["cropPositions"], [3, 7], 0)

            print("Positions val", mdlParams["cropPositions"])

            test_im = np.zeros(mdlParams["input_size_load"])
            height = mdlParams["input_size"][0]
            width = mdlParams["input_size"][1]
            for i in range(mdlParams["multiCropEval"]):
                im_crop = test_im[
                    np.int32(mdlParams["cropPositions"][i, 0] - height / 2) : np.int32(mdlParams["cropPositions"][i, 0] - height / 2) + height,
                    np.int32(mdlParams["cropPositions"][i, 1] - width / 2) : np.int32(mdlParams["cropPositions"][i, 1] - width / 2) + width,
                    :,
                ]
                print("Shape", i + 1, im_crop.shape)

        print(
            "Multi Crop with order with crop number:",
            mdlParams["multiCropEval"],
            " Voting scheme: ",
            mdlParams["voting_scheme"],
        )

        if "flip" in multi_spec:
            # additional flipping, example: flip2multiorder16
            mdlParams["eval_flipping"] = [int(s) for s in re.findall(r"\d+", multi_spec)][-2]
            print("Additional flipping", mdlParams["eval_flipping"])

    else:
        print("Multi Crop Eval with crop number:", mdlParams["multiCropEval"], " Voting scheme: ", mdlParams["voting_scheme"])
        mdlParams["orderedCrop"] = False

else:
    mdlParams["multiCropEval"] = 0
    mdlParams["orderedCrop"] = False

# Set training set to eval mode
mdlParams["trainSetState"] = "eval"

if mdlParams["numClasses"] == 9 and mdlParams.get("no_c9_eval", False):
    num_classes = mdlParams["numClasses"] - 1
else:
    num_classes = mdlParams["numClasses"]

# Save results in here
allData = {}
allData["f1Best"] = np.zeros([mdlParams["numCV"]])
allData["sensBest"] = np.zeros([mdlParams["numCV"], num_classes])
allData["specBest"] = np.zeros([mdlParams["numCV"], num_classes])
allData["accBest"] = np.zeros([mdlParams["numCV"]])
allData["waccBest"] = np.zeros([mdlParams["numCV"], num_classes])
allData["aucBest"] = np.zeros([mdlParams["numCV"], num_classes])
allData["convergeTime"] = {}
allData["bestPred"] = {}
allData["bestPredMC"] = {}
allData["targets"] = {}
allData["extPred"] = {}
allData["f1Best_meta"] = np.zeros([mdlParams["numCV"]])
allData["sensBest_meta"] = np.zeros([mdlParams["numCV"], num_classes])
allData["specBest_meta"] = np.zeros([mdlParams["numCV"], num_classes])
allData["accBest_meta"] = np.zeros([mdlParams["numCV"]])
allData["waccBest_meta"] = np.zeros([mdlParams["numCV"], num_classes])
allData["aucBest_meta"] = np.zeros([mdlParams["numCV"], num_classes])
allData["bestPred_meta"] = {}
allData["targets_meta"] = {}

NUM_WORKERS = safe_num_workers()

# -----------------------------
# Normal eval (no external path)
# -----------------------------
pklFileName = None  # will be set later

if not (len(sys.argv) > 8):
    for cv in range(mdlParams["numCV"]):
        # Reset model graph
        importlib.reload(models)

        # Collect model variables
        modelVars = {}
        modelVars["device"] = get_best_device()
        print("Using device:", modelVars["device"])

        # Def current CV set
        mdlParams["trainInd"] = mdlParams["trainIndCV"][cv]
        if "valIndCV" in mdlParams:
            mdlParams["valInd"] = mdlParams["valIndCV"][cv]

        # Def current path for saving stuff
        if "valIndCV" in mdlParams:
            mdlParams["saveDir"] = mdlParams["saveDirBase"] + "/CVSet" + str(cv)
        else:
            mdlParams["saveDir"] = mdlParams["saveDirBase"]

        # Potentially calculate setMean to subtract
        if mdlParams["subtract_set_mean"] == 1:
            mdlParams["setMean"] = np.mean(mdlParams["images_means"][mdlParams["trainInd"], :], (0))
            print("Set Mean", mdlParams["setMean"])

        # Potentially only HAM eval
        if mdlParams.get("eval_on_ham_only", False):
            print("Old val inds", len(mdlParams["valInd"]))
            mdlParams["valInd"] = np.intersect1d(mdlParams["valInd"], mdlParams["HAM10000_inds"])
            print("New val inds, HAM only", len(mdlParams["valInd"]))

        # ---------------------------------------------------------
        # IMPORTANT: Load checkpoint EARLY to align meta dim correctly
        # ---------------------------------------------------------
        chkPath, chosen_step = find_latest_checkpoint(
            mdlParams["saveDir"], mdlParams["ckpt_name"], mdlParams.get("use_first", False)
        )
        print("Restoring: ", chkPath)

        ckpt_state = torch.load(chkPath, map_location="cpu", weights_only=False)
        ckpt_state_dict = _extract_state_dict(ckpt_state)

        ckpt_meta_dim = align_meta_array_to_ckpt(mdlParams, ckpt_state_dict)

        # balance classes
        if mdlParams["balance_classes"] < 3 or mdlParams["balance_classes"] == 7 or mdlParams["balance_classes"] == 11:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(np.argmax(mdlParams["labels_array"][mdlParams["trainInd"], :], 1)),
                y=np.argmax(mdlParams["labels_array"][mdlParams["trainInd"], :], 1),
            )
            print("Current class weights", class_weights)
            class_weights = class_weights * mdlParams["extra_fac"]
            print("Current class weights with extra", class_weights)

        elif mdlParams["balance_classes"] == 3 or mdlParams["balance_classes"] == 4:
            not_one_hot = np.argmax(mdlParams["labels_array"], 1)
            mdlParams["class_indices"] = []
            for i in range(mdlParams["numClasses"]):
                mdlParams["class_indices"].append(np.where(not_one_hot == i)[0])
                mdlParams["class_indices"][i] = np.setdiff1d(mdlParams["class_indices"][i], mdlParams["valInd"])

        elif mdlParams["balance_classes"] == 5 or mdlParams["balance_classes"] == 6 or mdlParams["balance_classes"] == 13:
            class_weights = 1.0 / np.mean(mdlParams["labels_array"][mdlParams["trainInd"], :], axis=0)
            print("Current class weights", class_weights)
            class_weights = class_weights * mdlParams["extra_fac"]
            print("Current class weights with extra", class_weights)

        elif mdlParams["balance_classes"] == 9:
            print("Balance 9")
            indices_ham = mdlParams["trainInd"][mdlParams["trainInd"] < 25331]
            if mdlParams["numClasses"] == 9:
                class_weights_ = 1.0 / np.mean(mdlParams["labels_array"][indices_ham, :8], axis=0)
                class_weights = np.zeros([mdlParams["numClasses"]])
                class_weights[:8] = class_weights_
                class_weights[-1] = np.max(class_weights_)
            else:
                class_weights = 1.0 / np.mean(mdlParams["labels_array"][indices_ham, :], axis=0)
            print("Current class weights", class_weights)
            if isinstance(mdlParams["extra_fac"], float):
                class_weights = np.power(class_weights, mdlParams["extra_fac"])
            else:
                class_weights = class_weights * mdlParams["extra_fac"]
            print("Current class weights with extra", class_weights)

        # Meta scaler (after meta alignment)
        if mdlParams.get("meta_features", None) is not None and mdlParams["scale_features"]:
            mdlParams["feature_scaler_meta"] = sklearn.preprocessing.StandardScaler().fit(
                mdlParams["meta_array"][mdlParams["trainInd"], :]
            )

        # Datasets / loaders
        dataset_train = utils.ISICDataset(mdlParams, "trainInd")
        dataset_val = utils.ISICDataset(mdlParams, "valInd")

        if mdlParams["multiCropEval"] > 0:
            modelVars["dataloader_valInd"] = DataLoader(
                dataset_val,
                batch_size=mdlParams["multiCropEval"],
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            modelVars["dataloader_valInd"] = DataLoader(
                dataset_val,
                batch_size=mdlParams["batchSize"],
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )

        modelVars["dataloader_trainInd"] = DataLoader(
            dataset_train,
            batch_size=mdlParams["batchSize"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        # For test
        if "testInd" in mdlParams:
            dataset_test = utils.ISICDataset(mdlParams, "testInd")
            if mdlParams["multiCropEval"] > 0:
                modelVars["dataloader_testInd"] = DataLoader(
                    dataset_test,
                    batch_size=mdlParams["multiCropEval"],
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                modelVars["dataloader_testInd"] = DataLoader(
                    dataset_test,
                    batch_size=mdlParams["batchSize"],
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=torch.cuda.is_available(),
                )

        # Build model
        modelVars["model"] = models.getModel(mdlParams)()

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

        # modify model (meta)
        if mdlParams.get("meta_features", None) is not None:
            modelVars["model"] = models.modify_meta(mdlParams, modelVars["model"])
            # Force meta input dim to ckpt meta dim (robust)
            enforce_model_meta_input_dim(modelVars["model"], ckpt_meta_dim)

        modelVars["model"] = modelVars["model"].to(modelVars["device"])

        # Loss
        if mdlParams["balance_classes"] == 3 or mdlParams["balance_classes"] == 0 or mdlParams["balance_classes"] == 12:
            modelVars["criterion"] = nn.CrossEntropyLoss()

        elif mdlParams["balance_classes"] == 8:
            modelVars["criterion"] = nn.CrossEntropyLoss(reduction="none")

        elif mdlParams["balance_classes"] == 6 or mdlParams["balance_classes"] == 7:
            w = to_device_weight(class_weights, modelVars["device"])
            modelVars["criterion"] = nn.CrossEntropyLoss(weight=w, reduction="none")

        elif mdlParams["balance_classes"] == 10:
            modelVars["criterion"] = utils.FocalLoss(mdlParams["numClasses"])

        elif mdlParams["balance_classes"] == 11:
            w = to_device_weight(class_weights, modelVars["device"])
            modelVars["criterion"] = utils.FocalLoss(mdlParams["numClasses"], alpha=w)

        else:
            w = to_device_weight(class_weights, modelVars["device"])
            modelVars["criterion"] = nn.CrossEntropyLoss(weight=w)

        # Optimizer / scheduler (kept for compatibility)
        modelVars["optimizer"] = optim.Adam(modelVars["model"].parameters(), lr=mdlParams["learning_rate"])
        modelVars["scheduler"] = lr_scheduler.StepLR(
            modelVars["optimizer"], step_size=mdlParams["lowerLRAfter"], gamma=1 / np.float32(mdlParams["LRstep"])
        )
        modelVars["softmax"] = nn.Softmax(dim=1)

        # Load ckpt weights (already loaded on CPU)
        modelVars["model"].load_state_dict(ckpt_state_dict)

        # Construct pkl filename
        pklFileName = cfg_name + "_" + ckpt_arg + "_" + str(int(chosen_step)) + ".pkl"

        modelVars["model"].eval()

        if mdlParams["classification"]:
            print("CV Set ", cv + 1)
            print("------------------------------------")

            with torch.no_grad():
                if "valInd" in mdlParams and (len(sys.argv) <= 8):
                    loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(
                        mdlParams, "valInd", modelVars
                    )

                # Print + save
                print("Validation Results:")
                print("----------------------------------")
                print("Loss", np.mean(loss))
                print("F1 Score", f1)
                print("Sensitivity", sensitivity)
                print("Specificity", specificity)
                print("Accuracy", accuracy)
                print("Per Class Accuracy", waccuracy)
                print("Weighted Accuracy", np.mean(waccuracy))
                print("AUC", auc)
                print("Mean AUC", np.mean(auc))

                if "testInd" not in mdlParams:
                    allData["f1Best"][cv] = f1
                    allData["sensBest"][cv, :] = sensitivity
                    allData["specBest"][cv, :] = specificity
                    allData["accBest"][cv] = accuracy
                    allData["waccBest"][cv, :] = waccuracy
                    allData["aucBest"][cv, :] = auc

                allData["bestPred"][cv] = predictions
                allData["bestPredMC"][cv] = predictions_mc
                allData["targets"][cv] = targets
                print("Pred shape", predictions.shape, "Tar shape", targets.shape)

                # If testInd exists, run test too (kept exactly as your original intention)
                if "testInd" in mdlParams:
                    loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(
                        mdlParams, "testInd", modelVars
                    )
                    print("Test Results Normal:")
                    print("----------------------------------")
                    print("Loss", np.mean(loss))
                    print("F1 Score", f1)
                    print("Sensitivity", sensitivity)
                    print("Specificity", specificity)
                    print("Accuracy", accuracy)
                    print("Per Class Accuracy", waccuracy)
                    print("Weighted Accuracy", np.mean(waccuracy))
                    print("AUC", auc)
                    print("Mean AUC", np.mean(auc))

                    allData["f1Best"][cv] = f1
                    allData["sensBest"][cv, :] = sensitivity
                    allData["specBest"][cv, :] = specificity
                    allData["accBest"][cv] = accuracy
                    allData["waccBest"][cv, :] = waccuracy
                    allData["aucBest"][cv, :] = auc
        else:
            print("Not Implemented (Regression)")

# -----------------------------
# External set eval (sys.argv[8])
# -----------------------------
if len(sys.argv) > 8:
    ext_path = sys.argv[8]

    for cv in range(mdlParams["numCV"]):
        importlib.reload(models)

        modelVars = {}
        modelVars["device"] = get_best_device()
        print("Using device:", modelVars["device"])

        print("Creating predictions for path ", ext_path)

        # Define saveDir for this CV first (needed for checkpoint)
        mdlParams["saveDir"] = mdlParams["saveDirBase"] + "/CVSet" + str(cv)

        # Load checkpoint early to get ckpt_meta_dim
        chkPath, chosen_step = find_latest_checkpoint(
            mdlParams["saveDir"], mdlParams["ckpt_name"], mdlParams.get("use_first", False)
        )
        print("Restoring: ", chkPath)

        ckpt_state = torch.load(chkPath, map_location="cpu", weights_only=False)
        ckpt_state_dict = _extract_state_dict(ckpt_state)

        ckpt_meta_dim = None
        if mdlParams.get("meta_features", None) is not None and isinstance(ckpt_state_dict, dict) and "meta_before.0.weight" in ckpt_state_dict:
            ckpt_meta_dim = int(ckpt_state_dict["meta_before.0.weight"].shape[1])
            mdlParams["ckpt_meta_dim"] = ckpt_meta_dim

        # Add meta data (kept as original behavior/path)
        if mdlParams.get("meta_features", None) is not None:
            mdlParams["meta_dict"] = {}
            path1 = mdlParams["dataDir"] + "/meta_data/test_rez3_ll/meta_data_test.pkl"
            with open(path1, "rb") as f:
                meta_data = pickle.load(f)

            for k in range(len(meta_data["im_name"])):
                feature_vector = []
                if "age_oh" in mdlParams["meta_features"]:
                    if mdlParams["encode_nan"]:
                        feature_vector.append(meta_data["age_oh"][k, :])
                    else:
                        feature_vector.append(meta_data["age_oh"][k, 1:])
                if "age_num" in mdlParams["meta_features"]:
                    feature_vector.append(np.array([meta_data["age_num"][k]]))
                if "loc_oh" in mdlParams["meta_features"]:
                    if mdlParams["encode_nan"]:
                        feature_vector.append(meta_data["loc_oh"][k, :])
                    else:
                        feature_vector.append(meta_data["loc_oh"][k, 1:])
                if "sex_oh" in mdlParams["meta_features"]:
                    if mdlParams["encode_nan"]:
                        feature_vector.append(meta_data["sex_oh"][k, :])
                    else:
                        feature_vector.append(meta_data["sex_oh"][k, 1:])

                feature_vector = np.concatenate(feature_vector, axis=0)

                # Align feature_vector dim to ckpt (crop/pad) so it ALWAYS matches
                if ckpt_meta_dim is not None:
                    if feature_vector.shape[0] > ckpt_meta_dim:
                        feature_vector = feature_vector[:ckpt_meta_dim]
                    elif feature_vector.shape[0] < ckpt_meta_dim:
                        feature_vector = np.pad(feature_vector, (0, ckpt_meta_dim - feature_vector.shape[0]), mode="constant")

                mdlParams["meta_dict"][meta_data["im_name"][k]] = feature_vector

        # Define the path
        files = sorted(glob(ext_path + "/*"))
        mdlParams["im_paths"] = []
        mdlParams["meta_list"] = []

        for j in range(len(files)):
            if "ISIC_" in files[j]:
                mdlParams["im_paths"].append(files[j])
                if mdlParams.get("meta_features", None) is not None:
                    for key in mdlParams["meta_dict"]:
                        if key in files[j]:
                            mdlParams["meta_list"].append(mdlParams["meta_dict"][key])

        if mdlParams.get("meta_features", None) is not None:
            mdlParams["meta_array"] = np.array(mdlParams["meta_list"])

            # Align meta_array dim to ckpt (crop/pad)
            if ckpt_meta_dim is not None and isinstance(mdlParams["meta_array"], np.ndarray):
                cur_meta_dim = int(mdlParams["meta_array"].shape[1])
                if cur_meta_dim != ckpt_meta_dim:
                    print(f"[EVAL-EXT] Meta dim mismatch: current={cur_meta_dim} ckpt={ckpt_meta_dim}. Aligning meta_array...")
                    if cur_meta_dim > ckpt_meta_dim:
                        mdlParams["meta_array"] = mdlParams["meta_array"][:, :ckpt_meta_dim]
                    else:
                        pad = ckpt_meta_dim - cur_meta_dim
                        mdlParams["meta_array"] = np.pad(mdlParams["meta_array"], ((0, 0), (0, pad)), mode="constant")
                    print(f"[EVAL-EXT] meta_array shape now {mdlParams['meta_array'].shape}")

        # Add empty labels
        mdlParams["labels_array"] = np.zeros([len(mdlParams["im_paths"]), mdlParams["numClasses"]], dtype=np.float32)

        # Define everything as a valind set
        mdlParams["valInd"] = np.array(np.arange(len(mdlParams["im_paths"])))
        mdlParams["trainInd"] = mdlParams["valInd"]

        # If var_im_size, recompute crop positions
        if mdlParams.get("var_im_size", False):
            mdlParams["cropPositions"] = np.zeros([len(mdlParams["im_paths"]), mdlParams["multiCropEval"], 2], dtype=np.int64)

            for u in range(len(mdlParams["im_paths"])):
                height, width = imagesize.get(mdlParams["im_paths"][u])
                if width < mdlParams["input_size"][0]:
                    height = int(mdlParams["input_size"][0] / float(width)) * height
                    width = mdlParams["input_size"][0]
                if height < mdlParams["input_size"][0]:
                    width = int(mdlParams["input_size"][0] / float(height)) * width
                    height = mdlParams["input_size"][0]
                if mdlParams.get("resize_large_ones") is not None:
                    if width == mdlParams["large_size"] and height == mdlParams["large_size"]:
                        width, height = (mdlParams["resize_large_ones"], mdlParams["resize_large_ones"])

                ind = 0
                for i in range(np.int32(np.sqrt(mdlParams["multiCropEval"]))):
                    for j in range(np.int32(np.sqrt(mdlParams["multiCropEval"]))):
                        mdlParams["cropPositions"][u, ind, 0] = (
                            mdlParams["input_size"][0] / 2
                            + i * ((width - mdlParams["input_size"][1]) / (np.sqrt(mdlParams["multiCropEval"]) - 1))
                        )
                        mdlParams["cropPositions"][u, ind, 1] = (
                            mdlParams["input_size"][1] / 2
                            + j * ((height - mdlParams["input_size"][0]) / (np.sqrt(mdlParams["multiCropEval"]) - 1))
                        )
                        ind += 1

            # Sanity checks
            test_im = np.zeros(mdlParams["input_size_load"])
            height = mdlParams["input_size"][0]
            width = mdlParams["input_size"][1]
            for u in range(len(mdlParams["im_paths"])):
                height_test, width_test = imagesize.get(mdlParams["im_paths"][u])
                if width_test < mdlParams["input_size"][0]:
                    height_test = int(mdlParams["input_size"][0] / float(width_test)) * height_test
                    width_test = mdlParams["input_size"][0]
                if height_test < mdlParams["input_size"][0]:
                    width_test = int(mdlParams["input_size"][0] / float(height_test)) * width_test
                    height_test = mdlParams["input_size"][0]
                if mdlParams.get("resize_large_ones") is not None:
                    if width_test == mdlParams["large_size"] and height_test == mdlParams["large_size"]:
                        width_test, height_test = (mdlParams["resize_large_ones"], mdlParams["resize_large_ones"])
                test_im = np.zeros([width_test, height_test])
                for i in range(mdlParams["multiCropEval"]):
                    im_crop = test_im[
                        np.int32(mdlParams["cropPositions"][u, i, 0] - height / 2) : np.int32(mdlParams["cropPositions"][u, i, 0] - height / 2) + height,
                        np.int32(mdlParams["cropPositions"][u, i, 1] - width / 2) : np.int32(mdlParams["cropPositions"][u, i, 1] - width / 2) + width,
                    ]
                    if im_crop.shape[0] != mdlParams["input_size"][0]:
                        print("Wrong shape", im_crop.shape[0], mdlParams["im_paths"][u])
                    if im_crop.shape[1] != mdlParams["input_size"][1]:
                        print("Wrong shape", im_crop.shape[1], mdlParams["im_paths"][u])

        # Meta scaler
        if mdlParams.get("meta_features", None) is not None and mdlParams["scale_features"]:
            mdlParams["feature_scaler_meta"] = sklearn.preprocessing.StandardScaler().fit(mdlParams["meta_array"][mdlParams["trainInd"], :])

        dataset_train = utils.ISICDataset(mdlParams, "trainInd")
        dataset_val = utils.ISICDataset(mdlParams, "valInd")

        if mdlParams["multiCropEval"] > 0:
            modelVars["dataloader_valInd"] = DataLoader(
                dataset_val,
                batch_size=mdlParams["multiCropEval"],
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            modelVars["dataloader_valInd"] = DataLoader(
                dataset_val,
                batch_size=mdlParams["batchSize"],
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )

        modelVars["dataloader_trainInd"] = DataLoader(
            dataset_train,
            batch_size=mdlParams["batchSize"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        # Define model
        modelVars["model"] = models.getModel(mdlParams)()

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

        if mdlParams.get("meta_features", None) is not None:
            modelVars["model"] = models.modify_meta(mdlParams, modelVars["model"])
            enforce_model_meta_input_dim(modelVars["model"], ckpt_meta_dim)

        modelVars["model"] = modelVars["model"].to(modelVars["device"])

        # Loss
        # NOTE: for external pred, weights don't matter much but keep as original
        if mdlParams["balance_classes"] == 3 or mdlParams["balance_classes"] == 0 or mdlParams["balance_classes"] == 12:
            modelVars["criterion"] = nn.CrossEntropyLoss()
        elif mdlParams["balance_classes"] == 8:
            modelVars["criterion"] = nn.CrossEntropyLoss(reduction="none")
        elif mdlParams["balance_classes"] == 6 or mdlParams["balance_classes"] == 7:
            # class_weights might be undefined in external flow; keep safe
            modelVars["criterion"] = nn.CrossEntropyLoss(reduction="none")
        elif mdlParams["balance_classes"] == 10:
            modelVars["criterion"] = utils.FocalLoss(mdlParams["numClasses"])
        elif mdlParams["balance_classes"] == 11:
            modelVars["criterion"] = utils.FocalLoss(mdlParams["numClasses"])
        else:
            modelVars["criterion"] = nn.CrossEntropyLoss()

        modelVars["optimizer"] = optim.Adam(modelVars["model"].parameters(), lr=mdlParams["learning_rate"])
        modelVars["scheduler"] = lr_scheduler.StepLR(
            modelVars["optimizer"], step_size=mdlParams["lowerLRAfter"], gamma=1 / np.float32(mdlParams["LRstep"])
        )
        modelVars["softmax"] = nn.Softmax(dim=1)

        # Load weights
        modelVars["model"].load_state_dict(ckpt_state_dict)

        modelVars["model"].eval()
        with torch.no_grad():
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(
                mdlParams, "valInd", modelVars
            )

        allData["extPred"][cv] = predictions
        print("extPred shape", allData["extPred"][cv].shape)

        pklFileName = cfg_name + "_" + ckpt_arg + "_" + str(int(chosen_step)) + "_predn.pkl"

# -----------------------------
# Mean results over all folds + save
# -----------------------------
np.set_printoptions(precision=4)
print("-------------------------------------------------")
print("Mean over all Folds")
print("-------------------------------------------------")
print("F1 Score", np.array([np.mean(allData["f1Best"])]), "+-", np.array([np.std(allData["f1Best"])]))
print("Sensitivtiy", np.mean(allData["sensBest"], 0), "+-", np.std(allData["sensBest"], 0))
print("Specificity", np.mean(allData["specBest"], 0), "+-", np.std(allData["specBest"], 0))
print("Mean Specificity", np.array([np.mean(allData["specBest"])]), "+-", np.array([np.std(np.mean(allData["specBest"], 1))]))
print("Accuracy", np.array([np.mean(allData["accBest"])]), "+-", np.array([np.std(allData["accBest"])]))
print("Per Class Accuracy", np.mean(allData["waccBest"], 0), "+-", np.std(allData["waccBest"], 0))
print("Weighted Accuracy", np.array([np.mean(allData["waccBest"])]), "+-", np.array([np.std(np.mean(allData["waccBest"], 1))]))
print("AUC", np.mean(allData["aucBest"], 0), "+-", np.std(allData["aucBest"], 0))
print("Mean AUC", np.array([np.mean(allData["aucBest"])]), "+-", np.array([np.std(np.mean(allData["aucBest"], 1))]))

if pklFileName is None:
    # Fallback (should not happen if you ran at least one CV fold)
    pklFileName = cfg_name + "_" + ckpt_arg + "_eval.pkl"

save_path = os.path.join(mdlParams["saveDirBase"], pklFileName)
with open(save_path, "wb") as f:
    pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)

print("[EVAL] Saved:", save_path)
