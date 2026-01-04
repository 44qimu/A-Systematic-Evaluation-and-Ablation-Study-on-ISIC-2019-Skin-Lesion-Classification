# -*- coding: utf-8 -*-
import os
import re
import csv
import pickle
from glob import glob

import numpy as np
import imagesize


def init(mdlParams_):
    mdlParams = {}

    # =======================
    # Basic paths (same style as your ss_meta)
    # =======================
    mdlParams['saveDir'] = os.path.join(mdlParams_['pathBase'], 'data', 'saveDir') + '/'
    mdlParams['dataDir'] = os.path.join(mdlParams_['pathBase'], 'data', 'dataDir')
    os.makedirs(mdlParams['saveDir'], exist_ok=True)

    # =======================
    # Model Selection (align to ss_meta)
    # =======================
    mdlParams['model_type'] = 'efficientnet-b0'
    mdlParams['dataset_names'] = ['official']
    mdlParams['file_ending'] = '.jpg'

    mdlParams['exclude_inds'] = False
    mdlParams['same_sized_crops'] = True
    mdlParams['multiCropEval'] = 25            # ss_meta = 25
    mdlParams['var_im_size'] = True
    mdlParams['orderedCrop'] = True
    mdlParams['voting_scheme'] = 'average'
    mdlParams['classification'] = True
    mdlParams['balance_classes'] = 9
    mdlParams['extra_fac'] = 1.0
    mdlParams['numClasses'] = 9
    mdlParams['no_c9_eval'] = True
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 5
    mdlParams['trans_norm_first'] = True
    mdlParams['input_size'] = [224, 224, 3]

    # =======================
    # Training Parameters (align to ss_meta)
    # =======================
    mdlParams['batchSize'] = 20
    mdlParams['learning_rate'] = 1.5e-5
    mdlParams['lowerLRAfter'] = 5
    mdlParams['lowerLRat'] = 10
    mdlParams['LRstep'] = 5
    mdlParams['training_steps'] = 15
    mdlParams['display_step'] = 5

    mdlParams['scale_targets'] = False
    mdlParams['peak_at_testerr'] = False
    mdlParams['print_trainerr'] = False
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])
    mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])

    # =======================
    # Data AUG (align to ss_meta)
    # =======================
    mdlParams['full_color_distort'] = True     # ss_meta has this
    mdlParams['autoaugment'] = True
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8, 1.2)
    mdlParams['shear'] = 10
    mdlParams['cutout'] = 16

    # =======================
    # Meta settings (align to ss_meta)
    # =======================
    mdlParams['meta_features'] = ['age_num', 'sex_oh', 'loc_oh']   # ss_meta: NO img_size
    mdlParams['meta_feature_sizes'] = [1, 2, 8]
    mdlParams['encode_nan'] = False

    # ss_meta meta-net settings (keep aligned)
    mdlParams['model_load_path'] = os.path.join(mdlParams['saveDir'], '2019.test_effb0_ss')
    mdlParams['fc_layers_before'] = [256, 256]
    scale_up_with_larger_b = 1.0
    mdlParams['fc_layers_after'] = [int(1024 * scale_up_with_larger_b)]
    mdlParams['freeze_cnn'] = False
    mdlParams['learning_rate_meta'] = 1.5e-4
    mdlParams['drop_augment'] = 0.12
    mdlParams['dropout_meta'] = 0.25
    mdlParams['scale_features'] = True

    # =======================
    # Data
    # =======================
    mdlParams['preload'] = False

    # -------------------------------------------------------
    # 1) Load labels -> labels_dict  (BOM-safe)
    # -------------------------------------------------------
    mdlParams['labels_dict'] = {}
    labels_root = os.path.join(mdlParams['dataDir'], 'labels')
    all_label_sets = glob(os.path.join(labels_root, '*/'))

    for set_dir in all_label_sets:
        if not any(ds in set_dir for ds in mdlParams['dataset_names']):
            continue

        files = sorted(glob(os.path.join(set_dir, '*')))
        csv_file = None
        for f in files:
            if f.lower().endswith('.csv'):
                csv_file = f
                break
        if csv_file is None:
            continue

        with open(csv_file, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if not row:
                    continue
                row0 = (row[0] or '').strip().replace('\ufeff', '')
                if row0.lower() in ('image', 'image_id'):
                    continue

                # drop duplicates if downsampled exists
                if row0 + '_downsampled' in mdlParams['labels_dict']:
                    continue

                if mdlParams['numClasses'] == 7:
                    mdlParams['labels_dict'][row0] = np.array(
                        [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
                         int(float(row[5])), int(float(row[6])), int(float(row[7]))],
                        dtype=np.int64
                    )
                elif mdlParams['numClasses'] == 8:
                    class_8 = int(float(row[8])) if (len(row) >= 9 and row[8] != '') else 0
                    mdlParams['labels_dict'][row0] = np.array(
                        [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
                         int(float(row[5])), int(float(row[6])), int(float(row[7])), class_8],
                        dtype=np.int64
                    )
                elif mdlParams['numClasses'] == 9:
                    class_8 = int(float(row[8])) if (len(row) >= 9 and row[8] != '') else 0
                    class_9 = int(float(row[9])) if (len(row) >= 10 and row[9] != '') else 0
                    mdlParams['labels_dict'][row0] = np.array(
                        [int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4])),
                         int(float(row[5])), int(float(row[6])), int(float(row[7])), class_8, class_9],
                        dtype=np.int64
                    )
                else:
                    raise ValueError(f"Unsupported numClasses={mdlParams['numClasses']}")

    if len(mdlParams['labels_dict']) == 0:
        raise RuntimeError(f"No labels loaded. Check: {labels_root}")

    # -------------------------------------------------------
    # 2) Load meta pkl -> meta_dict  (key forced to str)
    # -------------------------------------------------------
    mdlParams['meta_dict'] = {}
    meta_root = os.path.join(mdlParams['dataDir'], 'meta_data')
    all_meta_sets = glob(os.path.join(meta_root, '*/'))
    meta_data = None

    for set_dir in all_meta_sets:
        if not any(ds in set_dir for ds in mdlParams['dataset_names']):
            continue

        files = sorted(glob(os.path.join(set_dir, '*')))
        pkl_file = None
        for f in files:
            if f.lower().endswith('.pkl'):
                pkl_file = f
                break
        if pkl_file is None:
            continue

        with open(pkl_file, 'rb') as f:
            meta_data = pickle.load(f)

        for k in range(len(meta_data['im_name'])):
            im_name = str(meta_data['im_name'][k])
            fv_parts = []

            # age_num
            if 'age_num' in mdlParams['meta_features'] and 'age_num' in meta_data:
                fv_parts.append(np.array([meta_data['age_num'][k]], dtype=np.float32))

            # loc_oh
            if 'loc_oh' in mdlParams['meta_features'] and 'loc_oh' in meta_data:
                arr = meta_data['loc_oh'][k, :]
                if (not mdlParams['encode_nan']) and arr.shape[0] > 1:
                    arr = arr[1:]
                fv_parts.append(arr.astype(np.float32))

            # sex_oh
            if 'sex_oh' in mdlParams['meta_features'] and 'sex_oh' in meta_data:
                arr = meta_data['sex_oh'][k, :]
                if (not mdlParams['encode_nan']) and arr.shape[0] > 1:
                    arr = arr[1:]
                fv_parts.append(arr.astype(np.float32))

            fv = np.concatenate(fv_parts, axis=0).astype(np.float32) if fv_parts else np.zeros((1,), dtype=np.float32)
            mdlParams['meta_dict'][im_name] = fv

        # -------------------------------------------------------
        # 3) Auto-label GAN_* if not present in official CSV
        #    (GAN_<CLASS>_xxxxxx -> one-hot)   [v2 behavior]
        # -------------------------------------------------------
        cls8 = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        cls2i = {c: i for i, c in enumerate(cls8)}

        added = 0
        for im_name_raw in meta_data['im_name']:
            s = str(im_name_raw)
            if not s.startswith('GAN_'):
                continue
            if s in mdlParams['labels_dict']:
                continue
            parts = s.split('_')
            if len(parts) >= 2 and parts[1] in cls2i:
                y = np.zeros((mdlParams['numClasses'],), dtype=np.int64)
                y[cls2i[parts[1]]] = 1
                # last class stays 0
                mdlParams['labels_dict'][s] = y
                added += 1
        if added:
            print(f"[v2] Added GAN labels from name: {added}")

        break  # one pkl per dataset

    if meta_data is None or len(mdlParams['meta_dict']) == 0:
        raise FileNotFoundError(f"No meta .pkl found under: {meta_root}")

    # -------------------------------------------------------
    # 4) Build im_paths / labels_list / meta_list / key_list
    #    IMPORTANT v2 rule: OFFICIAL first, GAN appended
    # -------------------------------------------------------
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    mdlParams['meta_list'] = []
    mdlParams['key_list'] = []

    images_root = os.path.join(mdlParams['dataDir'], 'images')
    official_dir = os.path.join(images_root, 'official')
    if not os.path.isdir(official_dir):
        raise FileNotFoundError(f"Missing images directory: {official_dir}")

    # only accept exact '.jpg' to match ss_meta behavior ('.JPG' would not match key+'.jpg' in old code)
    img_files = sorted(glob(os.path.join(official_dir, '*' + mdlParams['file_ending'])))
    if len(img_files) == 0:
        raise RuntimeError(f"No images found under: {official_dir} ending with {mdlParams['file_ending']}")

    def _add_file(fp):
        key = os.path.splitext(os.path.basename(fp))[0]
        if key in mdlParams['labels_dict'] and key in mdlParams['meta_dict']:
            mdlParams['im_paths'].append(fp)
            mdlParams['labels_list'].append(mdlParams['labels_dict'][key])
            mdlParams['meta_list'].append(mdlParams['meta_dict'][key])
            mdlParams['key_list'].append(key)

    # pass 1: non-GAN
    for fp in img_files:
        key = os.path.splitext(os.path.basename(fp))[0]
        if not key.startswith('GAN_'):
            _add_file(fp)

    # pass 2: GAN
    for fp in img_files:
        key = os.path.splitext(os.path.basename(fp))[0]
        if key.startswith('GAN_'):
            _add_file(fp)

    mdlParams['labels_array'] = np.asarray(mdlParams['labels_list'], dtype=np.int64)
    mdlParams['meta_array'] = np.asarray(mdlParams['meta_list'], dtype=np.float32)

    if not (len(mdlParams['key_list']) == mdlParams['labels_array'].shape[0] == mdlParams['meta_array'].shape[0]):
        raise RuntimeError(
            "Mismatch among key_list / labels_array / meta_array sizes: "
            f"{len(mdlParams['key_list'])} vs {mdlParams['labels_array'].shape} vs {mdlParams['meta_array'].shape}"
        )

    print("[v2] labels_array shape:", mdlParams['labels_array'].shape)
    print("[v2] meta_array shape:", mdlParams['meta_array'].shape)
    if mdlParams['labels_array'].size:
        print("[v2] label mean:", np.mean(mdlParams['labels_array'], axis=0))
    print("[v2] GAN in key_list:", sum(k.startswith('GAN_') for k in mdlParams['key_list']))

    # -------------------------
    # v2 sanity: key_list order must be official first, GAN appended
    # -------------------------
    key_arr = np.asarray(mdlParams['key_list'], dtype=str)
    is_gan = np.char.startswith(key_arr, 'GAN_')
    gan_pos = np.where(is_gan)[0]

    if gan_pos.size == 0:
        raise RuntimeError("No GAN samples found in key_list, but you are using GAN-trainonly indices. Check GAN images/meta.")

    first_gan_idx = int(gan_pos[0])
    gan_count = int(gan_pos.size)
    mdlParams['first_gan_idx'] = first_gan_idx
    mdlParams['gan_count'] = gan_count

    if np.any(is_gan[:first_gan_idx]):
        raise RuntimeError("key_list ordering broken: found GAN before first_gan_idx (expected official first).")
    if np.any(~is_gan[first_gan_idx:]):
        raise RuntimeError("key_list ordering broken: found non-GAN after first_gan_idx (expected GAN appended).")

    print("[v2] first_gan_idx:", first_gan_idx, "gan_count:", gan_count)

    # -------------------------------------------------------
    # 5) Load indices (STRICT v2)
    #    ONLY allow gan_trainonly/trainonly_gan. Never read indices_isic2019.pkl.
    # -------------------------------------------------------
    idx1 = os.path.join(mdlParams['saveDir'], 'indices_isic2019_gan_trainonly.pkl')
    idx2 = os.path.join(mdlParams['saveDir'], 'indices_isic2019_trainonly_gan.pkl')
    if os.path.exists(idx1):
        idx_path = idx1
    elif os.path.exists(idx2):
        idx_path = idx2
    else:
        raise FileNotFoundError(
            "Required GAN indices pkl NOT found. Expected ONE of:\n"
            f" - {idx1}\n"
            f" - {idx2}\n"
            "And it must be placed under saveDir."
        )

    print("[v2] Using indices pkl:", idx_path)
    mdlParams['indices_pkl'] = idx_path

    with open(idx_path, 'rb') as f:
        indices = pickle.load(f)

    if 'trainIndCV' not in indices or 'valIndCV' not in indices:
        raise KeyError(f"indices pkl missing keys. Found keys: {list(indices.keys())}")

    mdlParams['trainIndCV'] = indices['trainIndCV']
    mdlParams['valIndCV'] = indices['valIndCV']

    if len(mdlParams['trainIndCV']) != mdlParams['numCV'] or len(mdlParams['valIndCV']) != mdlParams['numCV']:
        raise ValueError(
            f"Fold count mismatch: numCV={mdlParams['numCV']}, "
            f"train folds={len(mdlParams['trainIndCV'])}, val folds={len(mdlParams['valIndCV'])}"
        )

    # STRICT fold checks: val has NO GAN, train has ALL GAN, indices in range
    n_total = len(mdlParams['key_list'])
    for i in range(mdlParams['numCV']):
        tr = np.asarray(mdlParams['trainIndCV'][i], dtype=np.int64)
        va = np.asarray(mdlParams['valIndCV'][i], dtype=np.int64)

        if tr.size == 0 or va.size == 0:
            raise ValueError(f"Empty indices in fold {i}: train={tr.size}, val={va.size}")

        if tr.min() < 0 or va.min() < 0 or tr.max() >= n_total or va.max() >= n_total:
            raise IndexError(
                f"Index out of range in fold {i}: "
                f"train[{tr.min()}..{tr.max()}], val[{va.min()}..{va.max()}], n_total={n_total}"
            )

        # val must contain no GAN
        if va.max() >= first_gan_idx:
            raise RuntimeError(
                f"GAN samples found in val fold {i} (val max idx {va.max()} >= first_gan_idx {first_gan_idx})."
            )

        # train must include all GAN
        gan_in_train = int(np.sum(tr >= first_gan_idx))
        if gan_in_train != gan_count:
            raise RuntimeError(
                f"Fold {i}: expected ALL GAN in train. got gan_in_train={gan_in_train}, expected={gan_count}."
            )

    # print fold sizes (same as ss_meta)
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        print(np.asarray(mdlParams['trainIndCV'][i]).shape)
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        print(np.asarray(mdlParams['valIndCV'][i]).shape)

    # -------------------------------------------------------
    # 6) Ordered multi-crop positions (match ss_meta logic/style, including its width/height assignment)
    # -------------------------------------------------------
    if mdlParams['orderedCrop']:
        mdlParams['cropPositions'] = np.zeros([len(mdlParams['im_paths']), mdlParams['multiCropEval'], 2], dtype=np.int64)

        for u in range(len(mdlParams['im_paths'])):
            # NOTE: keep ss_meta's variable naming (it assigns height,width = imagesize.get which returns (w,h))
            height, width = imagesize.get(mdlParams['im_paths'][u])

            if width < mdlParams['input_size'][0]:
                height = int(mdlParams['input_size'][0] / float(width)) * height
                width = mdlParams['input_size'][0]
            if height < mdlParams['input_size'][0]:
                width = int(mdlParams['input_size'][0] / float(height)) * width
                height = mdlParams['input_size'][0]

            ind = 0
            for i in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                for j in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                    mdlParams['cropPositions'][u, ind, 0] = mdlParams['input_size'][0] / 2 + i * (
                        (width - mdlParams['input_size'][1]) / (np.sqrt(mdlParams['multiCropEval']) - 1)
                    )
                    mdlParams['cropPositions'][u, ind, 1] = mdlParams['input_size'][1] / 2 + j * (
                        (height - mdlParams['input_size'][0]) / (np.sqrt(mdlParams['multiCropEval']) - 1)
                    )
                    ind += 1

        # sanity checks (as in ss_meta)
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for u in range(len(mdlParams['im_paths'])):
            height_test, width_test = imagesize.get(mdlParams['im_paths'][u])

            if width_test < mdlParams['input_size'][0]:
                height_test = int(mdlParams['input_size'][0] / float(width_test)) * height_test
                width_test = mdlParams['input_size'][0]
            if height_test < mdlParams['input_size'][0]:
                width_test = int(mdlParams['input_size'][0] / float(height_test)) * width_test
                height_test = mdlParams['input_size'][0]

            test_im = np.zeros([width_test, height_test])
            for i in range(mdlParams['multiCropEval']):
                im_crop = test_im[
                    np.int32(mdlParams['cropPositions'][u, i, 0] - height / 2):np.int32(mdlParams['cropPositions'][u, i, 0] - height / 2) + height,
                    np.int32(mdlParams['cropPositions'][u, i, 1] - width / 2):np.int32(mdlParams['cropPositions'][u, i, 1] - width / 2) + width
                ]
                if im_crop.shape[0] != mdlParams['input_size'][0]:
                    print("Wrong shape", im_crop.shape[0], mdlParams['im_paths'][u])
                if im_crop.shape[1] != mdlParams['input_size'][1]:
                    print("Wrong shape", im_crop.shape[1], mdlParams['im_paths'][u])

    # Test indices
    mdlParams['testInd'] = []
    mdlParams['testSetState'] = 'val'

    return mdlParams
