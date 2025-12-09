import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
import utils
from pprint import pprint



def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    if isinstance(feats, torch.Tensor):
        feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
        return feats.cuda()
    elif isinstance(feats, list):
        return [metrics.remove_outliers(f.float(), q=q, exact=exact).cuda() for f in feats]
    else:
        raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    """
    Find best-aligned y-layer for each x-layer.

    Args:
        x_feats: torch.Tensor (N, Lx, D) OR list of Lx tensors (N, D)
        y_feats: torch.Tensor (N, Ly, D) OR list of Ly tensors (N, D)
        metric: alignment metric name
        topk: for knn-based metrics
        normalize: L2 normalize per layer before scoring

    Returns:
        best_alignment_layers: torch.LongTensor shape (Lx,)
        best_alignment_score_perlayer: torch.FloatTensor shape (Lx,)
    """

    # Convert x into list-of-layer tensors
    if isinstance(x_feats, torch.Tensor):
        x_layers = [x_feats[:, i, :] for i in range(x_feats.shape[1])]
    elif isinstance(x_feats, list):
        x_layers = x_feats
    else:
        raise ValueError(f"Unsupported x_feats type: {type(x_feats)}")

    # Convert y into list-of-layer tensors
    if isinstance(y_feats, torch.Tensor):
        y_layers = [y_feats[:, j, :] for j in range(y_feats.shape[1])]
    elif isinstance(y_feats, list):
        y_layers = y_feats
    else:
        raise ValueError(f"Unsupported y_feats type: {type(y_feats)}")

    Lx = len(x_layers)

    best_alignment_layers = torch.empty(Lx, dtype=torch.long)
    best_alignment_score_perlayer = torch.empty(Lx, dtype=torch.float32)

    kwargs = {}
    if "knn" in metric:
        kwargs["topk"] = topk

    for i, x in enumerate(x_layers):
        best_score = -float("inf")
        best_j = 0

        for j, y in enumerate(y_layers):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned, y_aligned = x, y

            score = metrics.AlignmentMetrics.measure(metric, x_aligned, y_aligned, **kwargs)
            score_val = float(score.item() if isinstance(score, torch.Tensor) else score)

            if score_val > best_score:
                best_score = score_val
                best_j = j

        best_alignment_layers[i] = best_j
        best_alignment_score_perlayer[i] = best_score

    return best_alignment_layers, best_alignment_score_perlayer

    
def compute_alignment(
    x_feat_paths,
    y_feat_paths,
    models_x_names,
    models_y_names,
    metric,
    topk,
    precise=True,
    normalize=True,
):
    """
    For each pair (model_x, model_y):
      - load features
      - compute best y-layer for each x-layer
      - store best layers + best scores

    Returns:
        results: dict
            results[model_x][model_y] = {
                "best_layers": np.ndarray (Lx,),
                "best_scores": np.ndarray (Lx,)
            }
    """

    # Nested results dict
    results = {mx: {} for mx in models_x_names}

    pbar = tqdm(total=len(x_feat_paths) * len(y_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        mx = models_x_names[i]

        raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
        if isinstance(raw_x, torch.Tensor):
            x_feats = prepare_features(raw_x.float(), exact=precise)
        else:
            x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]

        for j, y_fp in enumerate(y_feat_paths):
            my = models_y_names[j]

            raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]
            if isinstance(raw_y, torch.Tensor):
                y_feats = prepare_features(raw_y.float(), exact=precise)
            else:
                y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]

            best_layers, best_scores = compute_score(
                x_feats,
                y_feats,
                metric=metric,
                topk=topk,
                normalize=normalize,
            )

            results[mx][my] = {
                "best_layers": best_layers.detach().cpu().numpy(),
                "best_scores": best_scores.detach().cpu().numpy(),
            }

            pbar.update(1)

            # Free y feats each inner loop
            del y_feats
            torch.cuda.empty_cache()

        # Free x feats after finishing all y models
        del x_feats
        torch.cuda.empty_cache()

    pbar.close()
    return results


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--modality_x",     type=str, default="all", choices=["vision", "language", "imggpt", "all"])
    parser.add_argument("--prompt_x",       action="store_true")
    parser.add_argument("--pool_x",         type=str, default=None, choices=['avg', 'cls'])
    
    parser.add_argument("--modality_y",     type=str, default="all", choices=["vision", "language", "imggpt", "all"])
    parser.add_argument("--prompt_y",       action="store_true")
    parser.add_argument("--pool_y",         type=str, default=None, choices=['avg', 'cls'])

    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)

    parser.add_argument("--input_dir",      type=str, default="./results/features")
    parser.add_argument("--output_dir",     type=str, default="./results/layercomp")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    save_path = utils.to_alignment_filename(
            args.output_dir, args.dataset, args.modelset,
            args.modality_x, args.pool_x, args.prompt_x,
            args.modality_y, args.pool_y, args.prompt_y,
            args.metric, args.topk
    )
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"layer computation already exists at {save_path}")
        exit()
    
    llm_models, lvm_models, imggpt_models = get_models(args.modelset, modality='all')
    # models_x = llm_models if args.modality_x == "language" else lvm_models
    # models_y = llm_models if args.modality_y == "language" else lvm_models
    modality_to_models = {"language": llm_models, "vision": lvm_models, "imggpt": imggpt_models}
    models_x = modality_to_models[args.modality_x]
    models_y = modality_to_models[args.modality_y]
    
    models_x_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool_x, args.prompt_x) for m in models_x]
    models_y_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool_y, args.prompt_y) for m in models_y]
    
    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn
    
    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if 'knn' in args.metric:
        print(f"topk:\t{args.topk}")
    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')

    results = compute_alignment(
        models_x_paths, models_y_paths,
        models_x, models_y,
        args.metric, args.topk, args.precise
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_pt = os.path.splitext(save_path)[0] + ".pt"
    torch.save(results, save_pt)
    print(f"saved nested results to {save_pt}")