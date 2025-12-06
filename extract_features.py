import gc
import os
import argparse

from tqdm import trange

import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from tasks import get_models
from models import load_llm, load_tokenizer
import utils 
    

def extract_llm_features(filenames, dataset, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    texts = [str(x['text'][args.caption_idx]) for x in dataset]
        
    for llm_model_name in filenames[::-1]:
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, llm_model_name,
            pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{llm_model_name}")
        print(f'save_path: \t{save_path}')
        
        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue
        
        language_model = load_llm(llm_model_name, qlora=args.qlora, force_download=args.force_download)
        llm_param_count = sum([p.numel() for p in language_model.parameters()])
        tokenizer = load_tokenizer(llm_model_name)
    
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
        llm_feats, losses, bpb_losses = [], [], []

        # hack to get around HF mapping data incorrectly when using model-parallel
        device = next(language_model.parameters()).device

        for i in trange(0, len(dataset), args.batch_size):
            # get embedding cuda device
            token_inputs = {k: v[i:i+args.batch_size].to(device).long() for (k, v) in tokens.items()}

            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )

                loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
                losses.extend(avg_loss.cpu())
                
                bpb = utils.cross_entropy_to_bits_per_unit(loss.cpu(), texts[i:i+args.batch_size], unit="byte")
                bpb_losses.extend(bpb)
                
                # make sure to do all the processing in cpu to avoid memory problems
                if args.pool == 'avg':
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif args.pool == 'last':
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2) 
                else:
                    raise NotImplementedError(f"unknown pooling {args.pool}")
                llm_feats.append(feats.cpu())

        print(f"average loss:\t{torch.stack(losses).mean().item()}")
        save_dict = {
            "feats": torch.cat(llm_feats).cpu(),
            "num_params": llm_param_count,
            "mask": tokens["attention_mask"].cpu(),
            "loss": torch.stack(losses).mean(),
            "bpb": torch.stack(bpb_losses).mean(),
        }

        torch.save(save_dict, save_path)

        del language_model, tokenizer, llm_feats, llm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return
    
        
def extract_lvm_features(filenames, dataset, args):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argparse arguments
    """
    assert args.pool == 'cls', "pooling is not supported for lvm features"
    
    for lvm_model_name in filenames:
        assert 'vit' in lvm_model_name, "only vision transformers are supported"
        
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, lvm_model_name,
            pool=args.pool, prompt=None, caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{lvm_model_name}")
        print(f'save_path: \t{save_path}')

        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue

        vision_model = timm.create_model(lvm_model_name, pretrained=True).cuda().eval()
        lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

        transform = create_transform(
            **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        )

        if "vit" in lvm_model_name:
            return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
        else:
            raise NotImplementedError(f"unknown model {lvm_model_name}")

        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
        lvm_feats = []

        for i in trange(0, len(dataset), args.batch_size):
            with torch.no_grad():
                ims = torch.stack([transform(dataset[j]['image']) for j in range(i, i+args.batch_size)]).cuda()
                lvm_output = vision_model(ims)

                if args.pool == "cls":
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                    
                lvm_feats.append(feats.cpu())

        torch.save({"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path)

        del vision_model, transform, lvm_feats, lvm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
def extract_imagegpt_features(model_names, dataset, args):
    """
    Extract features from ImageGPT models.

    Args:
        model_names: list of model identifiers (e.g. ["openai/imagegpt-small"])
        dataset: list of dicts with {'image': PIL.Image}
        args: argparse arguments (must have output_dir, dataset, subset, batch_size, force_remake)
    """
    for model_name in model_names:

        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, model_name,
            pool=None, prompt=None, caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"\nDataset: \t{args.dataset}")
        print(f"Subset:  \t{args.subset}")
        print(f"Model:   \t{model_name}")
        print(f"Saving to:\t{save_path}")

        if os.path.exists(save_path) and not args.force_remake:
            print("File exists — skipping.")
            continue

        # Load model + processor
        from transformers import ImageGPTImageProcessor, ImageGPTForImageClassification

        processor = ImageGPTImageProcessor.from_pretrained(model_name)
        model = ImageGPTForImageClassification.from_pretrained(
            model_name,
            output_hidden_states=True
        ).cuda().eval()

        param_count = sum(p.numel() for p in model.parameters())

        lvm_feats = []

        # Batch images
        for i in trange(0, len(dataset), args.batch_size):

            batch_images = [dataset[j]['image'] for j in range(i, i + args.batch_size)]
            inputs = processor(batch_images, return_tensors="pt")

            input_ids = inputs["input_ids"].cuda()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_hidden_states=True)

            # hidden_states: list of [B, T, D]
            hidden = torch.stack(outputs.hidden_states)     # [L, B, T, D]
            hidden = hidden.permute(1, 0, 2, 3).cpu()       # [B, L, T, D]

            # mean over tokens → [B, L, D]
            feats = hidden.mean(dim=2)

            lvm_feats.append(feats)

        # Save
        torch.save({
            "feats": torch.cat(lvm_feats),
            "num_params": param_count,
            "model": model_name,
        }, save_path)

        # Cleanup
        del model, processor, lvm_feats
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--num_samples",    type=int, default=1024)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--pool",           type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--prompt",         action="store_true")
    parser.add_argument("--dataset",        type=str, default="prh")
    parser.add_argument("--subset",         type=str, default="wit_1024")
    parser.add_argument("--caption_idx",    type=int, default=0)
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "all", "imggpt"])
    parser.add_argument("--output_dir",     type=str, default="./results/features")
    parser.add_argument("--qlora",          action="store_true")
    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models, imggpt_models = get_models(args.modelset, modality=args.modality)
    
    # load dataset once outside    
    dataset = load_dataset(args.dataset, revision=args.subset, split='train')

    if args.modality in ["all", "language"]:
        # extract all language model features
        extract_llm_features(llm_models, dataset, args)
    
    if args.modality in ["all", "vision"]:
        # extract all vision model features
        extract_lvm_features(lvm_models, dataset, args)
        
    if args.modality in ["all", "imggpt"]:
        # extract all image gpt features
        extract_imagegpt_features(imggpt_models, dataset, args)
