import gc
import os
import argparse

from tqdm import trange
import torch.nn.functional as F

import torch
import math
import numpy as np

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from transformers import ImageGPTImageProcessor, ImageGPTModel

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
        lvm_feats, losses, bpp_losses = [], [], []
        loss_min, loss_max = float('inf'), float('-inf')
        bpp_min, bpp_max = float('inf'), float('-inf')

        for i in trange(0, len(dataset), args.batch_size):
            with torch.no_grad():
                ims = torch.stack([transform(dataset[j]['image']) for j in range(i, i+args.batch_size)]).cuda()
                lvm_output = vision_model(ims)

                if args.pool == "cls":
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                    
                lvm_feats.append(feats.cpu())
                
                # forward through full ViT
                full_out = vision_model(ims)

                embed = None
                if isinstance(full_out, torch.Tensor):
                    embed = full_out
                elif isinstance(full_out, dict):
                    for v in full_out.values():
                        if isinstance(v, torch.Tensor):
                            embed = v
                            break
                elif hasattr(full_out, "__dict__"):
                    for k, v in full_out.__dict__.items():
                        if isinstance(v, torch.Tensor):
                            embed = v
                            break
                if embed is None and hasattr(full_out, "sample"):
                    embed = full_out.sample
                if embed is None:
                    raise ValueError(f"Could not find tensor in model output type: {type(full_out)}")  
                    
                if embed.ndim == 2:
                    loss = torch.mean(embed**2, dim=1)
                elif embed.ndim == 3:
                    loss = torch.mean(embed**2, dim=[1, 2])
                elif embed.ndim == 4:
                    loss = torch.mean(embed**2, dim=[1,2,3])
                else:
                    print(f"Warning: unsupported embed.ndim={embed.ndim}, skipping.")
                    continue
                batch_min = loss.min().item()
                batch_max = loss.max().item()
                loss_min = min(loss_min, batch_min)
                loss_max = max(loss_max, batch_max)
                loss = (loss - loss_min) / (loss_max - loss_min)
                losses.append(loss.cpu())

                # compute bits per pixel
                H, W = ims.shape[2], ims.shape[3]
                bpp = loss / (H * W * np.log(2))
                bpp_min = min(bpp_min, bpp.min().item())
                bpp_max = max(bpp_max, bpp.max().item())
                bpp_losses.append(bpp.cpu())

        losses_tensor = torch.cat(losses)
        loss_norm = (losses_tensor - loss_min) / (loss_max - loss_min + 1e-12)
        bpp_tensor = torch.cat(bpp_losses)
        bpp_norm = (bpp_tensor - bpp_min) / (bpp_max - bpp_min + 1e-12)
        torch.save({
            "feats": torch.cat(lvm_feats), 
            "num_params": lvm_param_count,
            "loss": loss_norm,
            "bpp": bpp_norm
        }, save_path)

        del vision_model, transform, lvm_feats, lvm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        
def extract_imagegpt_features(model_names, dataset, args, use_all_layers=True, keep_from=1):
    """
    Extract features from ImageGPT models for platonic-rep alignment.

    Args:
        model_names: list of model identifiers (e.g. ["openai/imagegpt-small"])
        dataset: list / Dataset of dicts with {'image': PIL.Image}
        args: argparse arguments (must have output_dir, dataset, subset, batch_size, force_remake)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        processor = ImageGPTImageProcessor.from_pretrained(model_name)
        model = ImageGPTModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(device).eval()

        param_count = sum(p.numel() for p in model.parameters())
        all_batches, all_loss, all_bpp = [], [], []
        loss_min, loss_max = float('inf'), float('-inf')
        bpp_min, bpp_max = float('inf'), float('-inf')

        n = len(dataset)
        for i in trange(0, n, args.batch_size):
            j_end = min(i + args.batch_size, n)
            batch_images = [dataset[j]["image"] for j in range(i, j_end)]

            inputs = processor(batch_images, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_hidden_states=True)

            # outputs.hidden_states: tuple length (n_layers + 1), each [B, T, D]
            hs = outputs.hidden_states  # tuple

            # Option 1: use all transformer layers (drop embedding layer at index 0)
            if use_all_layers:
                hs = hs[keep_from:]   # e.g. keep_from=1 → drop embedding, keep all blocks

            # Stack → [L, B, T, D]
            hs = torch.stack(hs, dim=0)

            # Mean over tokens → [L, B, D]
            hs = hs.mean(dim=2)

            # Move batch to front → [B, L, D]
            feats = hs.permute(1, 0, 2)

            # Optional: per-layer L2 normalization
            feats = F.normalize(feats, dim=-1)

            all_batches.append(feats.cpu())
            
            LHS = outputs.last_hidden_state
            T = input_ids.shape[1]  # number of tokens
            H = W = int(math.sqrt(T))

            embed = LHS.mean(dim=1)
            loss = torch.mean(embed ** 2, dim=1)
            
            batch_min = loss.min().item()
            batch_max = loss.max().item()
            loss_min = min(loss_min, batch_min)
            loss_max = max(loss_max, batch_max)
            loss = (loss - loss_min) / (loss_max - loss_min)
            all_loss.append(loss.cpu())
            
            bpp = loss / (H * W * np.log(2))
            bpp_min = min(bpp_min, bpp.min().item())
            bpp_max = max(bpp_max, bpp.max().item())
            all_bpp.append(bpp.cpu())

        # Concatenate over batches → [N, L, D]
        all_feats = torch.cat(all_batches, dim=0)
        losses_tensor = torch.cat(all_loss)
        loss_norm = (losses_tensor - loss_min) / (loss_max - loss_min + 1e-12)
        bpp_tensor = torch.cat(all_bpp)
        bpp_norm = (bpp_tensor - bpp_min) / (bpp_max - bpp_min + 1e-12)

        torch.save(
            {
                "feats": all_feats,          # [N, L, D]
                "num_params": param_count,
                "loss": loss_norm,        # [N]
                "bpp": bpp_norm,          # [N]
                "model": model_name,
            },
            save_path,
        )

        del model, processor, all_batches, all_feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

def extract_diffusion_features(filenames, dataset, args):
    """
    Extract features from diffusion-based models (e.g., Stable Diffusion).
    - filenames: list of HF diffusion model IDs (Stable Diffusion-like)
    - dataset: huggingface dataset with images and optional captions
    - args: argparse args (reuses same pattern)
    Saves layer-wise features per sample with metadata.
    """


    # image preprocessing (resize/center-crop consistent with pipeline VAE expected size)
    preprocess = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


    for model_name in filenames:
        save_path = utils.to_feature_filename(args.output_dir, args.dataset, args.subset, model_name,
                                                pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\nProcessing diffusion model: {model_name} -> {save_path}")


        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue


        # load pipeline (choose dtype to save memory)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        pipe = pipe.to(device)
        pipe.safety_checker = None  # optional


        unet = pipe.unet
        vae = pipe.vae
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)


        param_count = sum(p.numel() for p in pipe.unet.parameters()) + sum(p.numel() for p in vae.parameters())
        if text_encoder is not None:
            param_count += sum(p.numel() for p in text_encoder.parameters())


        # helper: register forward hooks on UNet blocks to capture activations
        activations = {}
        hooks = []


        def make_hook(name):
            def hook(module, inp, out):
                # Some UNet blocks return tuples
                if isinstance(out, tuple):
                    out = out[0]  # take main feature map, ignore residuals

                if isinstance(out, torch.Tensor):
                    activations.setdefault(name, []).append(out.detach().cpu())
            return hook


        # Try registering hooks on down_blocks, mid_block, up_blocks if present
        for i, block in enumerate(getattr(unet, "down_blocks", [])):
            hooks.append(block.register_forward_hook(make_hook(f"down_{i}")))
        if hasattr(unet, "mid_block"):
            hooks.append(unet.mid_block.register_forward_hook(make_hook("mid")))
        for i, block in enumerate(getattr(unet, "up_blocks", [])):
            hooks.append(block.register_forward_hook(make_hook(f"up_{i}")))


        diffusion_feats = []     # list per batch: dict(layer_name -> tensor)
        meta_info = []           # store meta info (caption, timestep) per sample
        losses = []              # optional, if you compute any loss / metric


        # We'll process images in batches: encode with VAE to get latents, add noise for some timestep
        pipe.scheduler.set_timesteps(50)  # example: set scheduler timesteps; adjust per model/version
        timesteps = [10, 25, 40]         # example timesteps to probe (choose per experiment)


        for i in trange(0, len(dataset), args.batch_size):
            batch_images = []
            batch_texts = []
            end = min(i + args.batch_size, len(dataset))
            for j in range(i, end):
                img = dataset[j]['image']
                batch_images.append(preprocess(img).to(device))
                if "text" in dataset[j]:
                    batch_texts.append(str(dataset[j]['text'][args.caption_idx]))
                else:
                    batch_texts.append("")


            ims = torch.stack(batch_images)  # shape: Bx3xHxW


            # encode images -> latents via VAE encoder
            with torch.no_grad():
                # VAE expects images in [-1,1], normalized above does that
                encoded = vae.encode(ims.half() if device=="cuda" else ims).latent_dist
                latents = encoded.sample()  # shape: B x C x h x w
                latents = latents * vae.config.scaling_factor  # follow diffusers patterns


            # text conditioning
            if tokenizer is not None and text_encoder is not None:
                text_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
                text_embeds = text_encoder(text_inputs.input_ids)[0]
            else:
                text_embeds = None


            # for each chosen timestep, produce noisy latents and run UNet to capture activations
            batch_layer_feats = { }  # will be filled layer_name -> list of tensors for this batch
            for t in timesteps:
                # create noise
                noise = torch.randn_like(latents).to(device)
                # add noise using scheduler helper if available (safer) else simple add
                try:
                    noisy = pipe.scheduler.add_noise(latents, noise, torch.tensor([t]*latents.shape[0], device=device))
                except Exception:
                    # fallback simple noise injection (not equivalent to scheduler but ok for activations probing)
                    noisy = latents + noise


                # run UNet forward: sample/noisy_latents, timestep, encoder_hidden_states=text_embeds
                with torch.no_grad():
                    # UNet forward signature: (sample, timestep, encoder_hidden_states=...)
                    if text_embeds is not None:
                        out = unet(noisy.half() if device=="cuda" else noisy, torch.tensor([t], device=device), encoder_hidden_states=text_embeds.half() if device=="cuda" else text_embeds)
                    else:
                        out = unet(noisy.half() if device=="cuda" else noisy, torch.tensor([t], device=device))


                # after forward pass, hooks filled `activations`. collect & clear
                # activations entries are lists (one entry per forward call) — pop last
                for k, v_list in activations.items():
                    last_out = v_list.pop() if v_list else None
                    if last_out is not None:
                        # last_out shape depends on module; typically BxCxHxW or Bx...; apply pooling as requested
                        if args.pool == "avg":
                            pooled = last_out.mean(dim=[-2, -1]) if last_out.ndim == 4 else last_out.mean(dim=1)
                        elif args.pool == "cls":
                            # if module produces sequence-like tokens, pick first token
                            pooled = last_out[:, 0, :] if last_out.ndim == 3 else last_out.mean(dim=[-2, -1])
                        else:
                            pooled = last_out.view(last_out.shape[0], -1).cpu()
                        batch_layer_feats.setdefault(k, []).append(pooled)


            # stack and rearrange: for each layer, we have len(timesteps) pooled tensors of shape BxD → stack -> B x T x D
            per_batch_saved = {}
            for layer_name, pooled_list in batch_layer_feats.items():
                # pooled_list: list of pooled per-timestep tensors: [T tensors (B x D)]
                per_batch_saved[layer_name] = torch.stack(pooled_list, dim=1).cpu()  # B x T x D


            diffusion_feats.append(per_batch_saved)
            meta_info.extend([{"text": txt, "timesteps": timesteps} for txt in batch_texts])


            # optionally clear activations dict to keep memory low
            activations.clear()


        # remove hooks
        for h in hooks:
            h.remove()
        
        
        # ---- Build a single [N, L, D] tensor for alignment ----
        layer_names = list(diffusion_feats[0].keys())
        all_layer_feats = []
        
        for layer in layer_names:
            # shape: [N, T, D]
            per_layer = torch.cat([b[layer] for b in diffusion_feats], dim=0)
        
            # pool across timesteps so it becomes [N, D]
            per_layer = per_layer.mean(dim=1)
            all_layer_feats.append(per_layer)
        
        # stack layers -> [N, L, D]
        max_dim = max(f.shape[1] for f in all_layer_feats)

        padded_feats = []
        for f in all_layer_feats:
            if f.shape[1] < max_dim:
                pad = torch.zeros(f.shape[0], max_dim - f.shape[1])
                f = torch.cat([f, pad], dim=1)
            padded_feats.append(f)
        
        # stack -> [N, L, D]
        feat_tensor = torch.stack(padded_feats, dim=1)
        
        # ---- Save in expected format ----
        save_dict = {
            "feats": feat_tensor,     
            "num_params": param_count,
            "layers": {name: i for i, name in enumerate(layer_names)},
            "meta": meta_info
        }
        
        torch.save(save_dict, save_path)


        # cleanup
        del pipe, unet, vae, tokenizer, text_encoder, diffusion_feats
        torch.cuda.empty_cache()
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
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "imggpt", "diffusion", "all"])
    parser.add_argument("--output_dir",     type=str, default="./results/features")
    parser.add_argument("--qlora",          action="store_true")
    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models, imggpt_models, diffusion_models = get_models(args.modelset, modality=args.modality)
    
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

    if args.modality in ["all", "diffusion"]:
        # extract all diffusion model features
        extract_diffusion_features(diffusion_models, dataset, args)
