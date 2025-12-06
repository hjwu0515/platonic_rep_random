import platonic
from tqdm.auto import trange
import torch 
from pprint import pprint

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from transformers import ImageGPTImageProcessor, ImageGPTForImageClassification


# setup platonic metric
platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is the dataset 
                    subset="wit_1024",    # <--- this is the subset
                    models=["openllama_7b", "dinov2_l"], 
                    ) # you can also pass in device and dtype as arguments

# load images
images = platonic_metric.get_data(modality="image")

# your model (e.g. we will use dinov2 as an example)
model_name = "openai/imagegpt-small"
vision_model = ImageGPTForImageClassification.from_pretrained(model_name, output_hidden_states=True).cuda().eval()
processor = ImageGPTImageProcessor.from_pretrained(model_name)

# transform = create_transform(
#     **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
# )

# # extract features
# return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
# vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

lvm_feats = []
batch_size = 32

for i in trange(0, len(images), batch_size):
    batch_imgs = images[i:i+batch_size]
    inputs = processor(batch_imgs, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    with torch.no_grad():
        outputs = vision_model(input_ids=input_ids, output_hidden_states=True)
    hidden = torch.stack(outputs.hidden_states)
    hidden = hidden.permute(1, 0, 2, 3).cpu()
    feats = hidden.mean(dim=2)
    lvm_feats.append(feats)
    
# compute score 
lvm_feats = torch.cat(lvm_feats)
score = platonic_metric.score(lvm_feats, metric="mutual_knn", topk=10, normalize=True)
pprint(score) # it will print the score and the index of the layer the maximal alignment happened