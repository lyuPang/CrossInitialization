from models.clip_model import CLIPTextModel
from transformers import CLIPTokenizer
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import random

@torch.no_grad()
def celeb_names_cross_init(
        celeb_path : str,
        tokenizer : CLIPTokenizer,
        text_encoder: CLIPTextModel,
        n_column: int=2,
    ):
    with open(celeb_path, 'r') as f:
        celeb_names=f.read().splitlines()
    # get embeddings
    col_embeddings=[[]for _ in range(n_column)]
    for name in tqdm(celeb_names,desc='get embeddings'):
        token_ids=tokenizer(
            name,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] # (n,)
        embeddings = text_encoder.get_input_embeddings().weight.data[token_ids] # (n,1024)

        # remove the start and end characters
        for i in range(1,min(embeddings.shape[0]-1,n_column+1)):
            col_embeddings[i-1].append(embeddings[i].unsqueeze(0))
    # mean for all names
    for i in range(n_column): 
        col_embeddings[i]=torch.cat(col_embeddings[i]).mean(dim=0).unsqueeze(0)
    col_embeddings=torch.cat(col_embeddings) #(n,1024)
    bos_embed,eos_embed,pad_embed=text_encoder.get_input_embeddings().weight.data[[tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.pad_token_id]]
    input_embeds=torch.cat([bos_embed.unsqueeze(0),col_embeddings,eos_embed.unsqueeze(0),pad_embed.repeat(75-col_embeddings.shape[0],1)]) # (77,1024)
    # cross init
    col_embeddings=text_encoder(inputs_embeds=input_embeds.unsqueeze(0))[0][0][1:1+n_column] # (n,1024)

    return col_embeddings # (n,1024)

@torch.no_grad()
def token_cross_init(
    tokens : str|list[str],
    tokenizer : CLIPTokenizer,
    text_encoder: CLIPTextModel,
    return_first_embeds:bool=False,
):
    if isinstance(tokens,list):
        tokens=' '.join(tokens)
    
    token_ids=tokenizer(
        tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device) # (1,k)
    if return_first_embeds:
        embeds=text_encoder.get_input_embeddings().weight.data[token_ids[0]] # (k+2,1024)
    else:
        embeds=text_encoder(token_ids)[0][0] # (k+2,1024)
    return embeds[1:-1] #(k,1024)

@torch.no_grad()
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True