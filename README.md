# Cross Initialization (CVPR 2024)

Official Implementation of **"Cross Initialization for Face Personalization of Text-to-Image Models"** by Lianyu Pang, Jian Yin, Haoran Xie, Qiping Wang, Qing Li, Xudong Mao

## Abstract
> Recently, there has been a surge in face personalization techniques, benefiting from the advanced capabilities of pretrained text-to-image diffusion models. Among these, a notable method is Textual Inversion, which generates personalized images by inverting given images into textual embeddings. However, methods based on Textual Inversion still struggle with balancing the trade-off between reconstruction quality and editability. In this study, we examine this issue through the lens of initialization. Upon closely examining traditional initialization methods, we identified a significant disparity between the initial and learned embeddings in terms of both scale and orientation. The scale of the learned embedding can be up to 100 times greater than that of the initial embedding. Such a significant change in the embedding could increase the risk of overfitting, thereby compromising the editability. Driven by this observation, we introduce a novel initialization method, termed Cross Initialization, that significantly narrows the gap between the initial and learned embeddings. This method not only improves both reconstruction and editability but also reduces the optimization steps from 5,000 to 320. Furthermore, we apply a regularization term to keep the learned embedding close to the initial embedding. We show that when combined with Cross Initialization, this regularization term can effectively improve editability. We provide comprehensive empirical evidence to demonstrate the superior performance of our method compared to the baseline methods. Notably, in our experiments, Cross Initialization is the only method that successfully edits an individual's facial expression. Additionally, a fast version of our method allows for capturing an input image in roughly 26 seconds, while surpassing the baseline methods in terms of both reconstruction and editability.

<img src='assets/teaser.png'>
<a href="https://arxiv.org/abs/2312.15905"><img src="https://img.shields.io/badge/arXiv-2312.15905-b31b1b.svg" height=20.5></a>

## Update
+ **2024.3.2**: Code released!

## Setup
Our code mainly bases on [Diffusers-Textual Inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) and relies on the [diffusers](https://github.com/huggingface/diffusers) library.

To set up the environment, please run:
```
conda create -n ci python=3.10
conda activate ci

pip install -r requirements.txt
```

## Dataset
### Image Dataset
We use CelebA dataset to test our method, which can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

We also provide some images for testing in `./examples/input_images`, which are also from CelebA.

### Celeb Name Dataset
The Celeb names in `./examples/wiki_names_v2.txt` used in this project are from [Celeb Basis](https://github.com/ygtxr1997/CelebBasis/tree/main). We are very grateful for their contributions.

## Usage

<img src='assets/CI.jpg'>

### Logging into Huggingface
To use `stabilityai/stable-diffusion-2-1-base` model, you may have to log into Huggingface as following

+ Use `huggingface-cli` to login in Terminal
+ Input your token extracted from [Token](https://huggingface.co/settings/tokens)

### Training
You can simply run the `train_cross_init.py` script and pass the parameters to train your own result.

For example, to train the identity in `./examples/input_images/28017`, you can run:
```
python train_cross_init.py \
    --save_steps 100 \
    --only_save_embeds \
    --placeholder_token "<28017>" \
    --train_batch_size 8 \
    --scale_lr \
    --n_persudo_tokens 2 \
    --reg_weight "1e-5" \
    --learning_rate 0.000625 \
    --max_train_step 320 \
    --train_data_dir "./examples/input_images/28017" \
    --celeb_path "./examples/wiki_names_v2.txt" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --output_dir "./logs/28017/learned_embeddings" 
```
Please refer to `train_cross_init.py` for more details on all parameters.

### Inference
To run inference on a learned embedding, you can run:
```
python test_cross_init.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --num_inference_steps 50 \
    --learned_embedding_path "./logs/28017/learned_embeddings/learned_embeds.bin" \
    --prompt "a photo of a {} person" \
    --save_dir "./logs/28017/images" \
    --num_images_per_prompt=8 \
    --n_iter=1
```
**Note:**

+ We provide learned embeddings in `./examples/learned_embeddings` for anyone who wants to directly experiment with our methods.
+ For convenience, you can either specify a path to a text file with  `--prompt_file`, where each line contains a prompt. For example:
    ```
    A photo of a {} person
    A {} person eating bread in front of the Eiffel Tower
    A {} person latte art
    ```
+ The identity placement should be specified using `{}`, and we will replace `{}` with the identity's placeholder token that is saved in the learned embedding checkpoint.
+ The generated images will be saved to the path `{save_dir}/{prompt}`

Please refer to `test_cross_init.py` for more details on all parameters.

## Metrics
We use the same evaluation protocol as used in [Celeb Basis](https://github.com/ygtxr1997/CelebBasis/tree/main).

## Results of Our Fast Version
The following results are obtained after 25 optimization steps, each image taking 26 seconds on an A800 GPU.

<img src='assets/fast1.jpg'>

<img src='assets/fast2.jpg'>

## Acknowledgements
Our code mainly bases on [Diffusers-Textual Inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion), and the celeb names used in this project are from [Celeb Basis](https://github.com/ygtxr1997/CelebBasis/tree/main). A huge thank you to the authors for their valuable contributions.

## References

```
@article{pang2023crossinitialization,
  title = {Cross Initialization for Face Personalization of Text-to-Image Models},
  author = {Pang, Lianyu and Yin, Jian and Xie, Haoran and Wang, Qiping and Li, Qing and Mao, Xudong},
  journal = {arXiv preprint arXiv:2312.15905},
  year = {2023}
}
