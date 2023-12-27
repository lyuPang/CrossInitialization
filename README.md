# Cross Initialization

**Official Implementation of "Cross Initialization for Personalized Text-to-Image Generation".**

We will release the code soon.

## Abstract
Recently, there has been a surge in face personalization techniques, benefiting from the advanced capabilities of pretrained text-to-image diffusion models. Among these, a notable method is Textual Inversion, which generates personalized images by inverting given images into textual embeddings. However, methods based on Textual Inversion still struggle with balancing the trade-off between reconstruction quality and editability. In this study, we examine this issue through the lens of initialization. Upon closely examining traditional initialization methods, we identified a significant disparity between the initial and learned embeddings in terms of both scale and orientation. The scale of the learned embedding can be up to 100 times greater than that of the initial embedding. Such a significant change in the embedding could increase the risk of overfitting, thereby compromising the editability. Driven by this observation, we introduce a novel initialization method, termed Cross Initialization, that significantly narrows the gap between the initial and learned embeddings. This method not only improves both reconstruction and editability but also reduces the optimization steps from 5,000 to 320. Furthermore, we apply a regularization term to keep the learned embedding close to the initial embedding. We show that when combined with Cross Initialization, this regularization term can effectively improve editability. We provide comprehensive empirical evidence to demonstrate the superior performance of our method compared to the baseline methods. Notably, in our experiments, Cross Initialization is the only method that successfully edits an individual's facial expression. Additionally, a fast version of our method allows for capturing an input image in roughly 26 seconds, while surpassing the baseline methods in terms of both reconstruction and editability.


## Results
<img src='assets/teaser.png'>

<img src='assets/CI.jpg'>

## Results of Our Fast Version
The following results are obtained after 25 optimization steps, each image taking 26 seconds on an A800 GPU.

<img src='assets/fast1.jpg'>

<img src='assets/fast2.jpg'>

## References

```
@article{pang2023crossinitialization,
  title = {Cross Initialization for Personalized Text-to-Image Generation},
  author = {Pang, Lianyu and Yin, Jian and Xie, Haoran and Wang, Qiping and Li, Qing and Mao, Xudong},
  journal = {arXiv preprint arXiv:2312.15905},
  year = {2023}
}
