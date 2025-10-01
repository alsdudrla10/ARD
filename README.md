## [CVPR 2025 Oral] Autoregressive Distillation of Diffusion Transformers <br><sub><sub> This repository provides a re-implementation of the original work (ARD), reconstructed from the author's recollection. </sub></sub>
**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space), [Sotiris Anagnostidis](https://sanagnos.pages.dev/), [Yuming Du](https://dulucas.github.io/), [Edgar Schoenfeld](https://edgarschnfld.github.io/), [Jonas Kohler](https://scholar.google.de/citations?user=a1rCLUMAAAAJ&hl=de), [Markos Georgopoulos](https://scholar.google.com/citations?user=id7vw0UAAAAJ&hl=en), [Albert Pumarola](https://www.albertpumarola.com/), [Ali Thabet](https://www.alithabet.com/), [Artsiom Sanakoyeu](https://gdude.de/)**  


[![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.11295-b31b1b.svg)](https://arxiv.org/abs/2504.11295)&nbsp;

## Overview
<i>We propose AutoRegressive Distillation (ARD), a method that leverages the historical trajectory of diffusion ODEs to mitigate exposure bias and improve efficiency in distillation. ARD achieves strong performance on ImageNet and text-to-image synthesis with significantly fewer steps and minimal computational overhead.</i>
![Teaser image](./assets/figure1.JPG)

## Dependencies
The requirements for this code are the same as [DiT](https://github.com/facebookresearch/DiT).

## Save ODE trajectories
Make sure to save a sufficient number of ODE trajectories using `sample_trajectory.py`, and ensure they match the dataloader used in the subsequent training procedure (see the `--data-path` argument in the training script).

## Training
This procedure is performed using 8 A100 GPUs for 2 days.

```
torchrun --nnodes=1 --nproc_per_node=8 train_ARD.py --model DiT-XL/2 --global-batch-size=64 --stack=6
```

## Fine-tuning with GAN loss
This procedure is performed using 8 A100 GPUs over a few hours.
Since this procedure also uses real data, it needs to properly set the path to the actual ImageNet dataset using the --real-data-path argument.

```
torchrun --nnodes=1 --nproc_per_node=8 train_ARD_gan.py --model DiT-XL/2 --global-batch-size=48 --stack=6 --ckpt_path={$PATH}/checkpoints/0300000.pt"
```

## Generation
```
torchrun --nnodes=1 --nproc_per_node=1 sample_ARD.py --stack=6 --ckpt_path={$PATH}/checkpoints/0300000.pt"
```

## Performance
We follow the evaluation protocol of [ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations).
![Teaser image](./assets/figure2.JPG)
| Model | Steps | Latency | FID | Inception Score | Precision | Recall | 
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| DiT/XL-2 | 25 | 493.5 | 2.89 | 230.22 | 0.797 | 0.572 |
| Step Distillation (N=0) | 4 | 64.80 | 10.25 | 181.58 | 0.704 | 0.474 |
| ARD (N=6) | 4 | 66.34 | 4.32 | 209.03 | 0.770 | 0.574 |
| + GAN loss finetuning | 4 | 66.34 | 1.84 | 235.84 | 0.797 | 0.615 |

![Teaser image](./assets/figure3.JPG)

## Citation
If you find the code useful for your research, please consider citing
```bib
@inproceedings{kim2025autoregressive,
  title={Autoregressive Distillation of Diffusion Transformers},
  author={Kim, Yeongmin and Anagnostidis, Sotiris and Du, Yuming and Sch{\"o}nfeld, Edgar and Kohler, Jonas and Georgopoulos, Markos and Pumarola, Albert and Thabet, Ali and Sanakoyeu, Artsiom},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={15745--15756},
  year={2025}
}
```
