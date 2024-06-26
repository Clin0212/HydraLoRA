# HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning
This repo contains the source code of [HydraLoRA](https://arxiv.org/abs/2404.19245).

## Motivations

Fine-tuning a small subset of parameters offers a streamlined approach for domain adaptation, it’s well-recognized that model performance is closely tied to the number of parameters involved.
This intrinsic characteristic of methods like LoRA often results in them falling short of the FFT baseline, which updates all parameters, thereby creating a trade-off between efficiency and model quality. 

This issue of compromised quality in a low-parameter setting becomes even more pronounced in target domains characterized by complex sub-domains and diverse tasks. This situation presents a compelling research question:

**What is the optimal architecture that can deliver superior model performance while still capitalizing on the efficiency benefits of a reduced parameter footprint?**

<figure style="text-align:center">
  <img src="./figures/Heterogeneity.png"  height="150">
</figure>

**Figure 1**: The figure demostrates erformance impact of corpus heterogeneity on full fine-tuning vs. parameter-efficient fine-tuning. Heterogeneity signifies the diversity within the dataset, often leading to intereference due to its varied content and style. Parameter-efficient approaches are particularly sensitive, suffering greater performance losses in heterogeneous cases.





<figure style="text-align:center">
  <img src="./figures/LoRA_breakdown.png" height="200">
</figure>

**Figure 2**: Breakdown analysis of LoRA modules. Consider LLaMA2-7B (random seed=42), which contains 32 decoder layers, corresponding to 32 adaptive modules. Each module consists of 0: q_proj_A, 1: q_proj_B, 2: v_proj_A, 3: v_proj_B submodules. This makes a total of 32 X 4 submodules. (a,b) left displays all submodules. (a,b) center shows all even submodules, i.e. the A matrix. (a,b) right represents all odd submodules, i.e. the B matrix. It can be seen that the differences in the fine-tuned LoRA modules for different tasks arise mainly from the B matrix.

<figure style="text-align:center">
  <img src="./figures/lora.png"  height="150">
</figure>

**Figure 3**: llustration of LoRA architecture changes in HydraLoRA. Only the tunable parameters
are shown in this Figure. (a) LoRA architecture with matrix A to achieve low rank and matrix B to recover. (b) under the same parameter count, a monolithic LoRA is splitted into multiple smaller A and B matrices to avoid training interference. (c) based on (b), HydraLoRA has an asymmetric structure that has a shared A matrix and multiple B matrices.

## Workflow of HydraLoRA
<figure style="text-align:center">
  <img src="./figures/HydraLoRA.png"  height="250">
</figure>

**Figure 4**: Architecture and workflow of HydraLoRA. During the fine-tuning stage, HydraLoRA first adaptively identifies and initializes k of intrinsic components without specific domain knowledge. It then employs a trainable MoE router that treats each intrinsic component as an expert to automatically segregate training samples
into intrinsic components for fine-tuning. During the inference stage, HydraLoRA merges multiple B matrices flexibly and dynamically through a trained router.

**For more details please check out our paper.**
##  Install

**Implementation Environment**: The model is implemented by using Pytorch. Using this command to implement your environment.

```
conda create -n hydralora python=3.10
conda activate hydralora
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```
## Project Structure
The source code is organized as below:

``` shell
|-- Motivation
    -- tesn_lora.py # analyzing the Lora modules
|-- HydraLoRA
    -- peft
    -- fine-tuning.py # main code for hydralora learning
```

## Quickstart
**1. LoRA analysis**: 

```
bash motivation/tesn_lora.sh
```


**2. HydraLoRA training**: 

```
bash HydraLoRA/fine-tuning.sh
```

## Citation
If you find our work helpful, please consider citing our paper:
```
@article{tian2024hydralora,
  title={HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning},
  author={Tian, Chunlin and Shi, Zhan and Guo, Zhijiang and Li, Li and Xu, Chengzhong},
  journal={arXiv preprint arXiv:2404.19245},
  year={2024}
}
```

## References
The code refers to the repo [LoRAMoE](https://github.com/Ablustrund/LoRAMoE), [parameter-efficient-moe
](https://github.com/for-ai/parameter-efficient-moe), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).