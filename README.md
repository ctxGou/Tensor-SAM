We provide the paper, poster, and code for **Unpacking the Implicit Norm Dynamics of Sharpness-Aware Minimization in Tensorized Models**.

The conda environment is specified in `src/environment.yml`. The DAS method introduced in the paper is developed as a multi-core extension of Balancing-Aware Regularization (BAR) from [Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems](https://arxiv.org/pdf/2410.14802v1). In this repository, DAS is implemented under the name `gBAR` (generalized BAR).


### Code Structure
This repository contains a comprehensive set of experiments:
- `src/td_study`: Tensor decomposition / tensor completion experiments.
- `src/MeZO`: FLoRA experiments on RoBERTa-large.
- `src/loretta`: LoRETTA experiments.
- `src/model_tensor`: Tensorized layer implementations.
- Remaining modules: experiments on (tensorized) ResNets.

## Acknowledgement

The repository is based on multiple repositories. The experiment of FLoRA is based on [FLoRA](https://github.com/Chongjie-Si/Subspace-Tuning/blob/main/loralib/loralib/layers_flora.py) and [MeZO](https://github.com/princeton-nlp/MeZO). The experiment of LoRETTA is based on [LoRETTA](https://github.com/yifanycc/loretta).

## Citation
```
@misc{cao2025unpackingimplicitnormdynamics,
      title={Unpacking the Implicit Norm Dynamics of Sharpness-Aware Minimization in Tensorized Models}, 
      author={Tianxiao Cao and Kyohei Atarashi and Hisashi Kashima},
      year={2025},
      eprint={2508.10435},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.10435}, 
}
```