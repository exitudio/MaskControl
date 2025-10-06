# MaskControl: Spatio-Temporal Control for Masked Motion Synthesis (ICCV 2025 -- Oral)
## [Previous Name] ControlMM: Controllable Masked Motion Generation
### [[Project Page]](https://www.ekkasit.com/ControlMM-page/) [[Paper]](https://arxiv.org/abs/2410.10780)
![teaser_image](https://www.ekkasit.com/ControlMM-page/assets/landing.png)

If you find our code or paper helpful, please consider starring our repository and citing:
```
@inproceedings{pinyoanuntapong2025maskcontrol,
  title={MaskControl: Spatio-Temporal Control for Masked Motion Synthesis},
  author={Pinyoanuntapong, Ekkasit and Saleem, Muhammad Usama and Karunratanakul, Korrawe and Wang, Pu and Xue, Hongfei and Chen, Chen and Guo, Chuan and Cao, Junli and Ren, Jian and Tulyakov, Sergey},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## ✅ TODO List

### 🧪 Evaluation
- [x] Joint Control (GMD, OmniControl, and MMM Evaluation)
- [ ] ProMoGen Evaluation
- [ ] STMC Evaluation

### 🎯 Generation
- [x] Joint Control
- [x] Obstacle Avoidance
- [ ] Body Part Timeline Control

### 🏋️ Training
- [ ] Retrain MoMask with Cross Entropy for All Positions
- [ ] Add Logits Regularizer



## :round_pushpin: Getting Started

Our code built on top of [MoMask](https://github.com/EricGuo5513/momask-codes/tree/main). If you encounter any issues, please refer to the MoMask repository for setup and troubleshooting instructions.
<details>
  
### 1. Conda Environment
```
conda env create -f environment.yml
conda activate ControlMM
pip install git+https://github.com/openai/CLIP.git
```


#### Alternative: Pip Installation
<details>

```
pip install -r requirements.txt
```

</details>

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```


You have two options here:
* **Skip getting data**, if you just want to generate motions using *own* descriptions.
* **Get full data**, if you want to *re-train* and *evaluate* the model.

**(a). Full data (text + motion)**

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```
#### 

</details>





## :book: Evaluation on joint control:
<details>

#### ▶️ Pelvis Only (GMD Evaluation)
```
python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --dataset_name t2m \
    --ctrl_name 'z2024-08-23-01-27-51_CtrlNet_randCond1-196_l1.1XEnt.9TTT__fixRandCond' \
    --gpu_id 0 \
    --ext 0_each100Last600CtrnNet \
    --control trajectory \
    --density -1 \
    --each_iter 100 \
    --last_iter 600 \
    --ctrl_net T

```

#### ▶️ All Joints (OminControl and MMM Evaluation)
```
python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --dataset_name t2m \
    --ctrl_name 'z2024-08-27-21-07-55_CtrlNet_randCond1-196_l1.5XEnt.5TTT__cross' \
    --gpu_id 4 \
    --ext 0_each100_last600_ctrlNetT \
    --control cross \
    --density -1 \
    --each_iter 100 \
    --last_iter 600 \
    --ctrl_net T
```


#### 🎮 Control Joints
The following joints can be controlled:
```
[pelvis, left_foot, right_foot, head, left_wrist, right_wrist]
```


---

#### 🚀 Arguments

| Argument      | Description                                                                                                                                                                                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--res_name`  | Name of the residual transformer                                                                                                                                                                                                                                                                        |
| `--ctrl_name` | Name of the control transformer (VQ and Masked Transformer are also saved in this)                                                                                                                                                                                                                                              |
| `--gpu_id`    | GPU ID to use                                                                                                                                                                                                                                                                                           |
| `--ext`       | Log name used for saving results, stored in: `checkpoints/t2m/{ctrl_name}/eval/{ext}`                                                                                                                                                                                                                   |
| `--control`   | Type of random joint control:<br>• `trajectory` – pelvis only<br>• `random` – uniform random joints<br>• `cross` – random combinations, see section \[A.11 CROSS COMBINATION]<br>• Any single joint: `pelvis`, `l_foot`, `r_foot`, `head`, `left_wrist`, `right_wrist`, `lower`<br>• `all` – all joints |
| `--density`   | Number of control frames:<br>• `1`, `2`, `5` – exact number of control frames<br>• `49` – 25% of ground truth length<br>• `196` – 100% of ground truth length<br>(If GT length < 196, 49/196 are converted proportionally)                                                                              |
| `--each_iter` | Number of logits optimization iterations at **each unmask step**                                                                                                                                                                                                                                        |
| `--last_iter` | Number of logits optimization iterations at the **last unmask step**                                                                                                                                                                                                                                    |
| `--ctrl_net`  | Enable ControlNet with Logits Regularizer: `T` or `F`                                                                                                                                                                                                                                                   |


</details>

## 🎯 Generation
<details>

#### 🚀 Joints Control
```
python -m generation.control_joint --path_name ./output/control2 --iter_each 100 --iter_last 600
```
| Argument      | Type | Default         | Description                                                                |
| ------------- | ---- | --------------- | -------------------------------------------------------------------------- |
| `--path_name` | str  | `./output/test` | Output directory to save the optimization results.                         |
| `--iter_each` | int  | `100`           | Number of logits optimization steps at each unmasking step.                |
| `--iter_last` | int  | `600`           | Number of logits optimization steps at the final unmasking step.           |
| `--show`      | flag | `False`         | If set, automatically opens the result HTML visualization after execution. |

#### 🚀 Obstrucle Avoidance
Example 1 -- Trajectory avoidance
<!-- ![teaser_image](./assets/traj_avoid.gif) -->
<img src="./assets/traj_avoid.gif" width="300">

```
python -m generation.avoidance --path_name ./output/avoidance1 --iter_each 100 --iter_last 600
```

Example 2 -- Head avoidance
<!-- ![teaser_image](./assets/head_avoid.gif) -->
<img src="./assets/head_avoid.gif" width="300">

```
python -m generation.avoidance2 --path_name ./output/avoidance2 --iter_each 100 --iter_last 600
```

</details>

## License

This code is distributed under [LICENSE-CC-BY-NC-ND-4.0](https://github.com/exitudio/MMM/blob/main/LICENSE-CC-BY-NC-ND-4.0.md).

Note that our code depends on other libraries, including 
[MoMask](https://github.com/EricGuo5513/momask-codes/tree/main), [OmniControl](https://neu-vi.github.io/omnicontrol/), [GMD](https://github.com/korrawe/guided-motion-diffusion), [MMM](https://github.com/exitudio/MMM), [TLControl](https://github.com/HiWilliamWWL/TLControl), [STMC](https://github.com/nv-tlabs/stmc), [ProgMoGen](https://github.com/HanchaoLiu/ProgMoGen), [TEMOS](https://github.com/Mathux/TEMOS) and [BAMM](https://github.com/exitudio/BAMM/) which each have their own respective licenses that must also be followed.
