## SDumont usage manual

### A user point-of-view about the LNCC SDumont supercomputer.
---

First things first, What is the LNCC's SDumont supercomputer ?

SDumont is a heterogeneous supercomputing system, composed of many nodes interconnected by an InfiniBand network. Basically, it's a heterogeneous system because their nodes are not the same, being roughly divided into (1) base nodes and (2) extension nodes. There are around 750 base nodes, whether approximately 500 nodes are B710 CPU-only nodes; about 200 are B715 nodes containing 2x NVIDIA K40 GPUs of 12GB of main memory, and, lastly, 50 B710 nodes containing (deactivated) Xeon Phi technology. The extension nodes refer to the BullSequanaX product line, with approximately 300 X1120 CPU-Only nodes and 100 X1120 nodes containing 4x NVIDIA V100 GPUS of 32GB.

Other specs:
1. Base nodes (B700):
    - 2 x CPU Intel Xeon E5-2695v2 Ivy Bridge, 2.4GHz;
    - 12 cores per CPU;
    - 64GB DDR3 of main memory;
    - 2x NVIDIA K40 of 12GB when available (B715).

2. Extension nodes (BullSequana X)
    - 2x Intel Xeon Cascade Lake Gold 6252, 2.1GHz;
    - 24 cores per CPU;
    - 384GB of main memory;
    - 4x NVIDIA V100 of 32GB when available

More information about SDumont is available in the [SDumont manual](https://sdumont.lncc.br/support_manual.php?pg=support#1.1).

---

### Contents

My access to the SDumont system was granted through the [IDeepS project](https://github.com/vsantjr/IDeepS), managed by Prof. Dr. Valdivino Santiago Jr. from INPE. This project focuses on Deep Learning models applied to Remote Sensing problems. Given the project scope and my own research interest, I'm closer to the deep learning edge. For this reason most discussed topics and tutorials available here are related to the use of GPU nodes usage, DL-related software, and environmental setting.

1. **Environment Setting** related problems:
    - **Legacy GPUs usage:** Unfortunately K40 GPUs are not supported by newer Pytorch versions. [Nelson Liu](https://blog.nelsonliu.me/2020/10/13/newer-pytorch-binaries-for-older-gpus/) has a great and complete repository of pytorch buildings that can be easily installed through the `pip install` command
    - **Building Pytorch from Source:** [IDeepS suggested](https://github.com/vsantjr/IDeepS/blob/master/Mark/usermanualdl.md#sdumont-user-manual-deep-learning) a way of installing Pytorch on SDummont for K40 GPUs usage.
    - **Singularity Containers:** SDumont comes with some different Containerization software. In this matter, I wrote a small [Singularity](singularity.md) tutorial for using available images from the NGC NVIDIA catalog.

2. **SLURM Job submission software** related problems:
    - Interactive bash
    - Base nodes
    - *Sequana_gpu_shared* queue
    - silly_conf **DOES NOT WORK shared queues**

    **(This part is yet to be done, but I have some useful contents)**