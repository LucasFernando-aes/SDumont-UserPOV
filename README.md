It's important to note that this tutorial is related to the LNCC's supercomputer SDumont and the [IDeeps project](https://github.com/vsantjr/IDeepS). Even though it can be easily extended to other clusters environments with Slurm job scheduler, you should pay attention for SDumont specific terms, such as, queues and directory names. Moreover, I would like to sorry in advance for some typos and wrongly english writting that can appear during the tutorial.

In summary, SDumont have support for container softwares, like either with Docker or Singularity platforms. With them the user can take advantaged of the large NVIDIA NGC Catalog of container images by promptly using pre-build images with complete NVIDIA ecosystems for PyTorch or Tensorflow. Once the images are pulled, **just submit a job and run your code**, avoiding any problem with the error-prone ```module load``` step.

# Containerization on SDumont

Ok, we known that there are a lot of ways to use SDumont HPC, so why do I need to know another one, particularly, why should I use containers ?

Going directly to the point, for us (IDeepS project) and also the deep learning community, [NVIDIA NGC](https://catalog.ngc.nvidia.com/) catalog has a lot of ready to use docker containers images, for example, images that already have PyTorch or Tensorflow packages alongside many different other NVIDIA libraries carefully built to take advantage of their devices, like NVIDIA DALI, that improves the data throughput, and APEX, that enables training models with mixed precision.

Moreover, I ~~(Lucas Silva)~~ found that manually installing the NVIDIA side-libraries in the cluster environment is difficult (to not say boring), generally because it requires to load different types of modules, usually in specific versions, and be compiled in nodes with available GPUs (requiring iteractive job allocation and, consequentially, the unnecessary use of UA). Many different works in deep leaning are leveraging the capabilities of these NVIDIA new libraries, and to be able to fairly reproduce their results is important to, at least in the first instance, also correctly reproduce their project environment requirements. 

Nonetheless, reproducibility of deep learning projects is a well known issue in nowadays works (see Bucci *et. al.* [7] for a deeper discussion), so a fast way of delivering your work to other people or to simply execute your work in other computational environment without all the trouble of replicating the previous configuration can be easily done with containers.

## Singularity

From the [Sylabs website](https://sylabs.io/guides/3.5/user-guide/introduction.html), Singularity is a container platform that aims to run complex applications on HPC clusters in a simple, portable and reproducible way. Of course, there are other container platforms softwares that can be used, for example, Docker, one of the most known container platform with a mature and big community, an Image hub and many different side softwares that can help in orchestrating the communication and execution of containers. 

### **So, why singularity ?**

Actually, the two main reason of why to use Singularity in SDumont is that **it’s better suited for HPC environments** than Docker, and, also because it is the only container platform directly available to all standard users of SDumont. According to [Sdumont FAQ](https://sdumont.lncc.br/support_FAQ.php?pg=support#9) there is also a Docker in SDumont, but in order to use it the user must be part of a specific user group.

In any case, Singularity has some features that improves the research side of development, for instance, its images are portable files that can be easily shared among computers, it provides an easy way of exploiting GPUs devices (just add the option `—-nv` to the execution command), and also provides a security execution of containers by keeping the outside user privileges inside the container.

### How to use it ?

---

To feel how singularity works, let's see the **Default use case.**

Just pull the desired image, in this case it will download a pre-built image from singularity library (`library:`), but it also have access of external external resources, like docker hub (`docker:`) or local images (`.sif` files).

`$singularity pull library://godlovedc/funny/lolcow`

You can execute it as a executable file

`$./lolcow_latest.sif`

but it is the same of using the `run` command.

`$singularity run lolcow_latest.sif`

You can also iteratively use your new singularity container by 

`$singularity shell lolcow_latest.sif`

Where, depending on your OS, probably your prompt will change indicating that now you are inside the container. Once you are inside the container, you can run, for example, `cat /etc/os-release` to see what linux distribution the container are using or, any other command.

For more general use cases, please, visit [[2]](https://singularity-tutorial.github.io/02-basic-usage/)

---

Now, let's go through a more special use case: **NVIDIA containers in SDumont.**

Since Singularity is a container platform that isn't largely adopted by the community, in order to attain more users it has the interestingly feature of converting docker images to the singularity file format (i.e. `.sif` files). This feature can also be applied directly to images from docker hub or other container catalogs that is associated with docker ecosystem, e.g., [NVIDIA NGC container catalog](https://catalog.ngc.nvidia.com/containers).

Firstly, in order to be able to download NVIDIA containers from NGC, the very first step is to make an account in [NGC catalog](https://catalog.ngc.nvidia.com/containers) and generate an [NGC API Key](https://docs.nvidia.com/ngc/ngc-private-registry-user-guide/index.html#generating-api-key).

Once you get the API key, you will need to registry your API key on Singularity (inside Sdumont) by exporting two environment variables (Note that the username must be the same as the example *‘$oauthtoken’*).

```bash
$export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
$export SINGULARITY_DOCKER_PASSWORD=<NGC API key>
```

Now that you have set the credentials, you must be able to pull any image from the catalog.

Particularly, let's try pulling a PyTorch image. From the initial page of NGC catalog, find the PyTorch box and then go to tags tab.

**NGC Catalog → PyTorch → Tags**

Inside tags you will see different releases of the Image. Each of the releases refers to different software version, for example, the last release (in the time of writing) **22.03-py3** is composed of Ubuntu distribution 20.04, CUDA 11.6.1 and PyTorch 1.12.0. To use this container, simply pull it with the following command:

`$singularity pull pytorch22.03.sif docker://nvcr.io/nvidia/pytorch:22.03-py3`

Submit a job to the SDumont

`$salloc --partition=sequana_gpu_dev —-nodes=1 —-job-name singularity_test`

And then, inside the allocated node, run

`$singularity shell —-nv pytorch22.03.sif`

Now, since you are iteratively into the container (that is running inside the allocated node) you can test the GPUs visibility with `nvidia-smi` and, if you want, test with some deep learning script that uses GPU.

> 🤔 **What softwares and software's version come with each release tags ?** Specifically for PyTorch images, you can find it in the [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html).

> 🛠 **Mounting directories from the main file system to the container.**
use -B or —bind command option: *a user-bind path specification which receives a format string src[:dest[:opts]], where src and dest are outside and inside paths. If dest is not given, it is set equal to src. Mount options ('opts') may be specified as 'ro' (read-only) or 'rw' (read/write, which is the default). Multiple bind paths can be given by a comma separated list.* 
>
> Example:
> `$singularity shell -B /scratch/ideeps/<user>:my_scratch pytorch22.03.sif`

> 🏃🏻‍♀️ **Running with `sbatch`** To run python scripts inside singularity images by submitting the job with sbatch command, you can use the same slurm running script described in [IDeepS Github repository](https://github.com/vsantjr/IDeepS/blob/master/Mark/multigpus.md), but remember that you don't need to load neither conda nor any of the modules, and also change the execution line to `singularity run` command. Example:
> ```bash
> #!/bin/bash
> #SBATCH --job-name singularity_test     # SLURM_JOB_NAME
> #SBATCH --partition sequana_gpu_dev     # SLURM_JOB_PARTITION
> #SBATCH --nodes=1                       # SLURM_JOB_NUM_NODES
> #SBATCH --ntasks-per-node=1             # SLURM_NTASKS_PER_NODE
> #SBATCH --cpus-per-task=10              # SLURM_CPUS_PER_TASK
> #SBATCH --no-requeue                    # No automated job resubmission
> #SBATCH --time=00:20:00                 # Define execution time
>
> singularity run -B /scratch/ideeps/<user>:my_scratch pytorch20.03.sif sh -c 'cd my_scratch/my_repo && python cnn_gpu.py --epochs 5 --batch_size 16'
> ```

> ❌ **Singularity *SIGKILL* error during pulling an image :** If at the end of the pulling image command a SIGKILL error occur, you must pull the image inside an allocated job node, such as, inside a job on cpu_dev queue. for example, after the job allocation with `$salloc --partition=cpu_dev —-nodes=1 —-job-name singularity_pulling`, in a iterative session, run the singularity pulling command `$singularity pull pytorch22.03.sif docker://nvcr.io/nvidia/pytorch:22.03-py3`

### Compatibility with Legacy GPUs

---

The most important drawback of the NVIDIA available images of PyTorch is that the default PyTorch binaries installed in the images isn't compatible with the legacy K40 GPUs largely available in SDumont.

Because of this, I ~~(Lucas Silva)~~ built an image for the legacy GPUs K40. Particularly, the image is based in the tag 20.03-py3 of NGC PyTorch catalog and Legacy-PyTorch binaries from [Nelson Liu](https://blog.nelsonliu.me/2020/10/13/newer-pytorch-binaries-for-older-gpus/), which is composed of Python 3.6, CUDA 10.2, PyTorch 1.6.0, Torchvision 0.7.0 and all the other side-libraries from NVIDIA (NVIDIA DALI, Apex and so on). The image is publicly available at [Docker Hub [6]](https://hub.docker.com/repository/docker/lucasfernandoaes/pytorch-legacy/general) and can easily be pulled by the following command:

`singularity pull pytorch-k40.sif docker://lucasfernandoaes/pytorch-legacy:1.6.0`

You can use the same commands above to use this image, the only difference is that you can run you scripts in the partitions `nvidia_small`, `nvidia_dev`, `nvidia_long`.

Since in that time I only needed the version 1.6.0 of PyTorch, this was the only image available at my docker hub, but as long as new versions of PyTorch are being required, I'll adding them on demand in form of new tags.

## Aknowledgements

I would like to thanks the IDeepS project and LNCC/MCTI for the access and use of SDumont supercomputer.

## Useful links:

1. [https://sylabs.io/guides/3.5/user-guide/index.html](https://sylabs.io/guides/3.5/user-guide/index.html)
2. [https://singularity-tutorial.github.io/02-basic-usage/](https://singularity-tutorial.github.io/02-basic-usage/)
3. [https://catalog.ngc.nvidia.com/containers](https://catalog.ngc.nvidia.com)
4. [https://developer.nvidia.com/blog/docker-compatibility-singularity-hpc/](https://developer.nvidia.com/blog/docker-compatibility-singularity-hpc/)
5. [https://developer.nvidia.com/blog/how-to-run-ngc-deep-learning-containers-with-singularity/](https://developer.nvidia.com/blog/how-to-run-ngc-deep-learning-containers-with-singularity/)
6. [https://hub.docker.com/repository/docker/lucasfernandoaes/pytorch-legacy/general](https://hub.docker.com/repository/docker/lucasfernandoaes/pytorch-legacy/general)
7. [https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610409.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610409.pdf)