---
layout: post
current: post
cover: assets/images/spacenet-6.jpg
navigation: True
title: How to get started on Spacenet Challenge 6 using CosmiQ Baseline
date: 2020-04-20 10:18:00
tags: tutorials remote-sensing
class: post-template
subclass: 'post tag-tutorials'
author: matheus
---

Let's implement the SpaceNet Baseline to join the Spacenet Challenge 6 and extract buildings footprints from satellite imagery.

## **Introduction**

[Spacenet Challagence 6](https://spacenet.ai/sn6-challenge/) has been launched almost a month ago and has been an amazing journey applying and understanding how to use Deep Learning in satellite images. I’ve been working with Machine/Deep Learning algorithms for remote sensing for a couple of months now and I’m amazed how deep and rich are the challenges one can find in this field.

In this task, Spacenet challenges us to extract building footprints from satellite imagery using Synthetic Aperture Radar (SAR) and electro-optical (EO) imagery datasets. The area of interest (AOI) is Rotterdam, Netherlands, over 120 sq km and 48k building footprints labels. The high-resolution images are provided by Maxar’s WorldView 2 satellite. Although the participants could use both datasets (SAR and EO) for training, the test and scoring must be done using only the SAR dataset. This is intended to simulate real-world applications where one can not find matching data from both in the same location.

<p><img src="https://miro.medium.com/max/663/0*vKaGlrZJP8YUVNyT.png" alt="Test Image" /></p>

Example of EO and SAR image present in the dataset, with its inferred footprints and actual ground truth. Source: [The SpaceNet 6 Baseline](https://medium.com/the-downlinq/the-spacenet-6-baseline-3b8ae8068351)

In this tutorial, we are going to see how to implement the SpaceNet Baseline, as a first step to join the Spacenet Challenge 6. Implementing a baseline to enter a competition is not mandatory, but can help you in many ways. It can help you make sense of the dataset, have a benchmark to compare to, be a starting point or simply have a better feeling what your model’s output is supposed to look like.

You can learn more about the Baseline [here](https://medium.com/the-downlinq/the-spacenet-6-baseline-3b8ae8068351). In this baseline, CosmiQ implement a U-Net with a VGG-11 encoder. It is build using [Solaris](https://github.com/CosmiQ/solaris), which is a CosmiQ Works Geospatial Machine Learning analysis toolkit based in Python, and can achieve a score of 0.21±.02 using the Jaccard Index, also called the Intersection-over-Union (IoU). You can check the full code here: [https://github.com/CosmiQ/CosmiQ_SN6_Baseline](https://github.com/CosmiQ/CosmiQ_SN6_Baseline).



## **Walkthrough**



### **Prerequisites**

Although I think this tutorial can be used for any platform, at least to have a general overview, I’m currently working a Ubuntu Linux 18.04. So some of the commands and steps below are taking this into account.

The CosmiQ Works provides the baseline using Docker, and not in a notebook to run on Jupyter or Colab, although you could try to “translate” it, I guess. Using docker is the best way to share the baseline to a vast number of people using different systems and keeping up all the libraries and dependencies together. So the first thing is to make sure you have the latest version of Docker installed. Although it would work with other versions, the latest version (Docker 19.03) natively supports NVIDIA GPUs as devices. Furthermore, we can use NVIDIA Container Toolkit to help us even more, since with this we don’t need to install the CUDA toolkit at the host. Allowing us to have multiple projects with multiple CUDA versions in your machine. You just need to install the NVIDIA driver at your host machine and the specific CUDA Toolkit inside your container.

<p><img src="https://miro.medium.com/max/522/0*n3TB9NY4_3Xn7IOU.png" alt="Test Image" /></p>

NVIDIA Container Toolkit. Source: [nvidia-docker repository](https://github.com/NVIDIA/nvidia-docker)

So go ahead and make sure you have installed in your system:

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

- NVIDIA driver for your system

- Docker 19.03

- Git

  

### **The Datasets**

The training data contains processed tiles of 450x450 m with its corresponding buildings footprints labels. The whole folder has about 39 GB. On the other hand, the test set has 17GB and each tile covers the same area. Both can be downloaded inside the container or in the host system. Either way, you are going to need an AWS account.

Apparently, everyone with a normal Amazon account is already able to access AWS and get the credentials to download the images. If you don’t, you are going to need to create one and add a credit card. Once you have in hands your **[Access key ID** and **Secret access key**](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html), you need to install the AWS CLI. This can be easily done in Ubuntu using pip or conda. I prefer using conda due to its intuitive virtual environments and features.

**Using pip:**

`~$ sudo pip install awscli`

**Using conda:**

`~$ conda install -c conda-forge awscli`

Now you need to configure the awscli using the credentials I mentioned before. So type this and fill with your keys:

`~$ aws configure`

Awesome, now let’s download the images.

**Training Data (~39GB)**

`~$ aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz .`

**Test Data (~17GB)**

`~$ aws s3 cp s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz .`

If you followed these steps inside the container, you are good to go. Otherwise, if you downloaded the files in your host system (which I think is better), you can transfer this data to your running container using:

`~$ docker cp file_to_transfer.tar.gz ContainerID:/root/`

This will copy the dataset you choose to the same location where all the files of this project are.

Furthermore, if you want to know more, have some questions on how to obtain the data or are using Windows, you can check [this guide](https://docs.google.com/document/d/1mkBKtSpeYlH3PxGTcLvzaXEarElUQCb1IThFclIXG0k/edit) to download these datasets using AWS.



## **Running the baseline inside the container**



Now it is time to spin up this docker container and start using the baseline. Let’s go to the directory where we want to work on this project and clone the [baseline repository](https://github.com/CosmiQ/CosmiQ_SN6_Baseline).

`~$ git clone [https://github.com/CosmiQ/CosmiQ_SN6_Baseline.git](https://github.com/CosmiQ/CosmiQ_SN6_Baseline.git)`

In this folder, you will find several files that compose this baseline. The more important are:

- Weights — A folder containing the best weights trained by CosmiQ.
- baseline.py — Where all the magic happens. All the functions we need are here.
- model.py — Holds the model, as it states. A U-Net and a VGG-11 written in Pytorch.
- settings.sh, train.sh and test.sh — 3 shell scripts to configure, train and test the model.
- Dockerfile — contains all the commands to assembly the Docker image.

We then create the Docker image with the following command inside this folder:

`~$ docker build --tag baseline .`

and then run it

`~$ sudo docker run -ti --name sb6 --gpus all baseline /bin/bash`

We are now inside the container, read to work in an environment prepared to develop our solution or work with the baseline to improve it. To run the baseline model in the test set and check the results you can simply run the test.sh script with two arguments: location of the test examples and where to save the .csv, which would be read to submission.

`~$ ./test.sh /root/test_public/AOI_11_Rotterdam/ baseline_output.csv`

This command will search for the images in the root directory in the same structure from the .tar.gz file, and save the predictions .csv also in the root folder with the same “baseline_output”.

*Note: Unfortunately, my GPU is old (an NVIDIA GeForce 740M) and since version 0.3.1, Pytorch does not work with it because of Cuda capability. I then start a container without GPU to test it, it took 15 seconds for each inference, taking 8.5 hours to finish the 2000 test examples with an Intel(R) Core(TM) i5–3337U CPU @ 1.80GHz*.



## **Conclusion**



You are now free to modify the files, make changes to the model, choose another encoder or try to improve this baseline.

This tutorial was a walkthrough on how to join the Spacenet Challenge 6 using the baseline provided by CosmiQ. There are some minor details, like dependencies and drives, that slowed me down when I tried to implement this and I hope this tutorial helped you in some manner.

Happy coding and I wish you the best of luck with this challenge!