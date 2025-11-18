# PointMCD: Boosting Deep Point Cloud Encoders via Multi-view Cross-modal Distillation for 3D Shape Recognition

This is the official implementation of **[[PointMCD](https://arxiv.org/pdf/2207.03128.pdf)] (TMM 2023)**, which is designed for boosting deep 3D point cloud encoders by distilling discriminative cross-modal visual knowledge extracted from multi-view rendered images for a variety of 3D shape analysis and recognition applications.

<p align="center"> <img src="https://github.com/keeganhk/PointMCD/blob/master/flowchart.png" width="85%"> </p>

This code has been tested with Python 3.9, PyTorch 1.10.1, CUDA 11.1 and cuDNN 8.0.5 on Ubuntu 20.04.

### Usage
**[Datasets]** Download our [pre-processed datasets](https://pan.baidu.com/s/1uld2puuGqTxW5JbllypUxw?pwd=izsr) (password: izsr) and put them under the ```data``` folder. Or you can also use our [pre-processing code](https://pan.baidu.com/s/1hgqq3mfjZ6Ww7lmdxbhysw?pwd=293z) (password: 293z) to render multi-view images and the corresponding point-wise visibility for your own data.

**[Scripts]** We provided the training scripts of different teacher networks and released their corresponding pre-trained model parameters under the ```ckpt/teacher``` folder. The teacher knowledge information is exported in advance, which can be downloaded from [here](https://pan.baidu.com/s/1pj8ruvfFPsGQ9chQM2uwww?pwd=kzu7) (password: kzu7), and then put under the ```expt``` folder. The trained student models will be stored under the ```ckpt/student``` folder.

As a universal plug-in component for generic deep set architectures, one can easily integrate our approach to various types of deep point cloud encoders, such as [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) and [CurveNet](https://github.com/tiangexiang/CurveNet), as experimented in our paper. Your further efforts in applying PointMCD to other more powerful point cloud backbones and richer downstream task scenarios are warmly welcomed.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2023pointmcd,
	  title={PointMCD: Boosting Deep Point Cloud Encoders via Multi-view Cross-modal Distillation for 3D Shape Recognition},
	  author={Zhang, Qijian and Hou, Junhui and Qian, Yue},
	  journal={IEEE Transactions on Multimedia},
	  year={2023}
	}
