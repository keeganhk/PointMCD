# PointMCD: Boosting Deep Point Cloud Encoders via Multi-view Cross-modal Distillation for 3D Shape Recognition

This is the official implementation of **[[PointMCD](https://arxiv.org/pdf/2207.03128.pdf)] (TMM 2023)**, which is designed for boosting deep 3D point cloud encoders by distilling discriminative cross-modal visual knowledge extracted from multi-view rendered images for a variety of 3D shape analysis and recognition applications.

<p align="center"> <img src="https://github.com/keeganhk/PointMCD/blob/master/flowchart.png" width="85%"> </p>

This code has been tested with Python 3.9, PyTorch 1.10.1, CUDA 11.1 and cuDNN 8.0.5 on Ubuntu 20.04.

### Usage
**[Datasets]** Download our [pre-processed datasets](https://drive.google.com/drive/folders/1y0Wuhb2GJ9fXGLXMKAyyyQ3Ja4nPUAbG?usp=sharing) and put them under the ```data``` folder. Or you can also use our [pre-processing code](https://drive.google.com/drive/folders/19pA1NkMfwvoonpX7NpO7XXZ7japK8Ry3?usp=sharing) (in MATLAB) to render multi-view images and the corresponding point-wise visibility for your own data.

**[Scripts]** We provide the training scripts of different teacher models (with images as inputs) and release the corresponding pre-trained teacher network parameters under the ```ckpt/teacher``` folder. All the required teacher knowledge information has been exported in advance, which can be downloaded from [here](https://drive.google.com/drive/folders/1ntuRIr8cZ4NY-c36HI18Rf0FP-gun2UO?usp=sharing), and put under the ```expt``` folder. The trained student models will be stored under the ```ckpt/student``` folder.

As a universal plug-in component for generic deep set architectures, one can easily integrate our approach to various types of deep point cloud encoders, such as [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) and [CurveNet](https://github.com/tiangexiang/CurveNet), as experimented in our paper. Your further efforts in applying PointMCD to other more powerful point cloud backbones and richer downstream task scenarios are warmly welcomed.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2023pointmcd,
	  title={PointMCD: Boosting Deep Point Cloud Encoders via Multi-view Cross-modal Distillation for 3D Shape Recognition},
	  author={Zhang, Qijian and Hou, Junhui and Qian, Yue},
	  journal={IEEE Transactions on Multimedia},
	  year={2023}
	}
