# Learning to simulate complex scenes for street scene segmentation
This repository contains the code, engine and the generated data for our paper published in IEEE Trans. on Multimedia.<br>
The paper is entitled '[Learning to simulate complex scenes for street scene segmentation](https://ieeexplore.ieee.org/document/9366432)',<br>
or you can find the arxiv pre-print version '[Learning to simulate complex scenes](https://arxiv.org/abs/2006.14611v1).'<br>

# Requirements
Unity version: 2019.2.10f1<br>
MLAgents version: 0.10.0<br>
Other Python packages.

# SceneX Engine
SceneX engine contains 106 buildings, 200 pedestrians, 195 cars, 28 buses and 39 trucks, etc.<br>
To run the code on Linux OS, please:<br>
  [1] Download the [Assets.zip](https://pan.baidu.com/s/13kHsbz6MhzYwpFJood1QpQ)(pin: 0903, via BaiduDisk) and unpack it into directory "SceneX/".<br>
  [2] Download the [engine.zip](https://pan.baidu.com/s/1XCukK4S6FArDXHzcf-uErQ)(pin: 0903, via BaiduDisk) and unpack it into directory "SceneX/".<br>

# Usage
For attribute training: <br>
Edit file 'train_ssd2city_sdr.py' as follow:<br>
[1] NUM_INPUTS = 7 (Which means to optimize 7 attributes once) [line39]<br>
[2] MAIN_LR = 1e-2 (Learning rate = 0.01) [line41]<br>
[3] MAX_FRAMES = 40 (Maximal iterations = 40) [line42]<br>
[4] building_x_delta = attribute_list[0] (line146)<br>
[5] fence_x_delta = attribute_list[1] (line147)<br>
[6] tree_x_delta = attribute_list[2] (line148)<br>
[7] motorcycle_x_delta = attribute_list[3] (line149)<br>
[8] person_x_delta = attribute_list[4] (line150)<br>
[9] hang_x_delta = attribute_list[5] (line151)<br>
[10] car_x_delta = attribute_list[6] (line152)<br>
And run the script, then the final output may be like this: [0.8, 0.1, 0.0, 0.8, 0.0, 0.0, 0.9] <br>

For attribute testing: <br>
Edit file 'test_city.py' as follow:<br>
[1] building_x_delta = 0.8 (line118)<br>
[2] fence_x_delta = 0.1 (line119)<br>
[3] tree_x_delta = 0.0 (line120)<br>
[4] motorcycle_x_delta = 0.8 (line121)<br>
[5] person_x_delta = 0.0 (line122)<br>
[6] hang_x_delta = 0.0 (line123)<br>
[7] car_x_delta = 0.9 (line124)<br>
and run the script. Then the final output may be 329.24 , which means the mIoU is 18.1% for SceneX to Cityscapes cross validation.<br>

The detailed attribute optimization order can be found in the paper.

# SceneX to Cityscapes Adapted Images
Can be downloaded via the following link: [ssd2city](https://pan.baidu.com/s/1km9mC6RThTWb3QJTcKHg8Q)(pin: 0903, via BaiduDisk).

If you find this code useful for your research, please kindly cite our paper.<br>

```
@article{xue2020learning,
  title={Learning to simulate complex scenes for street scene segmentation},
  author={Zhenfeng Xue, Weijie Mao, Liang Zheng},
  booktitle={IEEE Transactions on Multimedia},
  year={2021},
  doi={10.1109/TMM.2021.3062497}
}
````
