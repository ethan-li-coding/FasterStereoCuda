# FasterStereoCuda

# 遗留
1. 只提供了visual studio工程，如果有朋友有兴趣且有闲时，提供CMakeLists且在windows和linux平台运行通过，提merge request给我，感谢！<br>
2. 最近测试发现40系列显卡达不到理想的效率，且运行多次有效率下降的趋势，还未定位到问题，有解决了的朋友，提merge request给我，感谢! <br>

# 简介

这是一个基于CUDA加速的SGM立体匹配代码，它的核心是SemiglobalMatching（SGM）算法，它不仅在时间效率上要远远优于基于CPU的常规SGM，而且明显占用更少的内存，这意味着它不仅可以在较低分辨率（百万级）图像上达到实时的帧率，且完全具备处理千万级甚至更高量级图像的能力。

你可以拉取本测试工程并在自己的数据上体验它的改进效果，也可以在右侧下载已经打包好的压缩包，直接在本地运行控制台程序，或者在你的工程里通过动态库的方式调用它。

# 环境

Windows 10<br>
Visual Studio 2019<br>
CUDA v11.8 (如果是其他版本的cuda，你可以用文本编辑器手动打开SgmStereoCuda.vcxproj，搜索cuda字符，修改cuda后面的版本号)<br>
Opencv3.2 (下载地址：[https://download.csdn.net/download/rs_lys/13193887](https://download.csdn.net/download/rs_lys/13193887)，下载后把opencv文件夹放到3rdparty文件夹下)

# 控制台调用方式
>**单像对：**<br>
>../FasterStereoConsole.exe ../Data/Cone/left.png ../Data/Cone/right.png ../Data/Cone/option.xml<br>
>**多像对：**（KITTI）<br>
>../FasterStereoConsole.exe ../Data/KITTI/image_2 ../Data/KITTI/image_3 png ../Data/KITTI/option.xml<br> <br> 
>把../换成你的路径。option.xml是算法参数文件，在Data/文件夹中，附有两类参数文件option.xml和option2.xml，分别对应视差空间和深度空间的参数，二者用其一即可。不同的数据，需要对应修改option.xml文件的参数值。

><b>关于视差范围有特殊要求：必须满足64x2<sup>n</sup>，如64、128、256、512。</b>

# 一些案例图片
## 概览
| 数据 | Cone（450x375x64） | Kitti（1242x375x64） | Building（4800x3409x256） |
| ------ | ------ | ------ | ------ |
| 帧率 | 341.2 | 154.7 | 6.0 |
| 显存(Mb) | 258.9 | 325.3 | 4185.9 |

案例数据下载地址：[https://download.csdn.net/download/rs_lys/13074343](https://download.csdn.net/download/rs_lys/13074343)<br>
测试平台：NVIDIA GTX1080

## 数据1：Cone（450*375）
<div align=center>
<img src="https://github.com/ethan-li-coding/FasterStereoCuda-Library/blob/master/Data/diagram/Cone.png">
</div>

## 数据2：Kitti（1242*375）
<div align=center>
<img src="https://github.com/ethan-li-coding/FasterStereoCuda-Library/blob/master/Data/diagram/Kitti.png">
</div>

## 数据3：Building（4800*3409）
<div align=center>
<img src="https://github.com/ethan-li-coding/FasterStereoCuda-Library/blob/master/Data/diagram/Building.png">
</div>
