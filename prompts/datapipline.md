
## 本项目是数据驱动的方法，因此着重强调 对于数据的处理。

# 数据获取： 
- 现在的数据的meta path 在 /home/gbsguest/Research/boson/BIO/ecgvec/disk1  这个路径定义在 config 文件里
数据预处理的逻辑（主要是这些内容读 WFDB，500Hz → 100Hz 重采样；0.67–40Hz 带通滤波；导联顺序规范化） 可以参考 /home/gbsguest/Research/boson/BIO/ecggen/prompts/refs/mimic_preprocess.py，
这里整体的数据加载逻辑是 对于原始数据做一个 加载层，然后暴露到 Dataset 层就都是一样了。 目前先全都动态加载，动态划分 train test 


# 数据probe；

对于获取的原始数据，在 /home/gbsguest/Research/boson/BIO/ecggen/probe/data这里要写一个 jupyter文件可视化，可以指定index 可视化获取的原始的ECG 信号
