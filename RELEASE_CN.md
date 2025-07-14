# MindSpore Lite Release Notes

[View English](./RELEASE.md)

## MindSpore Lite 2.6.0 Release Notes

### 主要特性及增强

- [STABLE] MindSpore Lite支持模型转换时配置算子并行推理加速，只需在模型转换时配置stream_label_file选项，指定需要进行并行推理的算子。
- [STABLE] MindSpore Lite支持在昇腾后端下转换onnx控制流中的if算子。

### API 变更

- [STABLE] acl模型转换配置中，ascend_context选项下新增stream_label_file选项，用于启用多流并行。

### 贡献者

熊攀,ZhangZGC,yanghaoran,李林杰,shenwei41,xiaotianci,panzhihui,guozhijian,胡彬,tangmengcheng,XianglongZeng,cccc1111,stavewu,刘思铭,r1chardf1d0,jiangshanfeng

## MindSpore Lite 2.3.1 Release Notes

### 主要特性及增强

昇腾后端模型转换时，支持使用配置文件中的[input_shape 参数](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/cloud_infer/converter_tool_ascend.html)来指定输入尺寸。

### API 变更

- [ModelGroup接口](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/cloud_infer/runtime_cpp.html) 新增模型权重共享支持，节省显存。
- [Model.get_model_info接口](https://www.mindspore.cn/lite/docs/zh-CN/r2.3.1/use/converter_tool.html?highlight=get_model_info) 新增支持获取模型的输入尺寸。

### 贡献者

熊攀;ZhangZGC;jxl;zhangyanhui;emmmmtang;huandong1;yefeng

## MindSpore Lite 2.3.0-rc2 Release Notes

### 主要特性和增强

- [STABLE] 支持云侧转换工具所用的配置文件配置FlashAttention相关属性。
- [STABLE] 支持在多张卡上进行内存共享。

### 贡献者

感谢以下人员做出的贡献:

emmmmtang,熊攀

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.11 Release Notes

### 问题修复

- [#I8TPLY] 修复 SSD MobileNetV2 FPN 网络在Atlas 推理系列产品平台上的推理失败问题。

### 贡献者

感谢以下人员做出的贡献:

wangtongyu6, zhuguodong, 徐永飞, 徐安越, yeyunpeng2020, moran, XinDu, gengdongjie.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.10 Release Notes

### 问题修复

- [#I8K7CC]优化get_model_info接口传入非str字段的报错

### 贡献者

感谢以下人员做出的贡献:

gengdongjie, zhangyanhui, xiaoxiongzhu, wangshaocong, jianghui58, moran, wangtongyu6, 徐安越, qinzheng, 徐永飞, youshu, XinDu, yeyunpeng2020, yefeng, wangpingan, zjun, 胡安东, 刘力力, 陈宇, chenjianping, kairui_kou, zhangdanyang, hangq, mengyuanli, 刘崇鸣

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.1 Release Notes

### Bug Fixes

- [#I88055] 修复MindSpore Lite推理gridsample算子format设置错误的问题。
- [#I8D80Y] 修复MindSpore Lite推理单算子调用流程资源释放异常的问题。

### 贡献者

感谢以下人员做出的贡献:

zhanghaibo, wangsiyuan, yefeng, wangshaocong, chenjianping

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.2.0 Release Notes

### 主要特性和增强

#### 支持FlashAttention算子融合

- [STABLE] 在Ascend系列硬件上，支持LLAMA、stable diffusion系列模型的FlashAttention大算子融合。

## MindSpore Lite 2.1.1 Release Notes

### Major Features and Improvements

- [STABLE] MindSpore Lite Cloud Inference adds support for Python 3.8 and Python 3.9

## MindSpore Lite 2.1.0 Release Notes

### 主要特性和增强

#### MindSpore Lite云侧推理

- [STABLE] 支持Ascend硬件后端单卡大模型以及单机多卡分布式大模型高性能推理。
- [STABLE] Python API Ascend后端支持多模型共享工作空间（Workspace）内存。
- [STABLE] [通过ModelGroup新增支持多模型共享权重](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/cloud_infer/runtime_cpp.html#%E5%A4%9A%E6%A8%A1%E5%9E%8B%E5%85%B1%E4%BA%AB%E6%9D%83%E9%87%8D)，比如大模型场景下全量模型和增量模型共享权重。

#### API

新增ModelGroup [Python](https://www.mindspore.cn/lite/api/zh-CN/r2.1/mindspore_lite/mindspore_lite.ModelGroup.html#mindspore_lite.ModelGroup)和[C++](https://mindspore.cn/lite/api/zh-CN/r2.1/api_cpp/mindspore.html#modelgroup)接口，接口定义如下：

```python
class ModelGroup
    def __init__(self, flags=ModelGroupFlag.SHARE_WORKSPACE)
    def add_model(self, models)
    def cal_max_size_of_workspace(self, model_type, context)
```

```C++
// class ModelGroup
ModelGroup(ModelGroupFlag flags = ModelGroupFlag::kShareWorkspace);
Status AddModel(const std::vector<std::string> &model_path_list);
Status AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list);
Status AddModel(const std::vector &model_list);
Status AddModel(const std::vector &model_list);
```

## MindSpore Lite 2.0.0-rc1 Release Notes

### 主要特性和增强

#### MindSpore Lite云侧推理

原MindSpore Lite版本主要面向手机、车机等边缘设备，新增云侧推理版本支持云侧多后端硬件资源的场景，支持Ascend及Nvidia GPU推理专用卡，高效利用云侧多核资源。

原通过MindSpore训练版本集成的推理方式可以变更为基于MindSpore Lite进行适配集成，具体可参考[云侧推理快速入门](https://mindspore.cn/lite/docs/zh-CN/r2.0/quick_start/one_hour_introduction_cloud.html)，如果想要保持原始集成方式可以参考[MindSpore推理FAQ](https://mindspore.cn/docs/zh-CN/r2.0/faq/inference.html)。

- [STABLE] 支持MindIR模型文件。
- [STABLE] 支持将第三方Onnx、Tensorflow、Caffe模型通过MindSpore Lite转换工具转换为MindIR模型文件。
- [STABLE] 一个发布包支持多种硬件后端：Ascend、Nvidia GPU、CPU。
- [STABLE] 支持`Model`接口和`ModelParallelRunner`并行推理接口。
- [STABLE] 支持C++、Python和Java推理接口。

#### API

- 因原Python API配置参数较多、使用较复杂，因此在2.0版本针对Python API易用性进行优化，包括类构造方法、类属性的调整等，此外2.0及之后的Python API将整合到云侧推理场景，与旧版本不兼容。详细参见[Python API说明文档](https://www.mindspore.cn/lite/api/zh-CN/r2.0/mindspore_lite.html)。

## MindSpore Lite 1.10.0 Release Notes

### Bug fixes

- 修复Arithmetic类CPU算子动态shape场景下可能的计算精度问题。
- 修复Deconv int8量化算子重量化写入地址错误问题。

## MindSpore Lite 1.8.0 Release Notes

### 主要特性和增强

#### API

- [STABLE] 新增模型转换的C++和Python API.
- [STABLE] 新增模型推理的Python API.

#### 后量化

- [STABLE] 后量化支持PerLayer量化，同时内置CLE算法优化精度。

## MindSpore Lite 1.7.0 Release Notes

### 主要特性和增强

#### 后量化

- [STABLE] 后量化支持动态量化算法。
- [BETA] 后量化模型支持在英伟达GPU上执行推理。
