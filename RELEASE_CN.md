# MindSpore Lite Release Notes

[View English](./RELEASE.md)

## MindSpore Lite 2.8.0 Release Notes

### 主要特性及增强

- MindSpore Lite支持Python3.12。

- MindSpore Lite支持保存转换过程的中间图，可以使用环境变量控制模型转换时是否保存中间图，用于转换时问题定位。

#### 云侧推理

- LoRA权重更新性能优化，调用Model.UpdateWeights()接口性能从秒级优化至百毫秒级。

- MindSpore Lite Ascend后端ACL推理支持TimeOut配置。

- MindSpore Lite 云侧推理支持模型并发加载。

- MindSpore Lite Ascend后端GE推理支持静态shape、动态分档下数据零拷贝。

#### 端侧推理

- MindSpore Lite支持Android NPU离线模型推理。

- MindSpore Lite移除数据预处理MindData模块。

- MindSpore Lite移除Micro对Cortex-m CMSIS的支持。

### API 变更

- LoRA权重更新转换配置变更，[variable_weights_file](https://www.mindspore.cn/lite/cloud_docs/zh-CN/master/mindir/runtime_python.html#%E5%88%9B%E5%BB%BA%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)内容格式由

    ```plaintext
    weight_name:(shape);node_name
    ```

    修改为

    ```plaintext
    weight_name:shape;node_name
    ```

- [环境变量](https://www.mindspore.cn/lite/cloud_docs/zh-CN/master/reference/environment_variable_support.html)新增保存转换过程中间图功能：

    ```plaintext
    当用户配置export MSLITE_DUMP_LEVEL=0 表示Dump详细的图结构，以及常量Tensor数据；
    当用户配置export MSLITE_DUMP_LEVEL=1 表示仅Dump图结构，不dump常量Tensor数据。
    当用户配置export MSLITE_DUMP_PATH="/xx/xx/" 表示dump graph的路径。
    ```

- 移除端侧训练Train()/Evaluate()高阶接口，可通过[RunStep()](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#runstep)低阶接口替代。

- MindSpore Lite 云侧推理新增c++接口[Model.Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build-1)以及python接口[Model.build_from_buffer](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_buffer)接口，用于支持权重分离下基于buffer加载模型。

### 贡献者

YeFeng_24,xiong-pan,jjfeing,liuf9,xu_anyue,yiguangzheng,zxx_xxz,jianghui58,hbhu_bin,chenyihang5,qll1998,yangyingchun1999,liuchengji3,cheng-chao23,gemini524,yangly

## MindSpore Lite 2.7.1 Release Notes

### 主要特性及增强

- MindSpore Lite与MindSpore解耦，CPU算子库等相关动态库独立于MindSpore进行演进。

- 支持图片生成等AIGC模型中的Cache算法基于图模式实现。

### API 变更

- 新增MultiModelRunner、ModelExecutor接口，支持Cache算法的图模式实现。

    ```python
    import mindspore_lite as mslite
    import numpy as np
    dtype_map = {
        mslite.DataType.FLOAT32: np.float32,
        mslite.DataType.INT32: np.int32,
        mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.INT8: np.int8
    }
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.devcie_id = 0
    runner = mslite.MultiModelRunner()
    model_path = "path_to_model"
    runner.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    execs = runner.get_model_executor()
    for exec_ in execs:
        exec_inputs = exec_.get_inputs()
        for input_ in exec_inputs:
            data = np.random.randn(*input_.shape).astype(dtype_map[input_.dtype])
            input_.set_data_from_numpy(data)
        exec_.predict(exec_inputs)
    ```

- 离线转换工具conver_lite通过配置SplitGraph参数以及split_node_name参数实现子图切分。

    ```bash
    [SplitGraph]
    split_node_name=[[node_name_1],[node_name_2]]
    ```

### 贡献者

YeFeng_24,xiong-pan,jjfeing,liuf9,zhangzhugucheng,xu_anyue,yiguangzheng,zxx_xxz,jianghui58,hbhu_bin,chenyihang5,qll1998,yangyingchun1999,liuchengji3,cheng-chao23,gemini524,yangly,yanghui00

## MindSpore Lite 2.7.0 Release Notes

MindSpore Lite面向不同硬件设备提供轻量化AI推理加速能力，使能智能应用，为开发者提供端到端的解决方案，为算法工程师和数据科学家提供开发友好、运行高效、部署灵活的体验。

为了更好地促进人工智能软硬件应用生态繁荣发展，MindSpore Lite独立建仓促进生态发展。未来MindSpore Lite将与MindSpore AI社区一起，致力于丰富AI软硬件应用生态。

更多详情请参阅[MindSpore Lite代码仓](https://gitee.com/mindspore/mindspore-lite)。

### 主要特性及增强

- [STABLE] 支持进程间模型共享权重，以减少显存占用。用户可以通过在mindspore_lite.build_from_file接口中传入config_dict参数，并在config_dict中设置shared_mem_handle以及pids关键字使能该功能。

### API 变更

- [STABLE] mindspore_lite.Model.build_from_file接口的config_dict参数新增支持配置关键字shared_mem_handle以及pids。
- [STABLE] mindspore_lite.Model.get_model_info()接口新增支持关键字current_pid以及shareable_weight_mem_handle。

### 贡献者

YeFeng_24,xiong-pan,jjfeing,liuf9,zhangzhugucheng,xu_anyue,yiguangzheng,zxx_xxz,jianghui58,hbhu_bin,chenyihang5,qll1998,yangyingchun1999,liuchengji3,cheng-chao23,gemini524,yangly,yanghui00

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
