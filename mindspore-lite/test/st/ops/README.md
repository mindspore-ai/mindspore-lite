# mslite-op-st

mslite-op-st是MindSpore单算子测试仓。它由Python（测试框架）+ Yaml（算子测试DSL描述）构成。实现基于Yaml配置描述的ONNX单算子测试。

## 如何使用？

下载测试套源代码

```bash

# 1、进入源代码根目录
cd mslite-op-st

# 2、创建Conda环境，或者用已有环境
conda create -n mslite_opst python=3.9 # 可选3.9及以上python版本
conda activate mslite_opst

# 3、安装Pip依赖
pip install -r requirements.txt

# 4、指定MindSpore Lite发布件，并设置动态图搜索目录LD_LIBRARY_PATH。
# 来源1：MindSpore官网 https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html
# 来源2：MindSpore CI每日构建 https://repo.mindspore.cn/mindspore/mindspore/version/
export MSLITE_PACKAGE_PATH=/path/to/mindspore-lite-2.4.0rc1-linux-x64 # 此处改成实际的MindSpore绝对路径！
export LD_LIBRARY_PATH=${MSLITE_PACKAGE_PATH}/runtime/lib:${MSLITE_PACKAGE_PATH}/tools/converter/lib:${MSLITE_PACKAGE_PATH}/runtime/third_party/dnnl/:${MSLITE_PACKAGE_PATH}

# 执行conv单算子用例
cd op
python run.py -o ../output -n convfusion
#执行全部测试用例
cd op
python run.py -a -o ../output
## run脚本参数: -a 表示执行全部测试用例，-o 为输出文件路基，-n 为要测试的单算子名称，-a 与 -n 不可以同时配置
```

## yaml文件说明

1. yaml文件配置分为四个部分：genonnx、gengold、convert、run
2. genonnx：配置模型基础参数。model_name单算子模型保存名称，node_param是onnx单算子的节点设置，需要设置输入、输出以及节点的attributes。
   graph_param是onnx单算子计算图构造所需要的参数，inputs为模型输入，其中dims为输入维度，None表示动态维度，例如dims:[None,None]表示二维动态维度。outputs同inputs,多个输出则写多个输出节点，initializer节点表示算子中需要初始化的参数，name为名称，data_type为数据类型，dims为参数维度，value为可选项，定义value则参数值为给定的value值，不给定value则为随机值。
3. gengold:生成标杆数据，gold_name为保存的标杆数据文件夹名称，in_model为用于生成标杆数据的onnx模型，input_dtypes为输入向量的数据类型列表，需要和每个输入一一对应，input_shapes为输入参数的维度列表，同意需要和每个输入一一对应。
4. convert:将onnx模型转换为ms模型，out_model为输出ms模型文件名，后缀.ms可选可不选，in_model为输入的onnx模型，因为conver有三种情况：1. 动态转动态。2.动态转静态。3.静态转静态。第一种情况动态转动态，不需要设置input_shapes，所以将input_shapes置为None。第二种情况需要指定输入的静态shape大小，所以需要设定input_shapes，设置方法为input_shapes: input_name1:dim1,dim2....,dimn;input_name2:dim1,dim2...,dim3;...
   fp16为可以选项，需要将模型转化为fp16数据类型，则添加fp16:on
5. run: 对ms模型进行基准测试，in_model表示需要测试的ms模型，gold_in为集中测试需要使用标杆数据。dtype为输入数据类型，可选可不选，默认fp32
6. 各个节点测试用例可选 `disabled:on/off`，表示是否跳过该用例，不设置默认为off

## 框架Release说明

### v0.3

支持自定义initializer输入，可以手动输入shape等参数。支持fp32的onnx算子转fp16的ms算子。

convert设置fp16: on/off为可选项，默认转换设置fp16=off。

initializer的value字段不设置默认为随机初始化，设置value字段则采用手动赋值给value。

更新：run设置可以不写input_shapes选项了，现在可以从gengold阶段和genonnx阶段解析input_shapes

### v0.2

将common.py中流程进一步抽象为五个类，OnnxModelConfig、GenGoldConfig、ConvertConfig、RunConfigs和Op_TEST，实现在frame中。
修改配置模板，支持多个模型配置，输入配置、转换配置和执行配置。

目前只修改了conv算子的模板，conv算子有点问题，最后run阶段还没有通过。可以适配其他算子试一下。

运行算子benchmark:

- 安装依赖： `pip install -r requirements.txt`
- 执行对应算子测试文件

如何添加算子：

- 修改本机环境变量 `export PACKAGE_ROOT_PATH=/home/l50041265/Project/mindspore-lite-2.3.0rc2-linux-x64`
- `export MSLITE_ROOT_PATH=/home/l50041265/Project/mindspore-lite-2.3.0rc2-linux-x64 export PACKAGE_ROOT_PATH=${MSLITE_ROOT_PATH} export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/runtime/lib:${PACKAGE_ROOT_PATH}/tools/converter/lib:${PACKAGE_ROOT_PATH}/runtime/third_party/dnnl/:${LD_LIBRARY_PATH} export GLOG_v=2`
- 在model_conf.yaml中修改op_name，算子配置，生成gold数据配置，注意input_dtypes长度与input_shapes长度相同
- `cd op/op_name` 执行 `python ../run.py`
- convert中包含三种转换方式

ms模型与onnx模型转换的依赖关系以及标杆数据的依赖关系图：
![pipeline.png](./doc/images/流程.png "相对路径演示")

### v0.1

每个算子一个文件夹，abs_op.py为测试文件，model_conf.yaml中进行算子的配置，run.sh设置benchmark输入输出文件和模型信息。
具体算子配置参考https://onnx.ai/onnx/operators/index.html

运行算子benchmark:

- 安装依赖： `pip install -r requirements.txt`
- 执行对应算子测试文件

如何添加算子：

- 参考onnx文档配置，修改model_conf.yaml文件
- 修改xxx_op.py文件，调用common.py中的功能函数
- 修改run.sh文件，修改ms模型文件、输入输出文件路径，以及输入tensor名和维度信息。

## Format问题说明

对于4D张量的输入：onnx模型默认的输入格式是NCHW，而ms模型默认的输入格式为NHWC，在onnx模型转ms模型的时，convert工具会将onnx的四维NCHW输入转换为ms模型的NHWC输入，进行模型算子计算之后，插入perm节点，将模型的输出维度调整为和onnx一致的维度。例如：
![nchw2nhwc.png](./doc/images/nchw2nhwc.png "输入维度转换")

解决方法：在onnx推理时，生成的NCHW输入保存时首先进行转置，转置为NHWC保存为input1.bin，benchmark时也要对inputshape参数进行NCHW转换为NHWC。
