# MindSpore Lite Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Lite 2.7.0 Release Notes

### Major Features and Improvements

- [STABLE] MindSpore Lite supports configuring operator parallel inference acceleration during model conversion. You only need to configure the stream_label_file option during model conversion to specify the operators that need parallel inference.
- [STABLE] MindSpore Lite supports the conversion of onnx if operators in the Ascend backend.

### API Change

- [STABLE] In the acl model conversion configuration, a new stream_label_file option is added under the ascend_context option to enable multi-stream parallel inference.

### Contributors

熊攀,ZhangZGC,yanghaoran,李林杰,shenwei41,xiaotianci,panzhihui,guozhijian,胡彬,tangmengcheng,XianglongZeng,cccc1111,stavewu,刘思铭,r1chardf1d0,jiangshanfeng

## MindSpore Lite 2.3.1 Release Notes

### Major Features and Improvements

When converting Ascend backend models, the [input_shape](https://www.mindspore.cn/lite/docs/en/r2.3.1/use/cloud_infer/converter_tool_ascend.html) parameter in the configuration file is supported to specify the input size.

### API Change

- [ModelGroup](https://www.mindspore.cn/lite/docs/en/r2.3.1/use/cloud_infer/runtime_cpp.html) interface adds model weight sharing support to save video memory.

- [Model.get_model_info](https://www.mindspore.cn/lite/docs/en/r2.3.1/use/converter_tool.html?highlight=get_model_info) interface adds support for obtaining the input size of the model.

### Contributors

熊攀;ZhangZGC;jxl;zhangyanhui;emmmmtang;huandong1;yefeng

## MindSpore Lite 2.3.0-rc2 Release Notes

### Major Features and Improvements

- [STABLE] Support the configuration of FlashAttention related properties in the configuration file used by the cloud-side conversion tool.
- [STABLE] Support multi-devices memory sharing.

### Contributors

Thanks goes to these wonderful people:

emmmmtang,熊攀

Contributions of any kind are welcome!

## MindSpore Lite 2.2.11 Release Notes

### Bug Fixes

- [#I8TPLY] Fixed SSD MobileNetV2 FPN network inference error on Atlas inference series products.

### Contributors

Thanks goes to these wonderful people:

wangtongyu6, zhuguodong, 徐永飞, 徐安越, yeyunpeng2020, moran, XinDu, gengdongjie.

Contributions of any kind are welcome!

## MindSpore Lite 2.2.10 Release Notes

### Bug Fixes

- [#I8K7CC] Optimize error message when non-string segments are passed to get_model_info.

### Contributors

Thanks goes to these wonderful people:

gengdongjie, zhangyanhui, xiaoxiongzhu, wangshaocong, jianghui58, moran, wangtongyu6, 徐安越, qinzheng, 徐永飞, youshu, XinDu, yeyunpeng2020, yefeng, wangpingan, zjun, 胡安东, 刘力力, 陈宇, chenjianping, kairui_kou, zhangdanyang, hangq, mengyuanli, 刘崇鸣

Contributions of any kind are welcome!

## MindSpore Lite 2.2.1 Release Notes

### Bug Fixes

- [#I88055] Fixed a function issue caused by incorrect format setting of the gridsample operator in MindSpore Lite inference.
- [#I8D80Y] The MindSpore Lite inference single-operator invoking process resources are not released and exits abnormally.

### Contributors

Thanks goes to these wonderful people:

zhanghaibo, wangsiyuan, wangshaocong, chenjianping

Contributions of any kind are welcome!

## MindSpore Lite 2.2.0 Release Notes

### Major Features and Improvements

#### FlashAttention Operator Fusion

- [STABLE] The OptiX OSN Ascend series supports the FlashAttention large operator fusion of the LLAMA and stable diffusion models.

## MindSpore Lite 2.1.1 Release Notes

### Major Features and Improvements

- [STABLE] MindSpore Lite Cloud Inference adds support for Python 3.8 and Python 3.9

## MindSpore Lite 2.1.0 Release Notes

### Major Features and Improvements

#### MindSpore Lite Cloud Inference

- [STABLE] Supports high-performance inference for single-device large model and single-node multi-device distributed large model at Ascend backend.
- [STABLE] Python API Ascend backend supports multiple models sharing workspace memory.
- [STABLE] [The weights can be shared by multiple models through ModelGroup](https://mindspore.cn/lite/docs/en/r2.1/use/cloud_infer/runtime_cpp.html#multiple-models-sharing-weights). For example, weights can be shared between full models and incremental models in the large model scenario.

#### API

The [Python](https://www.mindspore.cn/lite/api/en/r2.1/mindspore_lite/mindspore_lite.ModelGroup.html) and [C++](https://mindspore.cn/lite/api/en/r2.1/generate/classmindspore_ModelGroup.html) ModelGroup interface is added. The interface definition is as follows:

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

### Major Features and Improvements

#### MindSpore Lite Cloud Inference

The original MindSpore Lite is mainly used for edge devices such as mobile phones and head units. Cloud inference is added to support scenarios with multiple backend hardware resources on the cloud, supports Ascend and NVIDIA GPU inference cards, and efficiently utilizes multi-core resources on the cloud.

The original cloud inference integrated through MindSpore training can be changed to MindSpore Lite. For details, see [Quick Start to Cloud-side Inference](https://mindspore.cn/lite/docs/en/r2.0/quick_start/one_hour_introduction_cloud.html). To retain the original integration method, see [Inference](https://mindspore.cn/docs/en/r2.0/faq/inference.html).

- [STABLE] Support MindIR model files.
- [STABLE] Third-party Onnx, TensorFlow, and Caffe models can be converted to MindIR model files using the MindSpore Lite conversion tool.
- [STABLE] One release package supports multiple hardware backends: Ascend, NVIDIA GPU, CPU.
- [STABLE] Supports the `Model` interface and `ModelParallelRunner` concurrent inference interface.
- [STABLE] Supports C++, Python, and Java inference interfaces.

#### API

- Due to the defects of the original Python API that many configuration parameters and complex usage, the usability of The Python APIs are optimized in version 2.0. The optimizations include class construction methods and class attribute adjustment. In addition, the Python APIs in version 2.0 and later will be integrated into the cloud-side inference scenario, which are incompatible with Python APIs of the earlier versions. For details, see [Python API](https://www.mindspore.cn/lite/api/en/r2.0/mindspore_lite.html).

## MindSpore Lite 1.10.0 Release Notes

### Bug fixes

- Fixed potential accuracy problem of arithmetic type CPU kernels at dynamical shape case.
- Fixed the Incorrect Write Address of the Deconv Quantization Operator.

## MindSpore Lite 1.8.0 Release Notes

### Major Features and Improvements

#### API

- [STABLE] Add C++ and Python APIs for model conversion.
- [STABLE] Add Python APIs for model inference.

#### Post-Training Quantization

- [STABLE] Support perlayer quantization, and built-in CLE to optimize perlayer quantization accuracy.

## MindSpore Lite 1.7.0 Release Notes

### Major Features and Improvements

#### Post quantization

- [STABLE] Support post quantization to run dynamic quantization algorithm.
- [BETA] Support post quantized model to run on NVIDIA GPU.

## MindSpore Lite 1.6.0

### Major Features and Improvements

#### Converter and runtime

- [STABLE] Add more fusion patterns in the converter tool to improve runtime performance.
- [STABLE] Support take OpenGL texture as input and output of inference.
- [STABLE] Refactor the JAVA API.
- [BETA] Support inference on Ascend310.

#### x86 backend optimization

- [STABLE] Optimize kernels for x86 using Advanced Vector Extensions(AVX512).

#### ARM backend optimization

- [STABLE] Support heterogeneous parallel inference, including splitting operators, constructing heterogeneous subgraphs, and heterogeneous parallel scheduling between CPUs and GPUs.
- [STABLE] Add more FP16 operators.

#### Post quantization

- [STABLE] Post quantization supports debugging.
- [STABLE] Full quantization supports choosing non-quantized nodes.
- [STABLE] Mixed bit quantization supports auto-tune.

#### Training on Device

- [STABLE] Support user-defined algorithm models to access the federated learning framework.

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, [wangnan39@huawei.com](mailto:wangnan39@huawei.com), wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, [zhanghaibo5@huawei.com](mailto:zhanghaibo5@huawei.com), zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

## MindSpore Lite 1.5.0

### Major Features and Improvements

#### Converter and runtime

1. Optimize TDNN-like streaming model by reusing the result of last inference.
2. Support dynamic filter Convolution.
3. Support serializing float32 weight into float16 weight for reducing size of model file.
4. Provide unified runtime API for developer reusing their code between cloud side and end side.
5. Now developer can configure built-in pass as custom passes.
6. Now user can specify format and shape of model inputs while converting model.
7. Support multiple devices inference, includeing CPU, NPU, GPU. User can set devices in mindspore::Context.
8. Support mixed precision inference. User can set inference precision by LoadConfig API.
9. Support custom operator registration and enable inference on third-party hardware.

#### ARM backend optimization

1. Support the nchw data format of some Operators, such as Conv, InstanceNorm, etc. The performance of some models convertered from onnx and caffe is greatly improved.
2. Fix bugs of memory leak on NPU.

#### Post quantization

1. Weight quantization supports mixed bit quantization.
2. Full quantization supports data pre-processing.
3. Adjust the quantization parameters from the command line to the configuration file.

#### Training on Device

1. Unify lite external api with MindSpore.
2. Implement static memory allocator and common workspace for TOD，save memory 10-20%.
3. Provide getgradients and setgradients interface，get and set optimizer params interfaces to support MOE Model.
4. Support user specified output node when export IOD Model.
5. Support more text  networks (tinybert,albert) and operators.

#### Codegen

1. Support kernel register for custom op. Third-party hardware like NNIE can be accessed through it.

### API Change

#### API Incompatible Change

##### C++ API

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

## MindSpore Lite 1.3.0

### Major Features and Improvements

#### Converter and runtime

1. Support Caffe model running on Hi3516D.
2. Support delegate mechanism to run your models(part or whole) on user specified executor.
3. Support control flow models.
4. Support cross-compiling for iOS, so that we can inference models on iOS devices.

#### x86 backend optimization

1. Optimize kernels for x86 using Advanced Vector Extensions(AVX).

#### ARM backend optimization

1. Optimize fp16 kernels.
2. Support arm32 fp16 instruction acceleration on ARMv8.2.

#### Cuda backend optimization

1. Support NV GPU backend base on delegate mechanism(use TensorRT as delegate).

#### OpenCL backend

1. Optimize the strategy of workgroup and blocksize to improve performance.
2. Support OpenCL dynamic infershape.
3. Support INT32 type ops.

#### Post quantization

1. Support fp32 training model converts to quantization training model.

#### Training on Device

1. Support fp32 training model export to quantization model after training process end.
2. Unify APIs and output package name of training and inference.
3. Simplify implementation of Train Session.
4. Optimize train and infer compile, reduce libmindspore-lite-train.so memory.
5. Training memory optimization:  memory reduce 10-50% compare with  r1.2.
6. Training performance optimization:  for 1*1 special input shape Cov2DGradInput and SparseSoftmaxCrossEntropyWithLogits operator optimization, improved 10%-20%.
7. Support more networks(transformer, albert).

#### Codegen

1. Support deployment on HarmonyOS for device.

### API Change

#### API Incompatible Change

##### C++ API

###### Unify LiteSession and TrainSession, Merge LiteSession And TrainSession.([!17356](https://gitee.com/mindspore/mindspore/pulls/17356))

Previously, Training on Device use TrainSession while Inference on Device use LiteSession. To simplify implementation, we move TrainSession functions to LiteSession as virtual function. and move APIs previous defined in train_session.h to lite_session.h.

```cpp
class MS_API LiteSession {
...
static LiteSession *CreateTrainSession(const std::string &filename, const lite::Context *context,
                                         bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
 static LiteSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head,
                                            const lite::Context *context, bool train_mode = false,
                                            const lite::TrainCfg *cfg = nullptr);
virtual int Train() { return mindspore::lite::RET_ERROR; }
virtual int Eval() { return mindspore::lite::RET_OK; }
virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) {
    return mindspore::lite::RET_ERROR;
  }
virtual std::vector<tensor::MSTensor *> GetPredictions() const {
    std::vector<tensor::MSTensor *> outputs;
    return outputs;
 }
...
```

###### Add Export API for Training on device, obsolete SaveToFile API.([!17356](https://gitee.com/mindspore/mindspore/pulls/17356))

Previously, Training on Device uses SaveToFile API to save the training model to file. Export API was added in this release to support more format, more model type(train or interface part of the model), and save weight quant model of train.

```cpp
virtual int Export(const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType = lite::FT_FLATBUFFERS) {
    return mindspore::lite::RET_ERROR;
 }
```

###### Add GetFeatureMaps and UpdateFeatureMaps interface for Training on device.([!18344](https://gitee.com/mindspore/mindspore/pulls/18344))

When Training on the device, we may need to update the model featuremap and get model featuremap.particularly in MindSpore Federated Scenario.

```cpp
virtual std::vector<tensor::MSTensor *> GetFeatureMaps() const {
    std::vector<tensor::MSTensor *> features;
    return features;
  }
  virtual int UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features) { return mindspore::lite::RET_ERROR; }
```

#### New features

##### Java API

###### new static method for creating LiteSession by MSConifg in LiteSession.class

Previously, if we want to create a LiteSession object, we need to call two APIs:

```js
MSConfig config;
// config options ...
LiteSession liteSession = new LiteSession();
boolean ret = liteSession.init(config);
if (!ret) {
  // handle init LiteSession failed ...
}
```

now we can create a LiteSession object with new API just like:

```js
MSConfig config;
// config options ...
LiteSession liteSession = createSession(config);
if (liteSession == null) {
  // handle create LiteSession failed ...
}
```

###### new static method for creating LiteSession byModelBuffer and MSConfig in LiteSession.class

Previously, if we want to inference a model, we need to call APIs like:

```js
MSConfig config;
// config options ...
LiteSession liteSession = new LiteSession();
boolean initSessionRet = liteSession.init(config);
if (!initSessionRet) {
  // handle init LiteSession failed and return ...
}
Model model = new Model();
boolean loadModelRet = model.loadModel(modelMappedByteBuffer);
if (!loadModelRet) {
  // handle load model failed and return ...
}
boolean compileModelRet = liteSession.compileGraph(model);
if (!loadModelRet) {
  // handle compile model failed and return ...
}
model.free();
// liteSession is ready to inference model, call runGraph in LiteSession.class ...
```

now we can use new API just like:

```js
MSConfig config;
// config options ...
LiteSession liteSession = createSession(modelMappedByteBuffer, config);
if (liteSession == null) {
  // handle init LiteSession failed and return ...
}
// liteSession is ready to inference model, call runGraph in LiteSession.class ...
```

New createSession method is an API that integrates four old APIs: LiteSession.init, Model.loadModel, LiteSession.compileGraph and model.free. It is simple and efficient as it reduces one modelBuffer copy operation.

###### new methods getFeaturesMap and updateFeatures for in LiteSession.class

Recently, we add a new C++ api in LiteSession class, Correspondingly we add a new java API in LiteSession.java.

```java
public List<MSTensor> getFeaturesMap() {
         List<Long> ret = this.getFeaturesMap(this.sessionPtr);
                ArrayList<MSTensor> tensors = new ArrayList<MSTensor>();
                for (Long msTensorAddr : ret) {
                    MSTensor msTensor = new MSTensor(msTensorAddr);
                    tensors.add(msTensor);
                }
                return tensors;
   }
   public boolean updateFeatures(List<MSTensor> features) {
            long[] inputsArray = new long[features.size()];
            for (int i = 0; i < features.size(); i++) {
                inputsArray[i] = features.get(i).getMSTensorPtr();
            }
             return this.updateFeatures(this.sessionPtr, inputsArray);
   }
```

###### new methods export to replace saveToFile API in LiteSession.class

Recently, we add a new C++ api in LiteSession class, Correspondingly we add a new java API in LiteSession.java.

```java
public boolean export(String modelFileName, int modelType, int quantizationType) {
        return this.export(this.sessionPtr, modelFileName, modelType, quantizationType);
    }
```

###### new train related  API moved to LiteSession.class from TrainSession.class

Align with update of C++ api in LiteSession class, add new java API to LiteSession.java Correspondingly.

```java
public class LiteSession {
...
public static LiteSession createTrainSession(String modelName, final MSConfig config, boolean trainMode){...}
public boolean train() {...}
public boolean eval() {...}
...
```

### Bug fixes

1. Fix the bug that the train session does not release memory cause of refcount bug.

#### Deprecations

### Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, Zhenglong Li, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

Contributions of any kind are welcome!

# MindSpore 1.2.1

## MindSpore 1.2.1 Release Notes

### Major Features and Improvements

#### FrontEnd

- [STABLE] Add MaskedSelect aicpu operation.(Ascend)

#### Auto Parallel

- [STABLE] Support distributed checkpoint loading.(Ascend/GPU)

## MindSpore Lite 1.2.0

### Major Features and Improvements

#### Converter and runtime

1. Support TensorFlow model in Converter except aware-training model.
2. Add fusion pattern for same horizontal operators in Converter.
3. Support Jar in x86_64 system for integrating into server with Java backend conveniently.
4. Provide unified runtime API for developer reusing their code between cloud side and end side.[BETA]
5. Improve control-flow capabilities continually: Support GRU fusion in Converter; Support weight-quant for control-flow model; Support control-flow model inference with half precision; Support nested control-flow model.[BETA]

#### ARM backend optimization

1. Add NLP dependent float16 operators(like lstm) to enhance inference performance.
2. Optimize operators: lstm, gru, depthwise.
3. Add 6 NPU operators(like FullConnection), and fix some bugs about buildIR failed.

#### OpenCL backend

1. Add new ops: add 10+ ops，total 72 ops；
2. Performance optimization: by memory layout optimize，block tiling，Performance improved by 30% compared to version 1.1 at Adreno GPU.
3. Initialization time optimization: initialization time improve 100% vs MSLITE Version1.1 by store kernel cache as binary.
4. Support Java call on Mali or Adreno GPU.

#### Post quantization

1. Support quantization of gather and lstm ops.
2. Support quantizatizing TF Lite models with sub-graph node.
3. Add quantiztion strategy to decide quantize ops or not，less accuracy loss and higher compression rate.

#### Training on Device

1. Virtual batching, use mini-batch to minic large batch in theorical with few RAM consumption.
2. Converter unify, do not compile tod and iod converter separately.
3. Performance optimization to BWD ops.
4. TrainLoop with Off-The-Shelf Functionality blocks, like LR scheduler, Loss Monitor, Ckpt Saver, Accuracy Monitor.
5. Integration of code with Minddata lite.
6. Support more networks (googlenet, densenet, shufflenetv2, nin, vgg) and operators.

#### Codegen

1. Support 79 ops for the ARM platform and all CMSIS ops for Arm Cortex-M Series.
2. Multiplatform support, including Android, IoT Devices.
3. Support offline model weight preprocessing while compiling.
4. Support offline memory reuse computing for minimum runtime buffer size.
5. Support kernel register for custom op. Third-party hardware like NNIE can be accessed through it.

### API Change

#### API Incompatible Change

##### C++ API

###### Add header file named lite_types.h for some common data structs. ([!12262](https://gitee.com/mindspore/mindspore/pulls/12262))

Previously, some common data structs such as `CpuBindMode` and `DeviceType` are in context.h, this may cause cross-dependency between headers. So we create a new header named lite_types.h for some common data structs and move `CpuBindMode` and `DeviceType` from context.h into lite_types.h.

<table>
<tr>
<td style="text-align:center"> lite_types.h </td>
</tr>
<tr>
<td>

```cpp
namespace mindspore::lite {
/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND,    /**< no bind */
  HIGHER_CPU, /**< bind higher cpu first */
  MID_CPU     /**< bind middle cpu first */
} CpuBindMode;

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type */
} DeviceType;
}  // namespace mindspore::lite
```

</td>
</tr>
</table>

###### Add some new interfaces in ms_tensor.h for unified runtime API.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

Previously, users could not create `MSTensor` or modify ``MSTensor, all `MSTensor` are created and managed by framework. However users need to create or modify MSTensor sometimes such as pre-processing input data. So we provide two new interfaces in ms_tensor.h: `CreateTensor` interface for creating `MSTensor` by user and `set_shape` interface for modifying the shape of `MSTensor`.

<table>
<tr>
<td style="text-align:center"> CreateTensor </td>
</tr>
<tr>
<td>

```cpp
/// \brief Create a MSTensor.
///
/// \return Pointer to an instance of MindSpore Lite MSTensor.
static MSTensor *CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape, const void *data,
                                size_t data_len);
```

</td>
</tr>
</table>

<table>
<tr>
<td style="text-align:center"> set_shape </td>
</tr>
<tr>
<td>

```cpp
/// \brief Set the shape of MSTensor.
virtual void set_shape(const std::vector<int> &shape) = 0;
```

</td>
</tr>
</table>

Previously, users could access to data of `MSTensor` by interface named `MutableData`. However `MutableData` is not only returning data of tensor but also allocating data for tensor if its data is nullptr. So we provide a new interfaces in ms_tensor.h named `data` for returning data of tensor without allocating automatically.

<table>
<tr>
<td style="text-align:center"> data </td>
</tr>
<tr>
<td>

```cpp
/// \brief Get the pointer of data in MSTensor.
///
/// \note The data pointer can be used to both write and read data in MSTensor. No memory buffer will be
/// allocated.
///
/// \return the pointer points to data in MSTensor.
virtual void *data() = 0;
```

</td>
</tr>
</table>

###### Delete `DimensionSize()` in ms_tensor.h.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

The interface named `DimensionSize` is fuinctionally overlapped with the interface named `shape`. For the simplicity of the interface, we delete `DimensionSize` and recommend users to use the new interface named `shape` instead.

<table>
<tr>
<td style="text-align:center"> DimensionSize() </td>
</tr>
<tr>
<td>

```cpp
/// \brief Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.
///
/// \param[in] index Define index of dimension returned.
///
/// \return Size of dimension of the MindSpore Lite MSTensor.
virtual int DimensionSize(size_t index) const = 0;
```

</td>
</tr>
</table>

###### Move allocator from namespace mindspore::lite to namespace lite for unified runtime API.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

Previously, class `Allocator` is in namespace mindspore::lite. Considering unified allocator interface for unified runtime API, we move `Allocator` to namespace mindspore.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
namespace mindspore::lite {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
}
```

</td>
<td>

```cpp
namespace mindspore {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
}
```

</td>
</tr>
</table>

### Bug fixes

1. Fix the bug that the array in kernel registrar is not initialized.
2. Fix segment fault caused by releasing of OpParameter in Crop kernel in mistake.
3. Fix the bug that the MINDIR aware-training model is finally interpreted as weight-quant model.

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa.

Contributions of any kind are welcome!

# MindSpore 1.1.1 Release Notes

## MindSpore

### Major Features and Improvements

#### NewModels

- [STABLE] BGCF: a Bayesian Graph Collaborative Filtering(BGCF) framework used to model the uncertainty in the user-item interaction graph and thus recommend accurate and diverse items on Amazon recommendation dataset.(Ascend)
- [STABLE] GRU: a recurrent neural network architecture like the LSTM(Long-Short Term Memory) on Multi30K dataset.(Ascend)
- [STABLE] FastText: a simple and efficient text classification algorithm on AG's news topic classification dataset, DBPedia Ontology classification dataset and Yelp Review Polarity dataset.(Ascend)
- [STABLE] LSTM: a recurrent neural network architecture used to learn word vectors for sentiment analysis on aclImdb_v1 dataset.(Ascend)
- [STABLE] SimplePoseNet: a convolution-based neural network for the task of human pose estimation and tracking on COCO2017 dataset.(Ascend)

#### FrontEnd

- [BETA] Support Tensor Fancy Index Getitem with tuple and list. (Ascend/GPU/CPU)

### Backwards Incompatible Change

#### Python API

##### `ops.AvgPool`, `ops.MaxPool`, `ops.MaxPoolWithArgmax` change attr name from 'ksize', 'padding' to 'kernel_size', 'pad_mode' ([!11350](https://gitee.com/mindspore/mindspore/pulls/11350))

Previously the kernel size and pad mode attrs of pooling ops are named "ksize" and "padding", which is a little puzzling and inconsistent with convolution ops. So they are rename to "kernel_size" and "pad_mode".

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> avg_pool = ops.AvgPool(ksize=2, padding='same')
>>> max_pool = ops.MaxPool(ksize=2, padding='same')
>>> max_pool_with_argmax = ops.MaxPoolWithArgmax(ksize=2, padding='same')
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> avg_pool = ops.AvgPool(kernel_size=2, pad_mode='same')
>>> max_pool = ops.MaxPool(kernel_size=2, pad_mode='same')
>>> max_pool_with_argmax = ops.MaxPoolWithArgmax(kernel_size=2, pad_mode='same')
```

</td>
</tr>
</table>

##### `ops.TensorAdd`, change API name to `ops.Add` ([!11568](https://gitee.com/mindspore/mindspore/pulls/11568))

The operator name TensorAdd is not standardized, it is changed to Add. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> add = ops.TensorAdd()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> add = ops.Add()
```

</td>
</tr>
</table>

##### `ops.Gelu`, `ops.GeluGrad`, `ops.FastGelu`, `ops.FastGeluGrad`, change API name to `ops.GeLU`, `ops.GeLUGrad`, `ops.FastGeLU`, `ops.FastGeLUGrad` ([!11603](https://gitee.com/mindspore/mindspore/pulls/11603))

Gelu, GeluGrad, FastGelu, and FastGeluGrad names are unified into ReLU naming rules, "lu" is changed to the uppercase "LU". The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gelu = ops.Gelu()
>>> gelu_grad = ops.GeluGrad()
>>> fast_gelu = ops.FastGelu()
>>> fast_gelu_grad = ops.FastGeluGrad()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gelu = ops.GeLU()
>>> gelu_grad = ops.GeLUGrad()
>>> fast_gelu = ops.FastGeLU()
>>> fast_gelu_grad = ops.FastGeLUGrad()
```

</td>
</tr>
</table>

##### `ops.GatherV2`, change API name to `ops.Gather` ([!11713](https://gitee.com/mindspore/mindspore/pulls/11713))

GatherV2 is changed to Gather. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gather = ops.GatherV2()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gather = ops.Gather()
```

</td>
</tr>
</table>

##### `ops.Pack`、`ops.Unpack`, change API name to `ops.Stack`、`ops.Unstack` ([!11828](https://gitee.com/mindspore/mindspore/pulls/11828))

Pack is changed to Stack, and Unpack is changed to Unstack. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> pack= ops.Pack()
>>> unpack= ops.Unpack()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> stack= ops.Stack()
>>> unstack= ops.Unstack()
```

</td>
</tr>
</table>

##### `ops.ControlDepend`, add deprecated to ControlDepend ([!11844](https://gitee.com/mindspore/mindspore/pulls/11844))

ControlDepend is deprecated and will be removed in a future version, use Depend instead.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```pythonNote:
Note:
    This operation does not work in `PYNATIVE_MODE`.
```

</td>
<td>

```python
Note:
        This operation does not work in `PYNATIVE_MODE`.
        `ControlDepend` is deprecated from version 1.1 and will be removed in a future version, use `Depend` instead.
```

</td>
</tr>
</table>

##### `ops.Depend`, add operator description and use case ([!11815](https://gitee.com/mindspore/mindspore/pulls/11815)), ([!11879](https://gitee.com/mindspore/mindspore/pulls/11879))

Since the ControlDepend operator will be deprecated from version 1.2, it is recommended to use the Depend operator instead.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
Depend is used for processing side-effect operations.

Inputs:
    - **value** (Tensor) - the real value to return for depend operator.
    - **expr** (Expression) - the expression to execute with no outputs.

Outputs:
    Tensor, the value passed by last operator.

Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``
```

</td>
<td>

```python
Depend is used for processing dependency operations.

In some side-effect scenarios, we need to ensure the execution order of operators.
In order to ensure that operator A is executed before operator B, it is recommended
to insert the Depend operator between operators A and B.

Previously, the ControlDepend operator was used to control the execution order.
Since the ControlDepend operator will be deprecated from version 1.2, it is
recommended to use the Depend operator instead. The replacement method is as follows::

    a = A(x)                --->        a = A(x)
    b = B(y)                --->        y = Depend(y, a)
    ControlDepend(a, b)     --->        b = B(y)

Inputs:
    - **value** (Tensor) - the real value to return for depend operator.
    - **expr** (Expression) - the expression to execute with no outputs.

Outputs:
    Tensor, the value passed by last operator.

Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

Examples:
    >>> import numpy as np
    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.ops.operations as P
    >>> from mindspore import Tensor
    >>> class Net(nn.Cell):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.softmax = P.Softmax()
    ...         self.depend = P.Depend()
    ...
    ...     def construct(self, x, y):
    ...         mul = x - y
    ...         y = self.depend(y, mul)
    ...         ret = self.softmax(y)
    ...         return ret
    ...
    >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
    >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
    >>> net = Net()
    >>> output = net(x, y)
    >>> print(output)
    [[0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]]
```

</td>
</tr>
</table>

#### C++ API

##### change namespace from `mindspore::api` to `mindspore` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
namespace ms = mindspore::api;
```

</td>
<td>

```c++
namespace ms = mindspore;
```

</td>
</tr>
</table>

##### `Context` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Context::Instance().SetDeviceTarget(ms::kDeviceTypeAscend310).SetDeviceID(0);
```

</td>
<td>

```c++
ms::GlobalContext::SetGlobalDeviceTarget(ms::kDeviceTypeAscend310);
ms::GlobalContext::SetGlobalDeviceID(0);
```

</td>
</tr>
</table>

##### rename `Tensor` to `MSTensor` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Tensor a;
```

</td>
<td>

```c++
ms::MSTensor a;
```

</td>
</tr>
</table>

##### `Model` move setting of model options from `Build` to ctor `Model` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Model model(graph_cell);
model.Build(model_options);
```

</td>
<td>

```c++
ms::Model model(graph_cell, model_context);
model.Build();
```

</td>
</tr>
</table>

##### `Model` modify `GetInputsInfo`, `GetOutputsInfo` to `GetInputs`, `GetOutputs` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
std::vector<std::string> names;
std::vector<ms::DataType> types;
std::vector<std::vector<int64_t>> shapes;
std::vector<size_t> mem_sizes;
model.GetInputsInfo(&names, &types, &shapes, &mem_sizes);
std::cout << "Input 0 name: " << names[0] << std::endl;
```

</td>
<td>

```c++
auto inputs = model.GetInputs();
std::cout << "Input 0 name: " << inputs[0].Name() << std::endl;
```

</td>
</tr>
</table>

##### `Model` modify `Predict` parameters type from `Buffer` to `MSTensor` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
std::vector<ms::Buffer> inputs;
std::vector<ms::Buffer> outputs;
model.Predict(inputs, &outputs);
```

</td>
<td>

```c++
std::vector<ms::MSTensor> inputs;
std::vector<ms::MSTensor> outputs;
model.Predict(inputs, &outputs);
```

</td>
</tr>
</table>

### Deprecations

#### Python API

##### `ops.SpaceToBatch`, `ops.BatchToSpace` are deprecated in favor of `ops.SpaceToBatchND`, `ops.BatchToSpaceND`([!11527](https://gitee.com/mindspore/mindspore/pulls/11527))

The `ops.SpaceToBatchND`, `ops.BatchToSpaceND` are more general and have same behavior as `ops.SpaceToBatch`, `ops.BatchToSpace` when `block_shape` is a int.

##### `ops.DepthwiseConv2dNative` is deprecated in favor of `nn.Conv2D`([!11702](https://gitee.com/mindspore/mindspore/pulls/11702))

The `ops.DepthwiseConv2dNative` is only supported by Ascend, it is recommended to directly use `nn.Conv2D`. If `group` is equal to `in_ channels` and `out_channels`, the 2D convolution layer is also a 2D depthwise convolution layer.

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa

Contributions of any kind are welcome!

## MindSpore Lite 1.1.0 Release Notes

### Major Features and Improvements

#### Converter and runtime

1. Support dynamic shape in MindSpore Lite Converter.
2. Optimize sub-graph mechanism by dynamically splitting the entire graph into multiple subgraphs based on the operator supported, backend hardware and user configuration.
3. Support TensorList and TensorList operators such as TensorListFromTensor, TensorListGetItem and so on.
4. Support BatchMatMul fusion and LSTM fusion in MindSpore Lite Converter.
5. Support converting model and run inference on Windows operator system.
6. Support Model(.ms) visualization on Netron.
7. Support Tensorflow model in MindSpore Lite Converter
8. Add 86 converter parsers.
9. Convert aware training model without user's awareness
10. Support scalar tensor in MindSpore Lite Converter and Runtime
11. Support NPU backend on HUAWEI Kirin SoC.[BETA]
12. Merge timeprofiler into benchmark

#### CPU backend optimization

1. Add 50+ new operators, including new Op type(like Adder, Gru).
2. Enhanced performance on armv8.2 supported platform. For example, utilizing sdot instruction more efficiently.
3. Optimize all operators(fp32, fp16, int8) by implementing multi-thread, SIMD tech as much as possible. Model inference time can reduce at least 20% after these optimizations.
4. Extending to support operators for x86_64 platform based on SSE/AVX instruction set.

#### OpenCL backend

1. Add new ops: add 10+ ops, total 58 ops;
2. Performance optimization: by memory layout optimize, Winograd Convolution select strategyoptimize, SIMT local size optimize, local cache optimize,  GPU performance improvement up to 20+% vs MSLITE Version1.0
3. Add Online Graph optimzation: by fusion Convolution/Matmul/Fullconnection and add/mul/pad/reshape, improve performance up to 50+% for some networks;
4. Add auto tuning: by online tuning in the graph compilation phase, optimize performance up to 10%;
5. Add weight quant: support weight quant
6. Add opencl kernel binary cache: improve Initialization time .

#### Post quantization

MindSpore Lite supports both weight quantization and full quantization. Currently, Weights can be quantized into 1 ~ 16 bits according to user configuration. In internal testing, quantization of networks, such as classification, detection, segmentation and transformer are well supported. To ensure high accuracy of quantized models, MindSpore Lite uses a pipeline quantization method. In the first phase, the weight and activation value are quantized using linear quantization methods, such as MIN-MAX. In the second phase, the quantization error is analyzed, and uses statistical methods to compensate loss caused by fp32 quantization to a fixed point such as Int8 to quantized models. The features of Post-training quantization are:

1. perchannel asymmetric quantization for weights, such as MAX_MIN and KMEANS
2. Perlayer symmetric quantization for activation, such as KL and MAX_MIN.
3. perlayer asymmetrical quantization for activation, such as, RemoveOutlier.
4. accuracy loss compensation, such as BiasCorrection

| mobilenet_v2   | ACC (ImageNet)  |
|---|---|
| FP32  | 71.56%  |
|A8W8   | 71.16%  |
| A8W8(without BiasCorrection)  | 70.74% |
| A8W7  | 71.06%  |
| A7W7  | 70.78%  |

The above table uses the mobilenet_v2 model from TF official website. Using MindSpore Lite quantization, the precision of A8W8 (8-bit activation value quantization and 8-bit weight quantization) decreases from 0.82% to 0.4% after accuracy loss compensation, for 7-bit quantization, the precision loss is still no more than 1%.

#### Training on Device

Within MindSpore 1.1 release, the MindSpore Lite provides the following Training-on-Device (ToD) capabilities:

1. Learning from scratch and Transfer Learning strategies are supported
2. MindSpore based models can be converted and used in training on the device. (Third-party models such as TensorFlow and PyTorch for now cannot be directly imported to the framework)
3. Grad operations are supported for more than 30 operators such as Dense layers, Convolutions and Batch Normalizations. Momentum, SGD, and ADAM optimizers are supported.
4. Supports networks such as LeNet, Alexnet, Resnet, MobileNetV1/V2/V3, and EffectiveNet, and provides complete model loading, conversion, and Python training scripts on the device side.

The MindSpore Lite ToD framework is already in use in the newest Huawei Smart TV, providing a unique and personalized user experience as a family entertainment center.

### API Change

#### API Incompatible Change

##### C++ API

- [Modify] Context now support multi-context configuration.(Context.h)
- [Modify] Callback is move from lite_session.h into ms_tensor.h.
- [Modify] GetInputsByName in lite_session.h is changed into GetInputsByTensorName
- [Add] add static LiteSession *CreateSession(const char*model_buf, size_t size, const lite::Context *context) in lite_session.h
- [Add] add GetErrorInfo interface returning error message in errorcode.h
- [Delete] Remove model_generated.h, ops_generated.h and headers of FlatBuffers library from interfaces

##### Java API

- [Add] Implement JNI layer and add Java api for CPU and GPU backend

#### Deprecations

##### C++ API

Deprecate Interface GetOutputsByNodeName

### Bug fixes

- [BUGFIX] Fix the bug in sub-graph segmentation
- [BUGFIX] Fix the bug in Tensor getitem in which the ellipsis matches the wrong dim-size.
- [BUGFIX] Fix the bug that activation modification after defining Dense will not take effect.

## Contributors

Thanks goes to these wonderful people:

zhouyifengCode, huqi, JulyAi, damon0626, chenbo116, rmdyh, davidmc, gray0v0, doitH, Gogery, zymaa, xinyunfan

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa

Contributions of any kind are welcome!

## MindSpore Lite 1.0.0 Release Notes

- Converter
    - Add 6 TFLite op, 7 Caffe op, 1 ONNX op.
    - Add support for Windows.
    - Support parallel inference of multiple sessions to adapt to more scenarios
    - Support 8bits only weight-quantization, most main-stream models has small accuracy loss (less than 0.5%) when compared to non-qunantized fp32 model.

- CPU & GPU
    - Add 20 CPU ops，include FP32, int8/uint8, FP16 and int32 ops.
    - Add supporting FP16 for GPU, add 14 GPU ops include FP32/FP16.
    - Add Buffer/Image2D transform op for GPU
    - Performance optimization for CPU ops focus on ARM32.
    - Performance optimization for GPU Convolution using winograd.

- Tool & example
    - Add object detection Android Demo.

## Bugfixes

- Models
    - fix the constant folding problem in multiply.([!6092](https://gitee.com/mindspore/mindspore/pulls/6092))
    - move batch_size from bert_net_cfg to cfg in bert scripts.([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
    - modify the checkpoint file path.([!6137](https://gitee.com/mindspore/mindspore/pulls/6137))
- Python API
    - fix semi auto parallel parameter of reshape has another user([!5722](https://gitee.com/mindspore/mindspore/pulls/5722))
    - raise ValueError when call hook function in graph mode([!5831](https://gitee.com/mindspore/mindspore/pulls/5831))
- Executor
    - fix pynative mode to build temporary nn objects.（[!6189](https://gitee.com/mindspore/mindspore/pulls/6189))
    - fix the accuracy problem of multiple inputs of multi-card communication operator broadcast.([!6522](https://gitee.com/mindspore/mindspore/pulls/5622))
    - fix the problem that the sample distribution interface categorical does not support graph mode.([!5772](https://gitee.com/mindspore/mindspore/pulls/5772))
    - fix the random seed failure problem of the polynomial downsampling distribution operator.([!5948](https://gitee.com/mindspore/mindspore/pulls/5948))
    - fix unnecessary address binding issues in GPU heterogeneous scenarios.([!6232](https://gitee.com/mindspore/mindspore/pulls/6232))
- GPU platform
    - fix for kernel resource leak([!5315](https://gitee.com/mindspore/mindspore/pulls/5315))
    - fix for insufficient memory for continuous unit test running([!5617](https://gitee.com/mindspore/mindspore/pulls/5617))
    - fix for the memory leak in the sparse slicer([!5578](https://gitee.com/mindspore/mindspore/pulls/5578))
- Data processing
    - fix hang when use pyfunc([!6346](https://gitee.com/mindspore/mindspore/pulls/6346))
    - fix GPU device queue does not release GIL during resource clean up([!5964](https://gitee.com/mindspore/mindspore/pulls/5964))
    - fix hang if scripte exit unnormally([!6441](https://gitee.com/mindspore/mindspore/pulls/6441))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, danish, Danish, dayschan, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, gongdaguo, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, root, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, Zichun, Zirui, Ziyan, zjun, ZPaC

Contributions of any kind are welcome!

## MindSpore Lite 0.7.0-beta Release Notes

- Converter
    - Support third-party models, including TFLite/Caffe/ONNX.
    - Add 93 TFLite op.
    - Add 24 Caffe op.
    - Add 62 ONNX op.
    - Add 11 optimized passes, include fusion/const fold.
    - Support aware-training and Post-training quantization.
- CPU
    - Add 100+ops，support fp32, int8/uint8, FP16 ops
    - Support fast convolution algorithms: Sliding Window, Img2col + Gemm, Strassen, Winograd
    - Support assembly/neon instruction.
    - Support CPU fp16 and sdot on ARM v8.2+.
- GPU
    - Add 20+ ops for OpenCL.
    - Support image2D/buffer format.
    - Optimize online initialization time.
    - add optimized convolution1X1/3X3/depthwise/convolution_transposed for OpenCL.
- Tool & example
    - Add benchmark and TimeProfile tools.
    - Add image classification Android Demo.

## Bugfixes

- Models
    - normalize the readme file([!5410](https://gitee.com/mindspore/mindspore/pulls/5410))
    - fix a sink_size bug for transformer([!5393](https://gitee.com/mindspore/mindspore/pulls/5393))
    - fix bool type optional for resnet50([!5363](https://gitee.com/mindspore/mindspore/pulls/5363))
- Python API
    - improve interface '__bool__' for tensor([!4000](https://gitee.com/mindspore/mindspore/pulls/4000))
    - fix GPU-ResizeNearestNeighbor([!3760](https://gitee.com/mindspore/mindspore/pulls/3760))
    - fix topK multi dimension grad func([!3711](https://gitee.com/mindspore/mindspore/pulls/3711))
    - fix scatterop error msg([!3699](https://gitee.com/mindspore/mindspore/pulls/3699))
    - fix bug of cast dtype when using mix_presion in pynative mode([!3730](https://gitee.com/mindspore/mindspore/pulls/3730))
- Executor
    - fix etsnet train error when UnsegmentSum's first input shape is (1,) ([!4573](https://gitee.com/mindspore/mindspore/pulls/4573))
    - fix bug of result error in while control flow because of unsupporting for value reference ([!4103](https://gitee.com/mindspore/mindspore/pulls/4103))
    - fix bug of the output tensor does not carry device data type ([!3774](https://gitee.com/mindspore/mindspore/pulls/3774))
    - fix bug of avoiding multi attr value are eliminated in pynative mode ([!4225](https://gitee.com/mindspore/mindspore/pulls/4225))
    - fix bug of AssignAdd unable to work normally in multi-cases ([!5171](https://gitee.com/mindspore/mindspore/pulls/5171))
- GPU platform
    - improve the environment variable checking for nvcc compiler path ([!5140](https://gitee.com/mindspore/mindspore/pulls/5140))
    - fix bug of error in cast operator conversion from fp16 to fp32 ([!4147](https://gitee.com/mindspore/mindspore/pulls/4147))
    - fix bug of the array out of bound in case of make_tuple operator ([!5219](https://gitee.com/mindspore/mindspore/pulls/5219))
- Data processing and Pro
    - fix GeneratorDataset time out([!3624](https://gitee.com/mindspore/mindspore/pulls/3624))
    - fix concat operator get_dataset_size error([!4701](https://gitee.com/mindspore/mindspore/pulls/4701))
    - fixing python validator for Repeat Op([!4366](https://gitee.com/mindspore/mindspore/pulls/4366))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Adel, Alexey, andy, andy_wangrui, anthonyaje, anzhengqi, askmiao, avakh, baihuawei, bingyaweng, BowenK, buxue, caifubi, CaoJian, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chenzupeng, chujinjin, cjh9368, Corleone, cristoval, danish, dengyutao, eric, Eric, ervinzhang, etone-chan, fangzehua, fary86, fuzhiye, gengdongjie, genglishuai, Giancarlo, gongdaguo, gukecai, guohongzilong, GuoMengHao, hangq, hanhaocheng, hanhuifeng2020, hanjun996, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, hongxing, huangdongrun, huanghui, huangxinjing, islam_amin, Jesse, jianghui58, jiangzhiwen, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, kai00, kingfo, kpy, kswang, laiyongqiang, leilei_snow, leopz, Li, liangzelang, lianliguang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, lingyunli63, linqingke, lirongzhen1, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuzhongkai, Lixia, lixian, liyong, lizhenyu, looop5, luoyang, lvchangquan, lvliang, lvwenyuan, lyvette, mahdi, Mahdi, mamba_ni, maning202007, Margaret_wangrui, mayang, meixiaowei, meng_chunyang, ms_yan, nhussain, panbingao, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, pengyongrong, Pengyongrong, qianlong, qujianwei, root, shenwei41, shibeiji, simson, songhonglei413, Su, sunsuodong, suteng, tao_yunhao, TFbunny, tinazhang, tom__chen, tony_liu2, tronzhang, VectorSL, wandongdong, wangdongxu, wanghua, wangmin, wangshaocong, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, wuyongkang, xiefangqi, xuanyue, Xun, xutianchun, xuyongfei, yanghaitao, yangjie159, YangLuo, yangruoqi713, yangyongjie, yangzhenzhang, yankai, yao_yf, yelihua, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zhangxuetong, zhaizhiqiang, Zhang, zhangxinfeng3, zhangxuetong, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaoting, zhaozhenlong, zhengjun10, zhongligeng, zhoufeng, zhousiyi, zhouyaqiang, zhouyuanshen, Zichun, Zirui, zjun, zongha, ZPaC, lijiaqi, liangchenghui, wangminggui

Contributions of any kind are welcome!

# MindSpore 0.6.0-beta Release Notes

## Major Features and Improvements

### Ascend Training and Inference Framework

- New models
    - There are official, research and community under modelzoo.
        - Official is maintained  with the newest APIs by MindSpore team,  MaskRCNN are added.
        - Research is uploaded by researchers for official review, and APIs may not  be updated in time.
        - Community reprints the relevant links of partner research results.
    - Hub added on the same level as modelzoo, synchronous storage of materials needed for official hub web pages which will be launched soon.
    - Support pre-trained models, few lines of code can be used to download and load pre-trained models, supporting inference or transfer learning.
- Frontend and user interface
    - Supports user side operator compilation and graph execution error rendering.
    - Uniform definition dynamic learning rate behavior in optimizers.
    - Support IndexSlice in sparse expression.
    - Support use parent construct method during construct.
    - Support asynchronous execution save checkpoint file.
    - Support implicit type conversion in pynative mode.
    - User interfaces change log
        - unform learning rate behavior in optimizers([!2755](https://gitee.com/mindspore/mindspore/pulls/2755))
        - rename operator of sparse optimizer([!3217](https://gitee.com/mindspore/mindspore/pulls/3217))
        - move profiler module from mindinsight to mindspore([!3075](https://gitee.com/mindspore/mindspore/pulls/3075))
        - VOCDataset output change to multi-columns([!3093](https://gitee.com/mindspore/mindspore/pulls/3093))
        - GetDatasize feature([!3212](https://gitee.com/mindspore/mindspore/pulls/3212))
        - dataset: modify config api([!2936](https://gitee.com/mindspore/mindspore/pulls/2936))
- Executor and performance optimization
    - Decouple C++ and python, so make the architecture more extensible.
    - Parameter Server for distributed deep learning supported.
    - Serving: a flexible service deployment framework for deep learning models.
    - Memory reuse is enhanced, and the batch size of Bert large model is increased from 96 to 160 on a single server.
- Data processing, augmentation, and save format
    - Support MindRecord save operator after  date processing
    - Support automatic fusion operator, such as decode/resize/crop
    - Support CSV dataset loading

### Other Hardware Support

- GPU platform
    - New model supported: ResNext50, WarpCTC and GoogLeNet.
    - Support hyperparametric search and data enhanced automl on GPU.
    - Support Resnet50 automatic parallel in GPU backend.

## Bugfixes

- Models
    - Improved the performance and accuracy on ResNet50([!3456](https://gitee.com/mindspore/mindspore/pulls/3456))
    - Fixed the performance test case of bert([!3486](https://gitee.com/mindspore/mindspore/pulls/3486))
- Python API
    - Fix assign used in while loop([!2720](https://gitee.com/mindspore/mindspore/pulls/2720))
    - Revert optimize the graph output of all nop node.([!2857](https://gitee.com/mindspore/mindspore/pulls/2857))
    - Print tensor as numpy.([!2859](https://gitee.com/mindspore/mindspore/pulls/2859))
    - Support weight decay for sparse optimizer([!2668](https://gitee.com/mindspore/mindspore/pulls/2668))
    - Fix BatchToSpaceND([!2741](https://gitee.com/mindspore/mindspore/pulls/2741))
    - Fixing type check mistakes of InplaceAdd and Inplace Sub ops([!2744](https://gitee.com/mindspore/mindspore/pulls/2744]))
    - Change order param only equal to group param([!2748](https://gitee.com/mindspore/mindspore/pulls/2748))
- Executor
    - The performance of graph with control flow is optimized([!2931](https://gitee.com/mindspore/mindspore/pulls/2931))
    - Fix bug of wrong number of tuple layers([!3390](https://gitee.com/mindspore/mindspore/pulls/3390))
    - Fix cpu multi graph memory exception([!3631](https://gitee.com/mindspore/mindspore/pulls/3631))
    - Enable data sync when calling operator without defining a cell([!3081](https://gitee.com/mindspore/mindspore/pulls/3081))
    - Fix argmaxwith value error in pynative mode on GPU([!3082](https://gitee.com/mindspore/mindspore/pulls/3082))
    - Fix precision error with fp16 input on pynative mode([!3196](https://gitee.com/mindspore/mindspore/pulls/3196))
- Data processing
    - Fix bug of RandomColor and RandomSharpness default parameter checking  ([!2833](https://gitee.com/mindspore/mindspore/pulls/2833))
    - Fix process hung when training and eval  ([!3469](https://gitee.com/mindspore/mindspore/pulls/3469))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!
