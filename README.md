# 代码说明
代码以模块的方式封装在各个文件夹中，用于调用。 
原因如下：
不同的模型对数据的要求不一样，因此我将代码分成几个部分方便日后添加和修改,在复现不同论文时，可以调用不同的包来解决问题。

## 文件结构
模块名：classfication
包含以下几个文件
bin: 包含运行整个流程的文件:
* main.py， 在main中，通过调入设计好的config文件中的参数，即可正常运行试验, 因此在使用config时需要调入公共的变量名
* config.py 共享config文件，即公共的参数，方便在不同的服务器上部署.
* inception_config.py 模板config，这里为inceptionv3 设计的参数，需根据试验设计对模型参数进行修改
设计初衷：
由于文章复现中方法的多样化， 每个部分都有自己的异同，因而在自己的config中单独引用不同的类用于分析


data: 包含处理dataset的文件，即MaskDataSet，ListDataSet两大类。
* dataset: 编写DataSet类
> 补充说明， 由于是动态载入图片的，因此，我们实际上传入DataSet的是以csv文件格式的表格，使用pandas导入数据
* MaskDataSet： 用于返回Mask和Patch，包含了数据扩增处理
* ListDataSet： 用于返回Label和Patch，包含了数据扩增处理
* sampler： 编写Sampler， 针对不同论文的方法提供相应的数据的导入策略，

csv文件格式
slide_name：即tif文件名（不包含后缀）
x,y: 坐标
label： 标签，1表示tumor，0表示normal
|slide_name|x|y|label|
|--|---|-|-|---|
|tumor_001|100|100|1|

models： 存放模型代码的地方，目前包含：
* scannet:
* inceptionv3:

postprocess: 包含后处理代码的地方
* 等待完善

preprocess: 包含模型预处理代码的地方
**注意**： 这里的代码除了可以被调用外，也可作为单独的脚本运行。
* config.py： 为generateMask提供参数的代码，原意是为整个预处理代码设置config
* generateMask.py： 根据tif和annotation文件生成对应Mask的tif文件
* extract_patches.py： 为训练模型提取坐标点，生成csv文件的代码
* wsi_op.py: 为WSI处理提供操作的代码， 主要是OTSU的生成以及Mask的生成代码

train: 包含通用的模型训练框架
* train.py 训练通用框架

utils： 辅助性的代码
* checkpoint.py 提供断点保存功能
* metrics.py 提供训练分析的代码，acc，tp,fp等等相应的值
* config.py 无用的代码（待删）