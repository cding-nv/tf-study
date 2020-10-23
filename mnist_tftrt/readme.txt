g++ -std=c++11 -I/home/cding/tensorflow/ -c ./test.cc -o test.o
g++ -o test ./test.o -ltensorflow_cc -ltensorflow_framework
./test ./mnist_frozen_graph.pb ArgMax,Softmax

=====MLIR====
1	Intermediate Representations
1.1	什么是IR
IR是源语言和目标语言之间的程序表示。一个好的IR是完全独立于源语言和目标语言的IR ，这样它就可以最大限度地提高在可重定位编译器中使用的能力
1.2	为什么需要IR
a.	因为翻译似乎天生就需要分析和综合。
b.	把翻译的困难问题分成两个更简单、更易于管理的部分。
c.	要生成可重定向的编译器：
我们可以为现有的前端构建新的后端（使源语言在机器中更可移植）。
我们可以为现有的后端构建一个新的前端（这样一台新机器可以快速地为不同的源语言获取一组编译器）。
d.	做一些独立于机器的优化。
1.3	IR 种类
Structured 结构化的， 比如 图graph，树 tree-based
Tuple-based， 元组，比如关系数据库中的表
Stack-based
或者前三者的组合
1.4	Example IRs
LLVM
Java Virtual Machine
CIL Common Intermediate Langauge in Microsoft .NET framework
GNU RTL
SIL Swift Intermediate Language
Asm.js for JavaScript

2	 XLA   （accelerated linear algebra）
XLA是Google官方为在新硬件上运行TensorFlow而指定的后端开发接口

2.1	XLA HLO 和 backend
XLA 输入language 是 “HLO IR” (High Level Optimizer), HLO 是编译表示层 compiler IR。 语义链接 Operation Semantics
 
XLA 有一些优化和分析是平台无关的，比如 CSE （common subexpression elimination）， 还包括平台无关的算子融合，运行时的buffer 分析
平台无关的优化之后， XLA 就把HLO 计算图送给 backend。 Backend可以做平台相关的更深入的 HLO-level 的分析与优化。比如GPU backend 可以做更适合GPU 编程模型的算子融合和streams 的分配。在这个阶段也可以有特定的算子组合 pattern去做匹配去调用对应的优化库
再下一步就是 “target-specific code generation”, code 生成阶段。CPU 和 GPU backends 使用 LLVM进行 low-level IR， 优化， 和代码生成。 GPU backend目前支持 NVIDIA GPUs LLVM NVPTX backend。 CPU backend 支持多种CPU ISAs

2.2	XLA LLO
Low Level Optimization，面向体系结构的底层优化，由LLVM完成，主要有几个pass(AOT)，
AOT 主体流程，AOT将定义在pb文件中的graph编译成二进制目标文件，其主要目的消除图执行对tensorflow框架库的依赖，减少代码footprint，主要用在终端场景。

2.3	XLA JIT Compile 过程
第一步，CompileGraph()  转化计算图为 HLO reuqest 
第二步，BuildExecutable()   
a.	Lowering HLO request 为 HLO instruction; 
b.	DFS order; 
c.	optimize HLOModule;
d.	HloConstantFolding, HloCSE,GpuInstructionFusion,... 10+ pass
e.	Build HloSchedule, bufferAssignment, Convert HLO to LLVM  , compileToPTX, Generate GpuExecutable(ptx文本以字符串的形式保存于GpuExecutable对象中), 
第三步，executable.Run()   
a.	分配device buffer(BFC allocator) XLA有调用自己的runtime实现（bfc allocator）在device上面分配buffer
b.	Get cuda kernel from ptx(cuda runtime)
c.	Execute kernel (cuda runtime) 
 
2.4	XLA 和 GeOp在TF中的位置
session->Run() -> GetOrCreateExecutors() -> CreateGraph -> graph->optimizer() -> CreateExecutor
createGraph： 建立基础graph，标记所有xla可用节点，xla节点打包形成cluster节点（可出现多个cluster节点）
graph->optimizer() 一系列优化pass，RemoveDeadNodes()，RemoveIdentityNodes()，ConstantFold()，FixupSourceAndSinkEdges()，OptimizeCSE()，ExpandInlineFunctions() ...
createExecutor： 建立控制流信息；根据node所在的device，以及node的类型，创建cpu kernel或gpu kernel等；每个kernel有自己的compute函数，在执行kernel时会被调用进行计算
拆图部分： third_party/tensorflow/huawei_patch/tf-1.15.0/tf_plugin/tf_adapter/optimizers/om_partition_subgraphs_pass.cc  -> OMPartitionSubgraphsPass
GeOp: 
third_party/tensorflow/huawei_patch/tf-1.15.0/tf_plugin/tf_adapter/ops  -> GeOp

3	LLVM
https://zhuanlan.zhihu.com/p/75723370
Google已测试过使用LLVM编译后端代码到ARM64平台
对于非CPU硬件，有两种情况
  若设备存在LLVM后端，那么可以考虑复用xla::CPUCompiler或者xla::GPUCompiler，因为它们可以生成LLVM IR代码。
   若设备不存在LLVM后端，那么需要为设备定义完整的XLA后端代码。包括实现新的StreamExecutor，将算子映射到计算设备上。 xla::Compiler将优化过的HLO计算编译为xla::Executable。并使用xla::TransferManager管理数据在系统内存及设备内存间的传送

4	MLIR 
https://mlir.llvm.org/
Multi-Level Intermediate Representation (Compiler infrastructure)
MLIR 是 LLVM 的一部分
华为是 LLVM 创建的 Gold 赞助商 http://www.llvm.org/foundation/sponsors.html
使用了 MLIR 的框架：
•	Tensorflow (Google)
•	Tensorflow LITE (Google)
•	TFRT –Tensorflow Runtime (Google)
•	IREE (Google)
•	Swift for Tensorflow (Google)
•	PlaidML (Intel)
•	Ngrpah (Intel)
•	ONNX-MLIR (Microsoft)

4.1	MLIR 的不同
不仅仅是graph 序列化格式，还有图优化到code生成，模块化，可扩展，可选择适合自身device的 representation
多层次表示： 
tree-based 基于树的（ASTs， Abstract Syntax Tree 抽象语法树）
Graph-based 基于图的 （TF Graph， HLO）
Machine instructions （LLVM IR）

4.2	MLIR 目标
MLIR希望为各种DSL提供一种中间表达形式，将他们集成为一套生态系统，使用一种一致性强的方式编译到特定硬件平台的汇编语言上
https://mlir.llvm.org/
1.	能够表示数据流图（例如在TensorFlow中），包括动态shape、用户可扩展的op、TensorFlow变量等。
2.	图优化和变换。
3.	以适合优化的形式表示ML OP。
4.	多种优化方式，比如 fusion, loop interchange, tiling，转换数据内存布局等
5.	Code生成时的转换，比如 DMA插入，cache 管理， memory tiling等
6.	能够表示特定device的操作
7.	Deeplearning graph 量化和其他图变换。
MLIR 不支持低级机器代码生成算法（如寄存器分配、指令调度， 它们更适合用low-level优化器如LLVM）。不打算将MLIR作为最终用户自己编写内核的源语言（类似于CUDA C++），另一方面，MLIR为集成DSL到生态系统中提供了backbone。

4.3	Dialect
MLIR的核心思想就是我们可以定义自己的dialect 或者使用已有的dialects
Dialects是将所有的IR放在了同一个命名空间中，分别对每个IR定义对应的产生式以及绑定相应的操作，从而生成一个MLIR的模型。整个的编译过程，从源语言生成AST，借助Dialects遍历AST，产生MLIR的表达式，此处可为多层IR通过Lowering Pass依次进行分析，最后经过MLIR分析器，生成目标语言
Dialect 是用来扩展MLIR系统的一组功能
Dialect定义包括 operations， types， lowering 
Dialect创建一个唯一的namespace去定义operations， types， lowering， 属性等
MLIR是一个 meta-IR： 它的可扩展框架使得其可以不同的方式使用，比如编译的不同阶段。Dialects为MLIR的不同用途提供了抽象。
目前主要有的 dialects：
•	Affine (Polyhedral dialect)
•	GPU (Generic dialect)
•	Linalg (Linear algebra dialect)
•	LLVM (LLVM dialect to easy lowering to LLVM IR)
•	Quant (Quantization dialect)
•	SCF (Structured control flow dialect)
•	Shape (Shape dialect)
•	SPV (SPIR-V dialect)
•	STD (Standard dialect)
•	Vector (Vector dialect)

4.4	什么是lowering
将 operation 的较高层次表示转换为较低层次过程, 但语义上是等价的。
在MLIR中，是通过dialect conversion 来实现的。这就要求框架能指定lowering后哪些操作是合法的，哪些操作是非法的，以此来定义lower-level表示的要求。

4.5	Operations
    Operations 不是instructions， 而是类似 “Opaque functions”。 它是MLIR的code的单位，且具有特定应用的语义。 一个 operation 有0个或者多个regions， 注意，可以创建一个嵌套（nested）的IR结构，因为regions有blocks组成，而block又由operations列表组成。
Operations -> Regions -> Blocks
 
4.6	MLIR 构建
https://mlir.llvm.org/getting_started/

4.7	MLIR 表达式生成
首先要将源程序生成抽象语法树(AST)，然后再遍历抽象语法树来构建MLIR表达式。https://zhuanlan.zhihu.com/p/102395938

4.8	MLIRGen/ TableGen/ Dialect模块
https://zhuanlan.zhihu.com/p/102565792
MLIRGen模块的作用就是遍历抽象语法树(AST)
Dialect模块负责定义各种操作和分析，同时还具备可扩展性
Operation Definition Specification (ODS)框架是基于TableGen规范构造的。在源代码中，TableGen模块就是负责这一部分的描述，它定义了各种操作的类。对于transpose(a)来说，我们需要TableGen模块中的Ops.td文件定义TransposeOp
MLIR组装

4.9	什么是TableGen
https://zhuanlan.zhihu.com/p/102727417
它用于描述MLIR中Operation的类的定义，在源代码中它以.td文件的形式存在，在编译时会自动生成C++的相应文件，给Dialect模块文件提供支持
在编写完TableGen描述之后，我们可以使用mlir-tblgen工具来生成C++代码
 
builders字段会在Operation类声明中添加自定义的build方法，该方法将在Dialect模块中进行实现
https://zhuanlan.zhihu.com/p/105576276
llvm-project/mlir/examples/toy/Ch2/mlir/Dialect.cpp

4.10	优化表达式
生成了MLIR表达式之后，如何进行进一步的变型，产生更加优化的MLIR表达式 https://zhuanlan.zhihu.com/p/105905654
MLIR是一个看重代码优化的编译框架，因此在我们生成了MLIR表达式之后，就要进行进一步的处理，将MLIR表达式变型，从而实现优化的目的。既然要优化MLIR表达式，那么就少不了重写表达式，用更优的表达式替换原有表达式，而在此之前，我们当然要识别出哪个表达式存在更优化的形式。通过上述分析，我们就得出了表达式变型的两个关键步骤：表达式的匹配与重写。
MLIR中想要做到表达式的匹配与重写，可以使用两种方式，第一种方法就是直接用C++编写表达式的匹配与重写函数，第二种方式是使用Declarative Rewrite Rules (DRR)规则来定义重写规则，并使用ODS框架来自动生成代码
1.	C++编写表达式的匹配与重写函数
def transpose_transpose(x) {
  return transpose(transpose(x));
}
上面 transpose_transpose(x) 函数就是将x进行两次转置 ，也就相当于对x没有进行改变。那么这样的函数在我们没有添加变型规则的时候，生成的MLIR表达式如下所示：
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) to tensor<*xf64>
  %1 = "toy.transpose"(%0) : (tensor<*xf64>) to tensor<*xf64>
  "toy.return"(%1) : (tensor<*xf64>) -> ()
}
生成的表达式对x进行了两次转置操作，并且返回了转置后的结果。但是这样的操作是多余的，在这个例子中，我们直接返回传入的参数x 即可。因此，使用C++直接编写表达式的匹配与重写函数  llvm-project/mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
 
a.	取当前transpose操作输入的操作数
b.	若输入的操作数不是transpose 返回 failure()
c.	若输入的操作数是transpose，重写表达式返回 success()
在表达式变型后，转置嵌套操作的MLIR表达式就会直接返回参数，去掉了中间冗余的部分

2.	DRR 规则来定义重写规则
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
在这段代码中，a, b和c的行数、列数、维数和输入[1, 2]都一样，但都使用了整型操作，实际上是多余的操作，这样一来，生成的表达式就会有三个多余的reshape操作
module {
  func @main() {
    %0 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}
                           : () -> tensor<2xf64>
    %1 = "toy.reshape"(%0) : (tensor<2xf64>) -> tensor<2x1xf64>
    %2 = "toy.reshape"(%1) : (tensor<2x1xf64>) -> tensor<2x1xf64>
    %3 = "toy.reshape"(%2) : (tensor<2x1xf64>) -> tensor<2x1xf64>
    "toy.print"(%3) : (tensor<2x1xf64>) -> ()
    "toy.return"() : () -> ()
  }
}

llvm-project/mlir/examples/toy/Ch3/mlir/ToyCombine.td
 
"$0.getType() == $1.getType()" 定义匹配规则，若参数和结果类型一致，则匹配成功
“replaceWithValue $arg” 定义重写规则，若匹配成功则用参数 $arg, 直接代替 reshape 操作

将MLIR表达式变型步骤添加到编译流程中
表达式变型的步骤发生在MLIRGen模块生成表达式之后，拿到生成的MLIR表达式，添加一个Pass使用canonicalization框架来进行表达式变型。整个流程需要编写表达式匹配和重写模式，然后将自定义的模式登记为canonicalization模式。在PassManger模块中使用并运行定义好的Pass，就可以将MLIRGen生成的表达式进行变型，得到最优的MLIR表达式
llvm-project/mlir/examples/toy/Ch3/toyc.cpp
 
llvm-project/mlir/examples/toy/Ch3/include/toy/Ops.td

4.11	泛化表达式
https://zhuanlan.zhihu.com/p/106472878
内联pass 和Shape推断pass
./examples/toy/Ch4/toyc.cpp:    pm.addPass(mlir::createInlinerPass());
./examples/toy/Ch4/toyc.cpp:    optPM.addPass(mlir::toy::createShapeInferencePass());
./lib/Transforms/Inliner.cpp:std::unique_ptr<Pass> mlir::createInlinerPass() {
./examples/toy/Ch4/mlir/ShapeInferencePass.cpp

4.12	表达式优化-部分lowering
https://zhuanlan.zhihu.com/p/107137298

4.13	表达式Lowering到LLVM IR --完全Lowering
https://zhuanlan.zhihu.com/p/108386819
经过部分Lowering，产生了Toy Dialect, Affine Dialect 和 Standard Dialect混合的MLIR表达式：
我们需要继续进行Lowering，将混合Dialect完全Lowering到LLVM IR Dialect，从而输出LLVM IR表达式。为此，需要在PassManager模块中再添加一个Pass实现输出LLVM IR Dialect的MLIR表达式
./examples/toy/Ch6/toyc.cpp:    pm.addPass(mlir::toy::createLowerToLLVMPass());
./examples/toy/Ch6/toyc.cpp:  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
整个过程： MLIR表达式的生成、表达式变型、表达式优化、表达式Lowering ->
输出LLVM表达式 -> 使用JIT编译引擎来执行

5	参考资料
1.	https://mlir.llvm.org/
2.	https://www.zhihu.com/people/zhang-hong-bin-99
3.	dbox\02. System Engineering\04. Reference\4.4 Framework\Tensorflow XLA分析.docx
4.	Denis MLIR PPT
5.	dbox \02. System Engineering\04. Reference\4.11 编译器\ tensorflow xla调研.pptx

Question: how does transposeOP connect with native device code ?
Denis: In MLIR we can target native device code only by LLVM IR, so that mean we need to lower (translate) our high level ops like "Transpose" to other low level dialect and then to LLVM IR or directly to LLVM IR, and then run LLVM backend to generate a native code. Here is an example how we can do it https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/ https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/

Question：我们当初为什么没有考虑基于XLA实现一个npu 的backend 省掉一些GE pass
熊老师： 两个原因：1）我们没有XLA的控制权，无法合入XLA主线，不能自由的发展；2）我们除了要支持TF，还要支持其他框架，尤其是我们自研的MindSpore；

========xla========
1. 安装
configure时启用调试选项
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -O0 -g
编译：
bazel --output_user_root=build build --config=opt -c dbg //tensorflow/tools/pip_package:build_pip_package
打包：
bazel-bin/tensorflow/tools/pip_package/build_pip_package /xxx/path_to_build/tensorflow/build
安装：
pip install --user build/tensorflow-1.2.1-cp27-none-linux_x86_64.whl
so存放于 ~/.local/lib/python2.7/site-packages/tensorflow/python/

2.调试 
设置环境变量开启log打印（max=10）：export TF_CPP_MIN_VLOG_LEVEL=2
启动 gdb -args python mnist_soft_xla.py
在gdb命令行下设置源代码搜索目录dir /path_to_tensorflow/bazel-tensorflow

3. mnist_soft_xla.py
   -> CPYTHON 解释器
        1. session configure sess= tf.Session(config=config)
               registration (all sessions, devices, kernels); create direct session ; create device
         全局变量初始化
         训练train    tf.global_variables_initializer().run(session=sess)
       2. session->Run()  -> GetOrCreateExecutors() -> CreateGraph -> graph->optimizer() -> CreateExecutor
          createGraph：  建立基础graph，标记所有xla可用节点，xla节点打包形成cluster节点（可那出现多个cluster节点）
          graph->optimizer()   一系列优化pass，RemoveDeadNodes()，RemoveIdentityNodes()，ConstantFold()，FixupSourceAndSinkEdges()，OptimizeCSE()，ExpandInlineFunctions()  ...
          createExecutor： 建立控制流信息；根据node所在的device，以及node的类型，创建cpu kernel或gpu kernel等；每个kernel有自己的compute函数，在执行kernel时会被调用进行计算；XLA cluster将会在后面进行编译，不在此进行创建

4. 整体过程
     session->run(); -> 
     GetOrCreateExecutors() ; ->
     executor->RunAsync(); -> 
         push root node into ready_queue;
         while(!ready_queue.empty()) {
        node = ready_queue.frone();
    op_kernel = node.kernel;
    device->compute(op_kernel);
    push next nodes to ready_queue;
         }
    -> Constant kernel Compute()  
         MatMulOp kernel Compute() -> cuBlasSgemm()   Eigen3    cuBlas， cuDNN， Eigen3或者其他
         Softmax kernel Compute()   ->  Eigen3    cuBlas， cuDNN， Eigen3或者其他
         GeOp
         XlaOp kernel Compute()  ->  XLA JIT Compile 依赖LLVM编译

REGISTER_OP("GeOp")
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 2, OMPartitionSubgraphsPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 101, OMPartitionSubgraphsPass);


5. XLA JIT Compile 过程
   第一步，CompileGraph()  转化计算图为 HLO reuqest 
   第二步，BuildExecutable()   
   a.	Lowering HLO request 为 HLO instruction; 
   b.	DFS order; 
   c.	optimize HLOModule;
   d.	HloConstantFolding, HloCSE,GpuInstructionFusion,... 10+ pass
   e.	Build HloSchedule, bufferAssignment, Convert HLO to LLVM  , compileToPTX, Generate GpuExecutable(ptx文本以字符串的形式保存于                GpuExecutable对象中), 
第三步，executable.Run()   
    a.	分配device buffer(BFC allocator) XLA有调用自己的runtime实现（bfc allocator）在device上面分配buffer
    b.	Get cuda kernel from ptx(cuda runtime)
    c.	Execute kernel (cuda runtime) 

5. 
Eigen3 :
external/eigen_archive/unsupported/Eigen/CXX11/src/Tensor/TensorBase.h
http://eigen.tuxfamily.org/index.php?title=Main_Page
https://github.com/RLovelett/eigen/tree/master/unsupported/Eigen/CXX11/src/Tensor
https://eigen.tuxfamily.org/dox/unsupported/index.html

=====Tensorflow quantum  量子机器学习======
NLP embedding    vs  张量网络   张量表达信息更多 但存储猛增
学界： 张量网络可解释 从物理角度
量子硬件搞不定非线性 只能搞线性，   tensor network 是用来模拟量子硬件的行为
搞定精度 收敛是第一步 然后是速度
张量网络 其实是矩阵分解  3D矩阵


====预训练语言模型====
对语言意义数字化
one hot embedding   两两正交如何表示相近意思的词
改进：  分布式表达  word2vec   国王 - 男人 + 女人 = 女王
BERT 语境相关的语言意义数字化
   珍贵的标注语料  -> 预训练模型
华为语音助手   苹果siri  小米小爱
谷歌眼红BERT将搜索提升10%
BERT encoder     GPT decoder
语境就是词和词的关系   ->  attention   Q K V

BERT encoder   
    自编码结构
    transformer encoder
    自然语言理解， 文本分类
    推理速度很快
GPT decoder
    自回归结构
    transformer decoder
    自然语言生成任务  写诗 文章
    相对慢

模型趋势：
   训练任务创新： MT-DNN, SpanBERT
   训练流程创新：   ERNIE
   模型结构创新：  ALBERT， XLNET
模型规模：
   英伟达 Megatron-LM  83亿    GPT模型   ，   8路（8个GPU）模型并行， 64路（8组GPU 每组8个）数据并行   allreduce集合通信
   微软 Turing-NLG   170亿    也是生成式GPT
   谷歌 T5  110亿参数量        更复杂
分布式训练架构极为不同
一个32G v100只能装10亿参数模型

英伟达 Megatron-LM  83亿    GPT模型   ，   8路（8个GPU）模型并行， 64路（8组GPU 每组8个）数据并行   allreduce集合通信
1. 通过矩阵拆分来实现模型并行
2. 每个P计算矩阵乘法的一部份
3. 两个相邻的矩阵乘法拆分，第一个矩阵沿着列拆分，第二个沿着行拆分
4. 如此数学上符合矩阵分块运行规则 避免额外通信

微软 Turing-NLG   170亿    也是生成式GPT
 核心技术： ZeRO-os  分布式训练内存优化技术  一个计算设备可以容纳更多参数
算的时候需要，不算的时候其实不需要，不用放在显存。
1. N to N Parameter server 架构
2. 需要计算时再搜集或广播相应数据
3. 即时清空冗余内存
通信次数broadcast, reduce会非常多
天生适合大模型

谷歌 T5  110亿参数量        更复杂
将NLP一切问题统一到 序列到序列 问题
分布式训练： 模型平行+数据并行
内存优化技术：  AdaFactor Optimizer


====模型压缩====
  pruning， quantization， huffman encoding

在频域上的模型压缩技术
    input data -> DCT bases -> DCT feature maps  -> weighted combination -> feature maps of this layer
   DCT 是正交的变换
基于对抗学习的知识蒸馏方案  AAAI 2018
   教师神经网络 <--判别器度量特征空间上差异性---> 学生神经网络    
进化自动剪枝和量化  KDD 2018
   利用遗传算法对神经网络中冗余信息进行自动演化   -> 通道个数的搜索， 权值和特征的量化， 支持帕累托前沿的搜索
针对生成模型的协同进化压缩算法
   在cycleGAN中的生成器网络将会被同时压缩
高效的单元和结构设计
    多用卷积核： 空间多用卷积核； 用共享模板生成子卷积核； 通道多用卷积核； 用独用模板生成的子卷积核
    乐高卷积核
GhostNet 业界性能最好的端侧AI架构
    用来构建轻量神经网络架构的Ghost模块
    shuffleNet, mobileNet, ghostNet, FE-net, FBNet, 
AdderNet: 用加法代替乘法的深度神经网络
    卷积本质上也是一种度量函数  余弦相似度
    加法神经网络 BNN AddNN      乘法神经网络 CNN

神经网络架构搜索  neural arch search  
    EfficientNet
    基于连续进化的神经网络架构搜索算法   该算法可以在运行一次之后输出在 帕累托前沿 上的所有模型
    轻量的超分辨率模型架构搜索
    二值神经网络的等价问题

保护用户隐私
   data-free 方案  无需任何训练数据做压缩   iccv 2019
   在云上的  PU 压缩方案仅需要一小部分训练数据   NeurIPS 2019

模型蒸馏： 去掉原始网络中的光流模块
神经元剪枝  降低视频生成器的计算复杂度
算子优化
模型适配


====深度强化学习====  
   深度学习具有较强的感知能力， 强化学习具有决策能力。 两者结合，为复杂系统的感知决策问题提供解决思路


Dropout：Dropout通过定义的概率随机地将某个隐藏层神经元的输出设置为0，那么这
个神经元就不参与前向传播和反向传播，下一次迭代中又会根据概率重新对某个神经元输出
置0。
目标检测中衡量识别精度的指标AP（average precision）是指在多个类别物体检测中，每一个类别都可以以召回率和准确率作为横纵轴绘制一条曲线，AP 就是该曲线下的面积，mAP 是多个类别AP 的平均值


====GNN算法 graph neural network====
GCN的一个基本流程，首先是构图，采样邻居，信息传递、信息聚合、节点更新，来获取更好的用户和物品的表达
众多业务场景可以抽象为图graph上学习的问题
racket - holding - woman 抽象互动关系
edge node global  三个层次的互动 

1. cnn有规律（regular structured data）   依赖性没有规律（irregular structure data）

graph data 特征：
没有限制 neighborhood 是多少

2. Network embedding vs GNNs
   gnn 权重共享
   图嵌入需要

CNN vs GCN  Graph-CNNs
Graph-CNN： 
1.  图的空间域， weight sharing， 也有扩大感受野
2.  不同neighbors 提取特征
3.  池化  graph coarsening

Graph Neural Networks in a Nutshell
1. Sample 邻居采样 ， k=1的邻居， k=2的邻居，一跳 二跳，  随机采样， 邻居不够就上采样   邻居太多就下采样， 这一步最耗时
2. Aggregate feature info from neighbors  邻居聚合 邻居特征
3. Combine  和自身特征 concat   .   Predict graph context and label using aggregated info.  中心节点有 degree， 比如邻居比较多 说明比较重要

GraphSAGE  Thomas 2017 NeurlPS
Message Passing Graph Neural networks  消息传递 增加边上的特征

Pinterest  recommend related Pins to users 工业应用 
Pin: A visual bookmark someone has saved from the internet to a board they've created.
pin: Image, text, link
visual     ->推荐看上去像的
Random walk  -> 推荐  蔬菜
GraphSAGE   -> 节点特征 top结构关系 推荐的更靠谱

滴滴出行  对下一时段区域叫车人数预测
predict the Ride-Hailing Demand. Improve vehilcle utilization, reduce the wait-time, mitigate traffic congestion.
approach: spatial dependency modeling
1. 两个节点即区域关系 neighborhood， 公园办公区域 功能信息类似 func similarity， 公交相连 connectivity， 三种不同信息 encode pair-wise correlations between regions using multiple graph.
2. aggregat different observation with contextual Gated RNN   捕捉时序上的特征
3. Capture spatial dependency with graph convolution on multiple graphs  
4. Generate prediction

蚂蚁金服
异常 malicious accounts 检测 in the online cashless payment system at Alipay
Limitation: the current design of the receptive field is more or less hand-designed (either one-hop neighbors or K-hops neighbors)
Novelty: Adaptive receptive fields of neural networks.
用户登录信息 或 设备 手机 平板    每次登录就是一个边 某一个账号在很多设备上登录的话 那就是high risk 账号
当一个设备登录了很多不同的账号 同样 high risk
朋友信用如何

淘宝 基于异构图的大规模推荐搜索系统  Fan et al. KDD 19    交互更复杂，
Tencent  朋友圈新闻推荐  大规模数据高效采样策略   wu et al. WWW 19; Huang et al. NeurIPS 19
滴滴出行：  基于GNN时空预测  Geng et al. 2019

Uber 多伦多   交互关系行为预测  未来行程轨迹预测  Liao et al. arvix 19
DeepMind  伦敦  蒙特利尔   关系性强化学习，多智能体
Facebook  图注意力机制算法应用   社交网络节点行为预测   boy （wear shirt， use helmet，has band， wear pants， hover skateboard）

Tencent TGraph
Facebook pytorch-BigGraph
ali -  Euler 图计算平台
ali 达摩  AliGraph
亚马逊 Web Service  Deep Graph learning
huawei: 图引擎服务  graph engine service
蒙特利尔  是学术界图神经网络研究中心
    https://jian-tang.com    https://williamleif.github.io

传统深度学习模型仅仅考虑了user和item的特征，没有显式考虑连接关系
已有的GCN方法没有考虑到user和item属性的不同，用相同的方式聚合信息
已有的GCN方法使用二部图仅仅直接建模了user-item连接关系，user-user, item-item连接关系也很重要
利用GCN来同时考虑user, item的特征以及user-item的连接关系
对于user和item使用不同的聚合函数
显式建模user-user, item-item连接关系

huawei应用市场推荐系统 分两步  召回 和 排序
1 召回：粗排选取核实每个用户的内容  只推潜在的前100个可能的app 
2 排序：CTR预 精排 使用一个点击率预估模型对召回的内容进行排序   点击率预估是广告技术核心算法之一， 精确的用户点击预测可带来巨大收益
目标： 利用基于图神经网络的推荐搜索模型生产精确的embedding， 完成对用户选集app生产
关键难点： 1. 不是同构图   是异构 的   有两类不同的节点：    既有user 节点  又有 item节点
 3.  有很多top结构 多重相关性  有user-item下载  点击， user 年龄， iterm-iterm之间 都是射击游戏 体育游戏
4. 增量训练  灾难性遗忘， 只当当天的训
input embedding  ：   fiel -> 年龄 性别   离散化的数据连续化的方法
multi-graph encoding (user graph, item graph) -> skip-connection -> user embeding update / item embedding update -> info fusion , pairwise BPR loss

优化点：
1. 图表征 离散 稀疏性操作  sparse ->dense  频域？  采样，不采样也不行  cpu开很多work来做，拿到index去访问特征   90%时间
2. 采样  index 特征  比较大的cpu找出来 再聚合  memory 访问 random，  cpu-gpu 通信
3.  out of memory  采用的  mini-batch 来采样
DSSM 召回  把每一个维度特征学一个   embedding 

基于图神经网络的对无线网络场景建模  mobility robustness optimization MRO 负载均衡
流量预测 小区  traffic prediction

基于图神经网络对抗攻击 异常节点检测
含有拓扑探索的图神经网络  对错误连接比较敏感  或者重要关系没有连接，  提出动态更新拓扑结构的概率

cpu 生产者 最大的访问带宽 embedding -> gpu消费者
生产者 消费者  异步  buffer 维护好， 不要随机乱访问， 总是互相等也不行， cpu带宽会是瓶颈

QA

1. 为什么不直接采用训练时的model进行预测，而是把item embedding存起来，利用最近邻搜索的方法：
对每个用户，在线上推理的时候对几千个候选集逐一跑一遍模型的时间开销太大

2. 为什么不直接把user embedding存起来，线上推理的时候直接读取使用
对于应用市场游戏推荐业务，有上亿的用户，全部存起来需要50G+的内存，线上服务器撑不住

3. 为什么item侧用GCN，而user侧用的仍然是DSSM模型
User如果在线实时计算，需要80G+的显存，线上服务器撑不住

4. 为什么item侧使用了item ID作为特征，user侧没有使用user ID作为特征
如果把user ID进行编码作为特征，featureMap需要占用6G以上内存，线上服务器撑不住

5. 为什么GCN在训练过程中GPU利用率不能达到100%
训练速率快于采样速率，GPU需要等待CPU采样完成再进行训练

6. 为什么不用GPU来做采样
（1）邻居数目呈长尾分布，用GPU采样需要固定维度，丢失了邻居信息； （2）邻接表存储需要100G+的显存，GPU扛不住


