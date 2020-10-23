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
