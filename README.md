## How to use

修改`config.py`中的内容。

## Local Control

原理：设定两台主机，分别为控制端和被控端。控制端处根据按键按下的情况即`win32api.GetKeyState()`将计算机的KEYBOARD状态打包发送到被控端，随后被控端同步执行对应的语句。
画个图示意一下：


## Format Transform

两个数据转换模块：
- [x] Protobuf接受的和推理Tensor之间的转换：`P2T`

- [ ] 网络输出的Tensor到KeyBoard State列表的转换：`T2K`

原始从Protobuf得到的Python可解析数据是`json`格式。我们应该是可以将其转换为Tensor进行输出。

至于Tensor到list的映射，

