# dltm
深度学习与文本挖掘课程的讲义和资源

（1） curvefitting.py 是用神经网络进行曲线拟合的演示程序。

（2） mnist4beginner.py是手写数字识别的最基本程序（一个简单的Softmax regression模型）。可以在http://yann.lecun.com/exdb/mnist/ 下载数据集，也可以在百度网盘https://pan.baidu.com/s/1cLa-mb4wJGNinoHTQsMZjg 下载

（3）资源下载https://pan.baidu.com/s/1lbsC379B5McqaJIQRXT6Pw

（4）skipgram.py是skipgram模型
改编自https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

（5）CBOW.py是实施的CBOW模型

（6）用CNN进行文本分类的程序：

mycnn4text.py改编自http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/. 

mycnn4text_pretrain.py使用pretrained embedding的CNN，

mycnn4text_2C.py使用static和non-static两个通道（channel）的CNN。

data_helper.py是前述的原程序提供的数据文件读取程序，

preprocessor.py提供两个类，vocabulary是将元素文档转换成np的array数据结构；pretrained读取GloVe提供的pretrained embeddings文件。

另：
rt-polarity1.neg和rt-polarity1.pos是两个数据文件。
在https://nlp.stanford.edu/projects/glove/ 下载GloVe的pretrained Embedding

(6) min-char-rnn-tensorflow.py实现RNN字符级语言模型。需要的数据集是t1.txt
