### GP (Zero mean function)

<p float="center">
<img src="pics\Zero\2D.png" width="500"/>
<img src="pics\Zero\3D.png" width="500"/>
</p>

* GP核函数默认采用Matern2.5
* 二维图像中红色五角星标注的位置是模型的预测最优值所在位置
* 未知区域函数值基本全是0

### GP (Constant mean function)

<p float="center">
<img src="pics\Constant\2D.png" width="500"/>
<img src="pics\Constant\3D.png" width="500"/>
</p>

* 与Zero类似，未知区域由0变为了训练出的均值常数
* 模型训练采用梯度下降优化损失函数的对数似然(迭代1000次)，损失函数和超参数的具体收敛曲线如下图：

<p float="center">
 <img src="pics\Constant\Curves.png" width="800"/>
 </p>
* 其中左图代表loss function，右图是constant参数梯度下降的迭代曲线

### GP (Linear mean function)

<p float="center">
<img src="pics\Linear\2D.png" width="500"/>
<img src="pics\Linear\3D.png" width="500"/>
</p>

* 各参数收敛曲线如下：

<p float="center">
 <img src="pics\Linear\Curves.png" width="800"/>
 </p>
* 其中weights代表两个一次项的系数。

### GP (Quadratic mean function)

<p float="center">
<img src="pics\Quadratic\2D.png" width="500"/>
<img src="pics\Quadratic\3D.png" width="500"/>
</p>

* 边缘部分缺少观测值，数值极度不稳定
* 各参数收敛曲线如下：

<p float="center">
 <img src="pics\Quadratic\Curves.png" width="800"/>
 </p>
* 其中seconds代表两个二次项的系数，firsts代表两个一次项系数。mutual是乘积交互项系数。

### 总结：

* 观测点数量相对搜索空间太少，在没有观测的地区，先验均值函数是模型推理的主要驱动力。模型所能做的最好的事情就只能是简单地呈现均值函数中的先验知识，并没有太高的参考价值。
