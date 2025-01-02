## GP (Linear mean function)

* GP的均值函数为线性形式。
* 核函数使用RBF核函数，噪声设置为0.01(当噪声设定为0.05时，汤普森采样的结果几乎总是同一个点)

<p float="center">
<img src="pics\Linear\2D.png" width="500"/>
<img src="pics\Linear\3D.png" width="500"/>
</p>

* 左图为模型的二维地形图，橙色点为推荐点的位置。右图是模型的三维示意图。
* 模型的各参数收敛曲线如下：

<p float="center">
 <img src="pics\Linear\Curves.png" width="800"/>
 </p>

* 其中weight1, weight2分别代表两个一次项的系数。

### 推荐点集(By Thompson sampling)：

| x1   | x2     | predict_y |
| ---- | ------ | --------- |
| 65.0 | 550.0  | 0.630747  |
| 54.0 | 335.0  | 0.524727  |
| 52.5 | 1725.0 | 1.380901  |
| 60.0 | 550.0  | 0.159238  |
| 50.0 | 700.0  | 0.663481  |
| 57.5 | 400.0  | 0.279805  |
| 55.0 | 125.0  | 0.807741  |
| 67.5 | 500.0  | 0.857559  |
| 50.0 | 700.0  | 0.663481  |
| 52.5 | 175.0  | 0.677103  |
| 62.5 | 625.0  | 0.835930  |
| 75.0 | 175.0  | 1.989888  |
| 65.0 | 600.0  | 0.903570  |
| 50.0 | 350.0  | 0.372373  |
| 80.0 | 400.0  | 2.270178  |
| 57.5 | 550.0  | 0.062714  |
| 50.0 | 1200.0 | 1.057441  |
| 67.5 | 250.0  | 1.525432  |
| 50.0 | 425.0  | 0.160111  |

## GP (Quadratic mean function)

* GP的均值函数为二次形式。
* 核函数使用RBF核函数，噪声设置为0.01。

<p float="center">
<img src="pics\Quadratic\2D.png" width="500"/>
<img src="pics\Quadratic\3D.png" width="500"/>
</p>

* 各参数收敛曲线如下：

<p float="center">
 <img src="pics\Quadratic\Curves.png" width="800"/>
 </p>
* 其中seconds代表两个二次项的系数，firsts代表两个一次项系数。mutual是乘积交互项系数。

### 推荐点集(By Thompson sampling)：

| x1   | x2    | predict_y |
| ---- | ----- | --------- |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 350.0 | -1.520981 |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 675.0 | -0.467032 |
| 50.0 | 325.0 | -1.613668 |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 200.0 | -2.102762 |
| 50.0 | 275.0 | -1.804180 |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 100.0 | -2.524682 |
| 50.0 | 225.0 | -2.001531 |
| 50.0 | 425.0 | -1.253156 |
| 58.0 | 245.0 | -1.112746 |
| 50.0 | 500.0 | -1.000473 |
| 50.0 | 100.0 | -2.524682 |
| 75.0 | 175.0 | 0.192348  |
| 50.0 | 375.0 | -1.430005 |
| 50.0 | 100.0 | -2.524682 |

## RBF model

<p float="center">
<img src="pics\RBF\2D.png" width="500"/>
<img src="pics\RBF\3D.png" width="500"/>
</p>

### 推荐点集：

| x1    | x2     | predict_y |
| ----- | ------ | --------- |
| 80.0  | 100.0  | -1.416397 |
| 50.0  | 100.0  | -2.398030 |
| 120.0 | 100.0  | 3.723485  |
| 80.0  | 750.0  | 2.225478  |
| 80.0  | 2250.0 | 2.321823  |
| 65.0  | 100.0  | -1.994649 |
| 100.0 | 100.0  | 0.424263  |
| 77.5  | 425.0  | 0.081470  |
| 50.0  | 350.0  | -1.092438 |
| 57.5  | 100.0  | -2.198154 |
| 62.5  | 300.0  | -1.009611 |
| 72.5  | 100.0  | -1.752220 |
| 50.0  | 225.0  | -1.741311 |
| 52.5  | 100.0  | -2.330465 |
| 55.0  | 100.0  | -2.264031 |
| 50.0  | 125.0  | -2.266761 |
| 52.5  | 125.0  | -2.199028 |
| 60.0  | 100.0  | -2.132038 |
| 50.0  | 150.0  | -2.135369 |
| 55.0  | 125.0  | -2.132766 |

## 总结：

* 综合来说我会认为线性均值GP给出的的结果更合理.因为RBF过度倾向于推荐边界点，而二次均值GP的推荐点过于集中。
