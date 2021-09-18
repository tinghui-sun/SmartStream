1. 拷贝以下源码目录
		CBB\Algorithm\AlgorithmPlugin\算法\conf
		CBB\Algorithm\AlgorithmPlugin\算法\data
		CBB\Algorithm\AlgorithmPlugin\算法\model
	到plugins对应的以下目录中
		plugins\算法\conf
		plugins\算法\data
		plugins\算法\model
  如果plugins\算法\data有压缩包，需要解压
  
2. 拷贝CBB\Algorithm\AlgorithmPlugin\算法编译出的动态库到
		plugins\算法\lib目录下
		
3. 在plguins上层目录执行./vias	

4. 目录说明
	CBB\Algorithm\AlgorithmPlugin\算法\conf 存放算法配置文件
	CBB\Algorithm\AlgorithmPlugin\算法\data 存放生成TensorRT引擎需要权重文件和矫正数据
	CBB\Algorithm\AlgorithmPlugin\算法\model 存放TensorRT推理引擎
	当model目录下推理引擎不存在时,才会用到data目录中的文件生成。引擎生成很耗时