# practice_reading
读书周任务
24级 | 2026年春季学期 读书实践周

# 学习资料来源
# bilibili：https://www.bilibili.com/video/BV1k5ynBzESS/?spm_id_from=333.1391.0.0&vd_source=7bd80c33ab2945435d91a978ef74ba64
# bilibili：https://www.bilibili.com/video/BV1dH9QBBEaV/?spm_id_from=333.337.search-card.all.click&vd_source=7bd80c33ab2945435d91a978ef74ba64

# 仓库文件
|README.md|项目说明文档|
|感知机模型.py| 感知机算法的Python实现|
|打地鼠.html|打地鼠网页游戏实践|
|老鼠.png|图片文件|

# 提交内容说明
1.  第一次提交：创建仓库并初始化 `README.md` 文件，搭建项目基础结构。
2.  第二次提交：上传 `感知机模型.py`，完成感知机算法的基础实现。

# 实践内容
本次实践以感知机模型为载体，实现了基础的二分类算法，核心功能包括：
- 定义感知机类，实现初始化、训练与预测方法
- 手动实现梯度下降更新权重
- 对简单线性可分数据集进行分类测试

# 学习心得
通过本次读书实践周，我一定程度上掌握的github的使用方法，学习了感知机的基本逻辑（虽然没学明白）

# 遇到的问题与解决方法
#问题1：感知机模型无法正确收敛
原因：学习率设置过大，导致权重更新时震荡无法收敛。
解决方法：将学习率从1.0调整为0.1，算法成功在有限步骤内收敛。

# 问题2：Git提交时中文文件名乱码
原因：Git默认不支持中文文件名。
解决方法：执行命令 `git config --global core.quotepath false`，关闭中文文件名转义。
