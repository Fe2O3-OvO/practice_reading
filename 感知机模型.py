import numpy as np
import matplotlib.pyplot as plt

# 生成数据集
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
y = np.where(y == 0, -1, 1) # 标签映射为{-1, 1}

# 数据预处理
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#模型实现
class Perceptron:
    def __init__(self, lr=0.01, max_iter=1000,gd_type='sgd', batch_size=32):
        self.lr = lr
        self.max_iter = max_iter
        self.gd_type=gd_type
        self.batch_size=batch_size
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
        n_samples=X.shape[0]

        for epoch in range(self.max_iter):
            loss = 0
            if self.gd_type=='bgd':
                delta_w=np.zeros(self.w.shape)
                delta_b=0
                for xi, yi in zip(X, y):
                    if yi * (np.dot(self.w, xi) + self.b) <= 0:
                        delta_w += self.lr * yi * xi
                        delta_b += self.lr * yi
                        loss += 1
                        
                if loss>0:
                    self.w += delta_w
                    self.b += delta_b
                self.losses.append(loss)
                if loss == 0:
                    break

            elif self.gd_type == 'mini-batch':
                # 小批量梯度下降：按batch_size分割数据
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                for i in range(0, n_samples, self.batch_size):
                    X_batch = X_shuffled[i:i+self.batch_size]
                    y_batch = y_shuffled[i:i+self.batch_size]
                    delta_w = np.zeros_like(self.w)
                    delta_b = 0
                    batch_loss = 0
                    for xi, yi in zip(X_batch, y_batch):
                        if yi * (np.dot(self.w, xi) + self.b) <= 0:
                            delta_w += self.lr * yi * xi
                            delta_b += self.lr * yi
                            batch_loss += 1
                    if batch_loss > 0:
                        self.w += delta_w
                        self.b += delta_b
                    loss += batch_loss
                self.losses.append(loss)
                if loss == 0:
                    break

            else:  # 'sgd' 随机梯度下降
                # 随机梯度下降：每个样本更新一次
                for xi, yi in zip(X, y):
                    if yi * (np.dot(self.w, xi) + self.b) <= 0:
                        self.w += self.lr * yi * xi
                        self.b += self.lr * yi
                        loss += 1
                self.losses.append(loss)
                if loss == 0:
                    break
        return self
 
    def predict(self, X):
        # 预测函数：返回1或-1
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)


    
# 对偶形式
class DualPerceptron:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.gram = X.dot(X.T) # Gram矩阵
        self.x_train=X
        self.y_train=y

        for _ in range(self.max_iter):
            for i in range(n_samples):
                if self.y_train[i] * (np.sum(self.alpha * y_train * self.gram[i]) + self.b) <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
        return self
    def predict(self, X):
        # 对偶形式预测：基于训练样本内积
        y_pred = []
        for x in X:
            inner_product = np.dot(self.x_train, x)  # 与所有训练样本的内积
            linear_output = np.sum(self.alpha * self.y_train * inner_product) + self.b
            y_pred.append(1 if linear_output >= 0 else -1)
        return np.array(y_pred)

# 训练原始感知机（示例：SGD）
perceptron = Perceptron(lr=0.01, max_iter=1000, gd_type='sgd')
perceptron.fit(X_train, y_train)

# 训练对偶形式感知机
dual_perceptron = DualPerceptron(lr=0.01, max_iter=1000)
dual_perceptron.fit(X_train, y_train)



#可视化训练过程
#损失曲线
plt.plot(range(len(perceptron.losses)), perceptron.losses)
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassifications')
plt.title('Training Loss Curve')
plt.show()


#超平面动态变化
def plot_decision_boundary(model, X, y):
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(perceptron, X_train, y_train)
plot_decision_boundary(dual_perceptron, X_train, y_train)

#不知到感知机模型原理能否用图像表示
