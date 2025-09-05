import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import pickle
from sklearn.preprocessing import StandardScaler

def train_random_forest(X_train, y_train, X_test, y_test, config_name = 'svm_model', use_grid_search=False):
    if use_grid_search:
        print("正在进行SVM网格搜索寻找最佳超参数...")
        param_grid = {
            'C': [0.1, 1, 10, 100],                  # 正则化参数
            'gamma': ['scale', 'auto', 0.01, 0.1, 1], # 核函数系数
            'kernel': ['rbf', 'linear', 'poly'],     # 核函数类型
            'degree': [2, 3, 4],                     # 多项式核的度数（仅对poly核有效）
            'class_weight': [None, 'balanced']       # 类别权重
        }
        # 对于大数据集，可以简化参数网格以减少计算时间
        if X_train.shape[0] > 10000:
            print("数据量较大，使用简化的参数网格...")
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf', 'linear']
            }
        base_svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            cv=3,               # 3折交叉验证
            scoring='accuracy', # 使用准确率作为评估指标
            n_jobs=-1,          # 使用所有可用的CPU核心
            verbose=1,          # 显示进度
            error_score='raise' # 遇到错误时抛出异常
        )
        grid_search.fit(X_train, y_train)
        print("最佳超参数:", grid_search.best_params_)
        print("最佳交叉验证分数:", grid_search.best_score_)
        clf = grid_search.best_estimator_
    else:
        # 使用默认参数
        print("使用默认参数训练SVM模型...")
        clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        clf.fit(X_train, y_train)
    # Make predictions and calculate metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    # Save model to pickle file
    pkl_name = config_name + '.pkl'
    with open(pkl_name, 'wb') as f:
        pickle.dump(clf, f)

    # Save results to txt file
    txt_name = config_name + '.txt'
    with open(txt_name, 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(matrix, separator=', '))

    # Print results to console (optional)
    print("Results saved to rf_results.txt")
    print("Test Accuracy:", acc)
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    return clf

input_tif = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat'
train_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset\dataset_50'
test_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset\dataset_100测试集'
if __name__ == '__main__':
    config_name = 'svc_model1' # 配置输出名称
    img = Hyperspectral_Image()
    img.init(input_tif)
    # img.image_enhance(f='PCA', n_components = 24)
    data = img.get_dataset().transpose(1,2,0)[img.backward_mask]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # SVC需要对数据进行归一化，非常关键！！
    train_position = img.create_mask_from_mutivector(train_shp_dir) # 数据集的点位文件
    train_position = train_position[img.backward_mask]
    X_train = data[train_position > 0]
    y_train = train_position[train_position > 0] - 1

    test_position = img.create_mask(test_shp_dir) # 数据集的点位文件
    test_position = test_position[img.backward_mask]
    X_test = data[test_position > 0]
    y_test = test_position[test_position > 0] - 1
    # X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42, stratify=y)
    
    clf_rf = train_random_forest(X_train, y_train, X_test, y_test, use_grid_search=True)
