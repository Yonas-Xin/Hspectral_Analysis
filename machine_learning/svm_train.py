import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import pickle
from sklearn.preprocessing import StandardScaler

def train_random_forest(X_train, y_train, X_test, y_test, config_name = 'svm_model'):
    # Initialize and train the model
    clf = SVC(kernel='rbf', C=10, gamma='scale') 
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

if __name__ == '__main__':
    config_name = 'svc_model1' # 配置输出名称
    img = Hyperspectral_Image()
    img.init(r'D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat')
    img.image_enhance(f='PCA', n_components = 24)
    position = img.create_mask(r'D:\Data\Hgy\龚鑫涛试验数据\program_data\handle_class\handle_samples_15classes.shp') # 数据集的点位文件
    X = img.enhance_data[img.backward_mask]
    position = position[img.backward_mask]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # SVC需要对数据进行归一化，非常关键！！

    X = X[position > 0]
    y = position[position > 0] - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf_rf = train_random_forest(X_train, y_train, X_test, y_test)
