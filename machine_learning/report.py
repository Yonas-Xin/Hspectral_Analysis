import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import pickle
from utils import label_to_rgb
import matplotlib.pyplot as plt
def print_report(clf, X_test, y_test, config_name = 'rf_model'):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

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
    img = Hyperspectral_Image()
    test_datastet = r'd:\Data\Hgy\龚鑫涛试验数据\program_data\handle_class\handle_samples_15classes_add_valdataset.shp' # 测试数据集
    img.init(r'D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat')
    img.image_enhance(f='PCA', n_components = 24)
    X = img.enhance_data[img.create_mask(test_datastet) > 0]
    y = img.create_mask(test_datastet)[img.create_mask(test_datastet) > 0] - 1

    with open(r'D:\Programing\pythonProject\Hspectral_Analysis\machine_learning\rf_model.pkl', 'rb') as f:
        clf_loaded = pickle.load(f)
    print_report(clf_loaded, X, y, config_name='rf_model')