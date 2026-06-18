import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

def extract_image_features(image_path, size=(128,128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image {image_path}")
    img = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = rgb2gray(img_rgb)
    
    # Histogramme couleur RGB 8 bins
    hist_features = []
    for i in range(3):
        hist = np.histogram(img_rgb[:,:,i], bins=8, range=(0,256))[0]
        hist = hist / np.sum(hist)
        hist_features.extend(hist)
    
    # HOG
    hog_features = hog(img_gray, pixels_per_cell=(8,8),
                       cells_per_block=(2,2), visualize=False)
    
    # LBP uniforme
    lbp = local_binary_pattern(img_gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60))
    lbp_hist = lbp_hist / np.sum(lbp_hist)
    
    # Moments de Hu
    moments = cv2.moments(img_gray)
    hu_moments = np.array([cv2.HuMoments(moments)[i][0] for i in range(7)])
    
    return np.concatenate([hist_features, hog_features, lbp_hist, hu_moments])
