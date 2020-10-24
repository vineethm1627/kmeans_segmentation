import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

st.title("Image Segmentation using K-Means Clustering")
st.sidebar.title("K Means Clustering Algorithm :")
k = st.sidebar.number_input("Value of K : ", 0, 50, 3, step = 1)
st.sidebar.markdown("K Means is a clustering algorithm. Clustering algorithms are unsupervised algorithms which means that there is no labelled data available. It is used to identify different classes or clusters in the given data based on how similar the data is. Data points in the same group are more similar to other data points in that same group than those in other groups. K-means clustering is one of the most commonly used clustering algorithms. Here, k represents the number of clusters.")

img_file = st.file_uploader("Upload the input image : ", type = ['jpg', 'jpeg', 'png'])    

if img_file is not None:
    image = Image.open(img_file)
    image = np.array(image)
    # Change color to RGB (from BGR) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption = "Input Image", use_column_width = True)


    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) 

    # Convert to float type 
    pixel_vals = np.float32(pixel_vals)


    #the below line of code defines the criteria for the algorithm to stop running,  
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)  
    #becomes 85% 

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 

    # then perform k-means clustering wit h number of clusters defined as 3 
    #also random centres are initally chosed for k-means clustering 

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] 

    # reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((image.shape)) 
    cv2.imwrite("saved_output/output.jpg", segmented_image)
    st.image(segmented_image, caption = "Output Image", use_column_width = True)
