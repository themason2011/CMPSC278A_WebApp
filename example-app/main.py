"""
Ryan: Intro to Keypoints/Descriptors, Different Detectors (Harris)
Andrew: Different Detectors (Hessian, Difference of Gaussian)
Mason: Scale-Invariant Detection (Harris-Laplacian, SIFT)

streamlit run main.py
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_subpix, corner_peaks, hessian_matrix_det
from skimage.filters import difference_of_gaussians
import pandas as pd


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Keypoints/Descriptors', 'Harris Detector', 'Hessian Detector', 'Difference of Gaussian', 'Scale-Invariant Descriptors')
    )
    
    if selected_box == 'Keypoints/Descriptors':
        photo()
    if selected_box == 'Harris Detector':
        video()
    if selected_box == 'Hessian Detector':
        Hessian_detector()
    if selected_box == 'Difference of Gaussian':
        DoG()
    if selected_box == 'Scale-Invariant Descriptors':
        Scale_Invar()
 

def welcome():
    
    st.title('Feature Detection using Streamlit')
    
    st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
             + ' from the left. I have implemented only a few to show how it works on Streamlit. ' + 
             'You are free to add stuff to this app.')
    
    st.image('Library.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image
 
def photo():

    st.header("Thresholding, Edge Detection and Contours")
    
    if st.button('See Original Image of Tom'):
        
        original = Image.open('tom.jpg')
        st.image(original, use_column_width=True)
        
    image = cv2.imread('tom.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)  

    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True,clamp = True)
    
    st.text("Bar Chart of the image")
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    st.bar_chart(histr)
    
    st.text("Press the button below to view Canny Edge Detection Technique")
    if st.button('Canny Edge Detector'):
        image = load_image("jerry.jpg")
        edges = cv2.Canny(image,50,300)
        cv2.imwrite('edges.jpg',edges)
        st.image(edges,use_column_width=True,clamp=True)
      
    y = st.slider('Change Value to increase or decrease contours',min_value = 50,max_value = 255)     
    
    if st.button('Contours'):
        im = load_image("jerry1.jpg")
          
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,y,255,0)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
 
        
        st.image(thresh, use_column_width=True, clamp = True)
        st.image(img, use_column_width=True, clamp = True)
         

    
def video():
    uploaded_file = st.file_uploader("Choose a video file to play")
    if uploaded_file is not None:
         bytes_data = uploaded_file.read()
 
         st.video(bytes_data)
         
    video_file = open('typing.mp4', 'rb')
         
 
    video_bytes = video_file.read()
    st.video(video_bytes)
 

def Hessian_detector():
    #Andrew Yung
    st.header("Feature Detection with Hessian Detector")
    st.subheader("How it works:")
    st.write("1. The Hessian of the image corresponds to the curvature of the image based on its pixel values.")

    st.latex(r'''
    H(I) = \begin{bmatrix}
    I_{xx} & I_{xy} \\
    I_{xy} & I_{yy}
    \end{bmatrix}
     ''')
        
    st.write("2. When we perform the eigenvalue decomposition of H(I)(x,y), the eigenvectors correspond to the direction of greatest and lowest curvature and their respective eigenvalues correspond to the magnitude of curvature")
    st.latex(r'''
    eig(H(I)) = \left\{
        \begin{array}{ll}
            \underline{e_1} , \lambda_1 \text{=> Greatest curvature}\\
            \underline{e_2} , \lambda_2 \text{=>Lowest curvature}
        \end{array}
    \right.
    ''')

    st.write("3. Since we are only interested in the strength of curvature we can simply take the determinant of H to yield the overall curvature strength for all x,y coordinates")
    st.latex(r'''
    det(H) => \lambda_1 * \lambda_2
    ''')
    st.write("4. Threshold the determinant \"image\" to yield our coordinate features!")

    st.subheader("Hessian Detector Demo")
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if image_file is not None:
        image = Image.open(image_file)
        img = np.array(image)
        img_rgb = img
    else:
        img = load_image('Banff.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x,y = img_gray.shape
    rad = int(0.0065 * x)
    max_dis= 10*int( 0.004 *x)

    thres = st.slider('Change Threshold value',min_value = 0.01,max_value = 0.5, value=0.05)
    min_dis = st.slider('Change Minimum Distance',min_value = 1,max_value = max_dis)
    

    #st.image(img_gray, use_column_width=True)
    dets = hessian_matrix_det(img_gray)
    #st.image(-dets, clamp = True, channels = 'gray')

    coords_hessian = corner_peaks(hessian_matrix_det(img_gray), min_distance=min_dis, threshold_rel=thres)

    st.text("Hessian Features Detected")
    
    HesImg = img_rgb
    for (y,x) in coords_hessian:
        HesImg = cv2.circle(HesImg, (x,y), radius=rad, color=(255,0,0), thickness=-1)
    st.image(HesImg, use_column_width=True,clamp = True)
    

def sigmoid(x,s):
    #Andrew Yung
    if (s == 0):
        l = len(x)
        s = np.zeros(l)
        hf= l//2
        s[hf:l] = 1
        sig = s
    else:
        z = np.exp(-x/s)
        sig = 1 / (1 + z)
    
    return sig

def DoG():
    ## Andrew Yung
    st.header("Difference of Gaussian Detector")
    st.subheader("How it works:")
    st.write("1. We take two blurred versions of the image w.r.t two sigmas")
    sig0 = st.slider('Select a sigmas', 0.0, 10.0, (0.0, 0.0))
    st.write("2. We subtract the two blurred images and yield a bandpass filterd image")
    x = np.arange(-5,5,0.01, dtype = float)
    s0 = sigmoid(x,0)
    s1 = sigmoid(x,sig0[0])
    s2 = sigmoid(x,sig0[1])
    s3 = s2-s1
    s = np.stack((s0,s1,s2,s3),axis=1)
    
    df = pd.DataFrame(s, columns=['Edge','s1','s2',"s2-s1"])
    st.line_chart(df)
    st.write("3. We threshold the new image to yield our feature points/edges")

    

    st.subheader('Difference of Gaussian in images')
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if image_file is not None:
        image = Image.open(image_file)
        img = np.array(image)
        img_rgb = img
    else:
        img = load_image('jerry.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    x,_ = img_gray.shape
    rad = int(0.007 * x)
    max_dis= 10*int( 0.004 *x)

    thres = st.slider('Change Threshold value',min_value = 0.01,max_value = 1.0)
    min_dis = st.slider('Change Minimum Distance',min_value = 1,max_value = max_dis)
    sig = st.slider('Select a sigmas', 0.0, 50.0, (2.0, 10.0))
    dog = difference_of_gaussians(image=img_gray, low_sigma=sig[0], high_sigma=sig[1], channel_axis=-1)
    norm_image = cv2.normalize(dog, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    st.image(norm_image,use_column_width=True,clamp = True)
    coords_dog = corner_peaks(dog, min_distance=1, threshold_rel=thres)

    DogImg = img_rgb
    for (y,x) in coords_dog:
        DogImg = cv2.circle(DogImg, (x,y), radius=rad, color=(255,0,0), thickness=-1)
    st.image(DogImg, use_column_width=True,clamp = True)

def Scale_Invar():
    ##Mason Corey
    st.header('Scale-Invariant Detectors')
    st.subheader("Why do we need Scale Invariance?")
    st.write('Common detectors, like the Harris and Hessian detectors, are often invariant to things like illumination, translation, and rotation, but not scaling.')
    intro1 = load_image("scale_inv_intro1.png")
    intro1_rgb = cv2.cvtColor(intro1, cv2.COLOR_BGR2RGB)
    st.image(intro1_rgb)
    st.text('')

    st.write('Regions of different sizes will look the same in two images that only differ in scaling. How can we make a detector that will find the same keypoints independently in two images with different scale?')
    intro2 = load_image("scale_inv_intro2.png")
    intro2_rgb = cv2.cvtColor(intro2, cv2.COLOR_BGR2RGB)
    st.image(intro2_rgb)
    st.text('')

    st.subheader('Naive Approach')
    st.write('The Naive Approach is to take two images that differ only in scale, compute the Gaussian pyramid for both, and do NxN pairwise comparisons to match similar pyramids and determine the relative scale for keypoint detection.')
    st.markdown("""
    * Drawbacks:
        * Very computationally expensive
        * Requires more than one image to compare
    """)
    intro3 = load_image("scale_inv_intro3.png")
    intro3_rgb = cv2.cvtColor(intro3, cv2.COLOR_BGR2RGB)
    st.image(intro3_rgb)
    st.text('')

    st.subheader('More Robust Solution: The Laplacian Pyramid')
    st.write('We want to generate keypoints that will be found in the same location regardless of scale and can be found independently of other images (i.e. no comparison required).')
    st.write('To do this, we need to find a function to apply to the image that has some point which is identifiable regardless of scale, which we can set as a keypoint.')
    st.write('The easiest function to use is one with a single maximum peak. The maximum will not change with scale, so we can use the maximum point as our keypoint.')
    intro4 = load_image("scale_inv_intro4.png")
    intro4_rgb = cv2.cvtColor(intro4, cv2.COLOR_BGR2RGB)
    st.image(intro4_rgb)
    intro5 = load_image("scale_inv_intro5.png")
    intro5_rgb = cv2.cvtColor(intro5, cv2.COLOR_BGR2RGB)
    st.image(intro5_rgb)
    st.text('')
    st.write('The most ideal function that matches these characteristics is the Laplacian Pyramid, which can be quickly approximated using the Difference of Gaussians:')
    intro6 = load_image("scale_inv_intro6.png")
    intro6_rgb = cv2.cvtColor(intro6, cv2.COLOR_BGR2RGB)
    st.image(intro6_rgb, width=500)
    st.text('')
    st.write('You can then find the characteristic scale for each keypoint, which is the scale that produces the peak response for the Derivative of Gaussian of the image in the area of the keypoint.')
    st.write('The characteristic scale for a given keypoint will give the best invariance to scale for that keypoint.')
    intro7 = load_image("scale_inv_intro7.png")
    intro7_rgb = cv2.cvtColor(intro7, cv2.COLOR_BGR2RGB)
    st.image(intro7_rgb, width=500)
    intro8 = load_image("scale_inv_intro8.JPG")
    intro8_rgb = cv2.cvtColor(intro8, cv2.COLOR_BGR2RGB)
    st.image(intro8_rgb)
    st.text('')

    st.subheader('Implementation of Scale-Invariant Detection in Industry')
    st.markdown("""
    * There are two common implementations of Scale-Invariant Detectors:
        * Harris-Laplacian Detection
        * SIFT (Scale-Invariant Feature Transformation)
    """)
    intro9 = load_image("scale_inv_intro9.png")
    intro9_rgb = cv2.cvtColor(intro9, cv2.COLOR_BGR2RGB)
    st.image(intro9_rgb)
    
    clock = cv2.CascadeClassifier('haarcascade_wallclock.xml')  
    found = clock.detectMultiScale(img_gray,  
                                   minSize =(20, 20)) 
    amount_found = len(found)
    st.text("Detecting a clock from an image")
    if amount_found != 0:  
        for (x, y, width, height) in found:
     
            cv2.rectangle(img_rgb, (x, y),  
                          (x + height, y + width),  
                          (0, 255, 0), 5) 
    st.image(img_rgb, use_column_width=True,clamp = True)
    
    
    st.text("Detecting eyes from an image")
    
    image = load_image("eyes.jpg")
    img_gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    img_rgb_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
    eye = cv2.CascadeClassifier('haarcascade_eye.xml')  
    found = eye.detectMultiScale(img_gray_,  
                                       minSize =(20, 20)) 
    amount_found_ = len(found)
        
    if amount_found_ != 0:  
        for (x, y, width, height) in found:
         
            cv2.rectangle(img_rgb_, (x, y),  
                              (x + height, y + width),  
                              (0, 255, 0), 5) 
        st.image(img_rgb_, use_column_width=True,clamp = True)

if __name__ == "__main__":
    main()
