import streamlit as st
import pandas as pd


import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
##from skimage import data
##from skimage.color import rgb2gray
##from skimage.feature import corner_harris, corner_subpix, corner_peaks, hessian_matrix_det


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Keypoints/Descriptors', 'Harris Detector','slider')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Keypoints/Descriptors':
        keypoints_descriptors()
    if selected_box == 'Harris Detector':
        harris_detector()
    if selected_box == 'slider':
        photo()
    if selected_box == 'Hessian Detector':
        Hessian_detector()
    if selected_box == 'Difference of Gaussian':
        feature_detection()
    if selected_box == 'Harris-Laplacian':
        object_detection() 
    if selected_box == 'SIFT':
        object_detection()


def welcome():
    
    st.title('Feature Detection using Streamlit')
    
    st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
             + ' from the left. I have implemented only a few to show how it works on Streamlit. ' + 
             'You are free to add stuff to this app.')
    
    st.image('Library.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image

def keypoints_descriptors():
    st.header('Keypoints and Descriptors')

    st.latex(r'''
    \text{\underline{Keypoints} are specific locations of interest in an image:}
    ''')

    st.latex(r'''
    \text{eyes, mouth, nose, mountains, buildings, or corners.}
    ''')

     
    st.latex(r'''
    \text{A \underline{keypoint descriptor} or \underline{patch} is a vector that describes the appearance of the area surrounding a keypoint.}
    ''')


    st.image('pandafeatures.png', use_column_width=True,clamp = True)

    ## gives more information about the local region surrounding the keypoint


    ##example image here

    st.latex(r'''
    \text{Keypoints and keypoint descriptors are useful for object detection, or facial recognition.}
    ''')
    st.latex(r'''
    \text{As well as for image stitching, 3D reconstruction, or camera pose estimation.}
    ''')

    ## things that require matching between various images to reach an end goal. panorama


    st.latex(r'''
    \text{...}
    ''')
    st.header('Method: Detection, Description and Matching')
    
    st.latex(r'''
    \text{1.) Find keypoints.}
    ''')
    st.latex(r'''
    \text{2.) Take patches surrounding the keypoints. i.e. keypoint descriptors}
    ''')

    st.latex(r'''
    \text{3.) Match patches between images.}
    ''')


    st.latex(r'''
    \text{...}
    ''')

    st.subheader('Basic Intuition for Keypoint Selection')

    st.latex(r'''
    \color{green}\text{Good}\color{black}\text{ keypoints should be unique and allow for easy recognition and matching across various images.}
    ''')

    

    st.latex(r'''
    \color{red}\text{Bad}\color{black}\text{ keypoints are things such as flat regions, or regions with little deviation across x and y.}
    ''')

    st.image('houseexample.png', use_column_width=True,clamp = True)
    
    st.latex(r'''
    \text{...}
    ''')
    
    st.subheader('Additional Desireable Properties')

    st.latex(r'''
    \text{We need a certain quantity of patches, to successfully match between images.}
    ''')

    st.latex(r'''
    \text{Invariant to translation, rotation, and scale.}
    ''')

    st.latex(r'''
    \text{Resistant to affine transformations.}
    ''')

    st.latex(r'''
    \text{Resistant to lighting, color, or noise variations.}
    ''')

    st.latex(r'''
    \text{...}
    ''')

    st.subheader('Now we will see some various detectors...')
    


    


    

    
def harris_detector():
    st.header("Harris Detector")

    st.latex(r'''
    \text{The basic idea behind the harris detector is that}
    ''')

    st.image('harris_example.png',use_column_width=True)

    st.latex(r'''
        \color{red}\text{a flat region:} \color{black}\text{ no change in all directions.}

        ''')

    st.latex(r'''
        \color{red}\text{ an edge:}\color{black}\text{ no change along the edge direction.}
        ''')
    
    st.latex(r'''
        \color{green}\text{ a corner:}\color{black}\text{ significant changes in all directions.}
        ''')

    st.latex(r'''...''')
    
    st.latex(r'''
    E(u,v) = \sum_{x,y}\overbrace{w(x,y)}^{\text{window function}}\, [\, \underbrace{I(x+u,y+v)}_{\text{shifted intensity}}
    - \underbrace{I(x,y)}_{\text{intensity}}\, ]^2 
     ''')
    
    st.latex(r'''...''')

    st.latex(r'''
        \text{ If we look at the second term,}
        ''')
    st.latex(r'''
    \text{for flat regions,}\, [I(x+u,y+v) -I(x,y)]^2 \approx 0
     ''')

    st.latex(r'''
    \text{ and for distinct regions,}\, [I(x+u,y+v) -I(x,y)]^2 \approx large
    ''')

    st.latex(r'''
    \text{For corner detection we wish to } \color{red}\text{maximize}\,\color{black} E(u,v)
    ''')

    st.latex(r'''\downarrow''')
    st.latex(r'''math''')
    st.latex(r'''\downarrow''')

    
    st.latex(r'''
    E(u,v) \approx  \begin{bmatrix}
                    u & v\\
                    \end{bmatrix}
                    M
                    \begin{bmatrix}
                    u\\
                    v
                    \end{bmatrix}
     ''')


    st.latex(r'''
    M=  \sum_{x,y}w(x,y)
    
                    \begin{bmatrix}
                    I_x I_x & I_x I_y\\
                    I_y I_x & I_y I_y
                    \end{bmatrix}
                    
     ''')

    st.latex(r'''
    \text{Where } Ix \text{ and } Iy \text{ are image derivatives in x and y directions.}
    ''')

    st.latex(r'''
    \text{These can be found using the sobel kernel.}
''')


    st.latex(r'''
    G_x=
    
                    \begin{bmatrix}
                    -1 & 0 & 1\\
                    -2 & 0 & 2\\
                    -1 & 0 & 1
                    \end{bmatrix},\quad
                    
                    
                

                           
    \,\,\,G_y=
                    \begin{bmatrix}
                     1 & 2 & 1\\
                     0 & 0 & 0\\
                    -1 & -2 & -1
                    \end{bmatrix}

                                  
     ''')

    st.latex(r'''...''')

    st.latex(r'''
    \text{A scoring function R is created, which determines if a corner is captured in a window}
    ''')

    st.latex(r'''
   R = det\,M-k(\,\,Tr[M]\,\,)^2     
     ''')

    st.latex(r''' \quad det\,M = \lambda_1 \lambda_2 \quad \textrm{\&} \quad  Tr[M] = \lambda_1 + \lambda_2       
     ''')

    st.latex(r'''...''')

    st.latex(r'''
\text{Thresholding to R:}''')

    st.image('eigenvalues.png', use_column_width=True,clamp = True)

    st.latex(r'''
   \text{R}\approx \text{small} \implies \color{red}\text{flat region}      
     ''')
    st.latex(r'''
   \text{R}< 0 \implies \color{red}\text{edge}      
     ''')
    st.latex(r'''
   \text{R}\approx{large}\implies \color{green}\text{corner}      
     ''')


    
    

    filename = st.selectbox(
     'Which image do you want to process?',
     ('UCSB_Henley_gate.jpg', 'Building.jpeg', 'checkerboard.png','Library.jpg'))

    # sliders ------------------------------------------------------------------------------------------

    thresh = st.slider('Change Threshold', min_value=0.0000, max_value=.5000,step=0.0001, format='%f')

    block_size = st.slider('Change Block Size', min_value=2, max_value=10)

    aperture_size = st.slider('Change Aperture', min_value=1, max_value=31,step=2)

    k = st.slider('Harris Detector Free Variable', min_value=0.0000, max_value=.1000,step=0.0001,value=0.04, format='%f')

    iteration_count = st.slider('Change Dilation', min_value=1, max_value=100, value=2)

    # harris detector processing ------------------------------------------------------------------------
    img = cv2.imread(filename)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, block_size, aperture_size, k)

    # dilation of the points
    dst = cv2.dilate(dst, None, iterations=iteration_count)

    # Thresholding
    img[dst > thresh * dst.max()] = [0,0,255]
    st.image(img, use_column_width=True,channels="BGR")
    
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
         

    

if __name__ == "__main__":
    main()
