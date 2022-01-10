from numpy.core.records import record
import streamlit as st
st.set_page_config(
    page_title="Game Keys",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import time
from PIL import Image
import tempfile

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'test3.mp4'
prevTime=0
currTime=0 
tipIds= [4,8,12,16,20]
st.title('Game Keys with Hand Tracking Web-App')
import webbrowser

url = 'https://github.com/shashankanand13monu/Game-Automation'

# if st.button('Code'):
#     webbrowser.open_new_tab(url)

# st.set_page_config(
#     page_title="Ex-stream-ly Cool App",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

mp_drawing = mp.solutions.drawing_utils
mp_draw= mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_draw= mp.solutions.drawing_utils
mp_hand= mp.solutions.hands
mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
# mp_hand= mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
t = st.empty()
def draw(str):
    # t.subheader(str)
    # original_title = 'f<p style="font-family:Arial Black; color:red; font-size: 30px;">{str}</p>'
    # st.markdown('f<p style="font-family:Arial Black; color:red; font-size: 30px;">{str}</p>', unsafe_allow_html=True) 
    t.markdown(f'<p style="font-family:Arial Black;color:#FF0686;font-size:28px;;">{str}</p>', unsafe_allow_html=True)
#     variable_output = str
#     font_size = 30

#     html_str = f"""
# <style>
# p.a {{
#   font: bold {font_size}px Courier;
# }}
# </style>
# <p class="a">{variable_output}</p>
#     """

#     st.markdown(html_str, unsafe_allow_html=True)

st.markdown(
    """
    <style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
width: 350px
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
width: 350px
margin-left: -350px
</style>
    """,unsafe_allow_html=True,)
if st.sidebar.button('Code'):
    webbrowser.open_new_tab(url)
st.sidebar.title('Menu')

st.sidebar.subheader('Settings')

@st.cache ()
def image_resize(image, width=None, height=None, inter =cv2.INTER_AREA):
    
    dim = None
    (h ,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r= width/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    #resize the image
    resized =cv2.resize(image, dim ,interpolation=inter)
    return resized

app_mode= st.sidebar.selectbox('Choose the App Mode',
                               ['About App','Run on Image','Run On Video'])

# st.sidebar.markdown('---')
# st.sidebar.button('Code')
    # st.sibutton('Code')
# if st.sidebar.button('Code'):
#     webbrowser.open_new_tab(url)

if app_mode== 'About App':
    st.markdown('App Made using **Mediapipe** & **Open CV**')

    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """,unsafe_allow_html=True,)

    st.markdown('''
                # Tutorial \n
                
                '''
                )
    original_title = '<pre style="font-family:Aku & Kamu; color:#FD0177; font-size: 25px;font-weight:Bold">🕹️ W- 5 Fingers  🕹️ A- 2 or 3 Fingers</pre>'
    st.markdown(original_title, unsafe_allow_html=True)
    original_title = '<pre style="font-family:Aku & Kamu; color:#FD0177; font-size: 25px;font-weight:Bold;">🕹️ S- Fist       🕹️ D- 4 Fingers</pre>'
    st.markdown(original_title, unsafe_allow_html=True)
    # st.subheader('''W - 5 Fingers''')
    # st.subheader('S - Fist\n A - 2 or 3 Fingers\n D - 4 Fingers')
    st.image('wsad.jpg',width=200)
    original_title = '<pre style="font-family:Aku & Kamu; color:#FD0101 ; font-size: 28px;font-weight:Bold">*NOTE</pre>'
    st.markdown(original_title, unsafe_allow_html=True)
    original_title= '''<pre style="font-family:Aku & Kamu; color:#270F40; font-size: 24px;">
    Video Option will Experience Lag in  Browsers.
    If It's <strong>Lagging</strong> just <strong>Reload</strong> & Choose your option <strong>ASAP</strong> 
    eg: <strong>Choosing Max Hands</strong> or <strong>Using Webcam.</strong> 
    Webcam Will Take about <strong>20 Seconds</strong> to Load</pre>'''
    # st.markdown('''Video Option will Experience **Lag** in **Browsers**. If It's **Lagging** just **Reload** & Choose your option ASAP eg: **Choosing Max Hands** or **Using Webcam**. Webcam Will Take about **20 Seconds** to Load ''')
    st.markdown(original_title, unsafe_allow_html=True)
    
    
elif app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown ('---------' )
    
    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """,unsafe_allow_html=True,)

    # st.markdown("**Detected Hands**")
    st.header("**   Detected Hands   **")
    kpi1_text = st.markdown("0")
    
    max_hands= st.sidebar.number_input('Maximum Number of Hands',value=2,min_value=1,max_value=4)
    # st.sidebar('---')
    detection_confidence= st.sidebar.slider('Detection Confidence',min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')
    IMAGE_FILE=[]
    count=0
    
    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg", "png"])
    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image = opencv_image.copy()
        # image = np.array(Image.open(img_file_buffer))
    else:
        demo_image= DEMO_IMAGE
        # file_bytes = np.asarray(bytearray(demo_image.read()), dtype=np.uint8)
        # opencv_image = cv2.imdecode(file_bytes, 1)
        # image = opencv_image.copy()
        # image= np.array(Image.open(demo_image))
        image = 'demo.jpg'
        cap = cv2.imread('demo.jpg', cv2.IMREAD_UNCHANGED)
        image = cap.copy()

    # st.sidebar.text('Input Image')
    st.sidebar.subheader('Input Image')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    st.sidebar.image(image) 
    # st.sidebar.markdown('---')
    # st.sidebar.text('Demo Image')
    st.sidebar.subheader('Demo Images')
    # st.sidebar.image('WIN_20211106_14_34_25_Pro.jpg')
    st.sidebar.image('360_F_61499604_hkfPSZ4ZYs47Yp8H780DEb3I3cvWjdmH.jpg')  
    st.sidebar.image('Screenshot 2022-01-09 161732.png')
    st.sidebar.image('woman-showing-four-fingers-white-background-woman-showing-four-fingers-white-background-closeup-hand-134504006.jpg')

    st.sidebar.image('demo.jpg')
    hand_count =0
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cap = cv2.imread('demo.jpg', cv2.IMREAD_UNCHANGED)
    # xxx= image
    # xxx= f'{image}'
    # IMAGE_FILE.append(xxx)
    # plt.imshow(image_batch[i].numpy().astype("uint8"))
    # IMAGE_FILE.append(image)
    # image = cv2.imread(opencv_image, cv2.IMREAD_UNCHANGED)
# IMAGE_FILES.append(image)
    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=max_hands,
    min_detection_confidence=detection_confidence) as hands:
        
        hand_count+=1
#   for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    # image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
        # int('Handedness:', results.multi_handedness)
        # if not results.multi_hand_landmarks:
        #   continue
        try:
            
            age_height, image_width, _ = image.shape
            annotated_image = image.copy()
            lmList=[]
            lmList2forModel=[]
            for hand_landmarks in results.multi_hand_landmarks:
            #   print('hand_landmarks:', hand_landmarks)
            #   print(
            #       f'Index finger tip coordinates: (',
            #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            #   )
                myHands=results.multi_hand_landmarks[0]

                for id,lm in enumerate(myHands.landmark):
                    
                    h,w,c=image.shape
                    cx,cy=int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                    lmList2forModel.append([cx,cy])
                fingers=[]
                if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                    fingers.append(1)
                        
                else:
                    fingers.append(0)
                          
                    
                for id in range(1,5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id]-1][2]:
                        fingers.append(1)
                            
                            
                    else:
                        fingers.append(0)
                            
                total= fingers.count(1)
                    
                if total==5:
                    # st.text('Acclerate')
                    # st.subheader("Acclerate")
                    # mmm=st.markdown("Acclerate")
                    # st.markdown('---')
                    original_title = '<p style="font-family:Arial Black; color:#FD0177; font-size: 30px;">Acclerate</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                    # st.markdown('---')
                    
                    
                    overlay = cv2.imread('istockphoto-1179377734-612x612.jpg')
                    overlay = cv2.resize(overlay,(100,100))
                    x_offset = 80
                    y_offset = 10
                    x_end = x_offset + overlay.shape[1]
                    y_end = y_offset + overlay.shape[0]
                    annotated_image[y_offset:y_end,x_offset:x_end] = overlay
                    
                if total==4:
                    # st.text('Right')
                    original_title = '<p style="font-family:Arial Black; color:#FD0177; font-size: 30px;">Right</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                    overlay = cv2.imread('istockphoto-1179377734-612x612 (4).jpg')
                    overlay = cv2.resize(overlay,(100,100))
                    x_offset = 120
                    y_offset = 50
                    x_end = x_offset + overlay.shape[1]
                    y_end = y_offset + overlay.shape[0]
                    annotated_image[y_offset:y_end,x_offset:x_end] = overlay
                    
                if total==2 or total==3:
                    # st.text('Left')
                    original_title = '<p style="font-family:Arial Black; color:#FD0177; font-size: 30px;">Left</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                    overlay = cv2.imread('istockphoto-1179377734-612x612 (1).jpg')
                    overlay = cv2.resize(overlay,(100,100))
                    x_offset = 50
                    y_offset = 50
                    x_end = x_offset + overlay.shape[1]
                    y_end = y_offset + overlay.shape[0]
                    annotated_image[y_offset:y_end,x_offset:x_end] = overlay
                    
                if total==0:
                    # st.text('Brake')
                    original_title = '<p style="font-family:Arial Black; color:#FD0177; font-size: 30px;">Brake</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                    overlay = cv2.imread('istockphoto-1179377734-612x612 (3).jpg')
                    overlay = cv2.resize(overlay,(100,100))
                    x_offset = 50
                    y_offset = 90
                    x_end = x_offset + overlay.shape[1]
                    y_end = y_offset + overlay.shape[0]
                    annotated_image[y_offset:y_end,x_offset:x_end] = overlay
                mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
        mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
            # annotated_image= cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)    
            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
        # st.subheader('Output Image')
        # st.image(annotated_image, use_column_width=True)
        except TypeError:
            # st.text('Sorry No Hand Found!!')
            original_title = '<p style="font-family:Arial Black; color:Red; font-size: 30px;">Sorry, No Hand Found!!</p>'
            st.markdown(original_title, unsafe_allow_html=True)

            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{0}</h1>", unsafe_allow_html=True)

        #     annotated_image= cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)    
        #     kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        annotated_image = cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)
        st.image(annotated_image, use_column_width=False)
           
elif app_mode == 'Run On Video':
    
    st.set_option('deprecation.showfileUploaderEncoding',False)
    # st.text('This is some text.')
    # st.text('This is somext.')
    use_webcam = st.sidebar.button('Use Webcam')
    record= st.sidebar.checkbox("Record Video")
    
    if record:
        st.checkbox("Recording",value=True)
    
    # drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    # st.sidebar.markdown ('---------' )
    
    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """,unsafe_allow_html=True,)

    # st.markdown("**Detected Hands**")
    # kpi1_text = st.markdown("0")
    
    max_hands= st.sidebar.number_input('Maximum Number of Hand',value=1,min_value=1,max_value=4)
    # st.sidebar('---')
    detection_confidence= st.sidebar.slider('Detection Confidence',min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence= st.sidebar.slider('Tracking Confidence Confidence',min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')
    
    # st.markdown("Output")
    st.subheader("Input Video")    
    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    ##We get our input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    #Recording Part
    codec = cv2.VideoWriter_fourcc('V', 'P', '0','9')
    out= cv2.VideoWriter('output.mp4',codec,fps_input,(width,height))
    
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    # st.markdown('Reload , if webpage hangs') 
     
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        # st.markdown("**Frame Rate**")
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Frame Rate</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi1_text = st.markdown ("0")
    with kpi2:
        # st.markdown("**Detected Hands**")
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Detected Hands</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi2_text = st.markdown ("0")
    with kpi3:
        # st.markdown("**Width**")
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Video Width</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi3_text = st.markdown("0")
    st.markdown ("<hr/>", unsafe_allow_html=True)
    st.subheader('Reload , if webpage hangs')
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # st.subheader("Input Video")
        
    with mp_hand.Hands(max_num_hands=max_hands,min_detection_confidence=detection_confidence,
                       min_tracking_confidence=tracking_confidence) as hands:
    
        
        while vid.isOpened():
            
            i +=1
            ret, image = vid.read()
            # image.set(3, 640)
            # image.set(4, 480)
            if not ret:
                continue
        
          
            image.flags.writeable=False
            results= hands.process(image)
            image.flags.writeable=True
            image= cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            lmList=[]
            lmList2forModel=[]
            hand_count=0
            if results.multi_hand_landmarks:
                
                # original_title = 'Acclerate'
                

            # Face Landmark Drawing
                for hand_landmark in results.multi_hand_landmarks:
                    hand_count += 1
                    myHands=results.multi_hand_landmarks[0]
                    for id,lm in enumerate(myHands.landmark):
                        h,w,c=image.shape
                        cx,cy=int(lm.x*w), int(lm.y*h)
                        lmList.append([id,cx,cy])
                        lmList2forModel.append([cx,cy])
                    
                    if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                        fingers.append(1)

                    else:
                        fingers.append(0)


                    for id in range(1,5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id]-1][2]:
                            fingers.append(1)


                        else:
                            fingers.append(0)

                    total= fingers.count(1)
                    # print('**********')
                    # print(total)
                    if total==5:
                        # print('brake') 
                        # st.text('Acclerate')
                        # original_title= 'Acc'
                        sh= "Acclerate"
                        draw(sh)
                    if total==2 or total==3:
                        sh= "Left"
                        draw(sh)
                    if total==4:
                        sh= "Right"
                        draw(sh)
                    if total==0:
                        sh= "Brake"
                        draw(sh)
                        # time.sleep(1) 
                        # original_title = '<p style="font-family:Arial Black; color:Blue; font-size: 30px;">{original_title}</p>'
                        # st.markdown(original_title, unsafe_allow_html=True) 
                    
                    mp_draw.draw_landmarks(image,hand_landmark,mp_hand.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
                #FPS Counter Logic
            currTime = time.time()
            fps = 1/ (currTime - prevTime)
            prevTime = currTime
            fingers=[]
            
            if record:
                out.write(image)
            image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
            
            kpi3_text.write(f"<h1 style='text-align: center; color:red; '>{width}</h1>", unsafe_allow_html=True)
            
            image = cv2.resize(image, (0,0), fx = 0.8, fy =0.8)
            image = image_resize(image = image, width = 320,height=360)
            stframe.image(image, channels = 'BGR', use_column_width=False)
            # else:
            #     output_video = open('output1.mp4','rb')
            #     out_bytes= output_video.read()
            #     st.video(out_bytes)
    st.subheader('Output Image')
    st.text('Video Processed')
    output_video = open('output1.mp4','rb')
    out_bytes= output_video.read()
    st.video(out_bytes)
    # st.markdown('Reload , if webpage hangs')
        
    vid.release()
    out.release()
    

            
        
        
        
            
        