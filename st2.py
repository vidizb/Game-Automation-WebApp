from bokeh.themes import theme
from numpy.core.records import record
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import time
from PIL import Image
import tempfile
from bokeh.models.widgets import Div
import streamlit as st
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Game Keys",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'test3.mp4'
prevTime=0
currTime=0 
tipIds= [4,8,12,16,20]
st.title('Game Keys with Hand Tracking Web-App')
url = 'https://github.com/shashankanand13monu/Game-Automation'

# ----------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_draw= mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_draw= mp.solutions.drawing_utils
mp_hand= mp.solutions.hands
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# ----------------------------------------------------------------------

t = st.empty()
def draw(str):
    t.markdown(f'<p style="font-family:Arial Black;color:#FF0686;font-size:28px;;">{str}</p>', unsafe_allow_html=True)

# ----------------------------------------------------------------------

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
# ----------------------------------------------------------------------
if st.sidebar.button('Github'):
    js = "window.open('https://github.com/shashankanand13monu/Game-Automation')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

# ----------------------------------------------------------------------

st.sidebar.title('Menu')
st.sidebar.subheader('Settings')

# ----------------------------------------------------------------------
@st.cache ()
# ----------------------------------------------------------------------

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
# ----------------------------------------------------------------------

app_mode= st.sidebar.selectbox('Choose the App Mode',
                               ['About App','Run on Image','Run On Video','Show Code'])

# ----------------------------------------------------------------------

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
    original_title= '''<pre style="font-family:Aku & Kamu; color:#FD0160 ; font-size: 24px;">
    Video Option will Experience Lag in  Browsers.
    If It's <strong>Lagging</strong> just <strong>Reload</strong> & Choose your option <strong>ASAP</strong> 
    eg: <strong>Choosing Max Hands</strong> or <strong>Using Webcam.</strong> 
    Webcam Will Take about <strong>20 Seconds</strong> to Load
    
    Update :
    1) I Discovered that you can't use Webcam Online,
    Because then it will try Access Server's Which i don't Own.
    
    2) Hand Marks are not showing online + Video freezes
    
    <strong>Solution :</strong>
    Go to main Streamlit WebApp Code & Run it Locally by typing
    <strong>streamlit run st2.py</strong>
    </pre>'''
    # st.markdown('''Video Option will Experience **Lag** in **Browsers**. If It's **Lagging** just **Reload** & Choose your option ASAP eg: **Choosing Max Hands** or **Using Webcam**. Webcam Will Take about **20 Seconds** to Load ''')
    st.markdown(original_title, unsafe_allow_html=True)
    
# ----------------------------------------------------------------------
    
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
        
    else:
        demo_image= DEMO_IMAGE
        
        image = 'demo.jpg'
        cap = cv2.imread('demo.jpg', cv2.IMREAD_UNCHANGED)
        image = cap.copy()

    # st.sidebar.text('Input Image')
    st.sidebar.subheader('Input Image')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    st.sidebar.image(image) 
    st.sidebar.subheader('Demo Images')
    
    st.sidebar.image('dddemo.jpg')  
    st.sidebar.image('Screenshot 2022-01-09 161732.png')
    st.sidebar.image('woman-showing-four-fingers-white-background-woman-showing-four-fingers-white-background-closeup-hand-134504006.jpg')

    st.sidebar.image('demo.jpg')
    hand_count =0
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cap = cv2.imread('demo.jpg', cv2.IMREAD_UNCHANGED)
   
    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=max_hands,
    min_detection_confidence=detection_confidence) as hands:
        
        hand_count+=1
  
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
        try:
            
            age_height, image_width, _ = image.shape
            annotated_image = image.copy()
            lmList=[]
            lmList2forModel=[]
            for hand_landmarks in results.multi_hand_landmarks:
            
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
            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
        except TypeError:
            original_title = '<p style="font-family:Arial Black; color:Red; font-size: 30px;">Sorry, No Hand Found!!</p>'
            st.markdown(original_title, unsafe_allow_html=True)

            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{0}</h1>", unsafe_allow_html=True)

      
        st.subheader('Output Image')
        annotated_image = cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)
        st.image(annotated_image, use_column_width=False)

# ----------------------------------------------------------------------
           
elif app_mode == 'Run On Video':
    
    st.set_option('deprecation.showfileUploaderEncoding',False)
    use_webcam = st.sidebar.button('Use Webcam')
    record= st.sidebar.checkbox("Record Video")
    
    if record:
        st.checkbox("Recording",value=True)
    
    
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

    
    max_hands= st.sidebar.number_input('Maximum Number of Hand',value=1,min_value=1,max_value=4)
    detection_confidence= st.sidebar.slider('Detection Confidence',min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence= st.sidebar.slider('Tracking Confidence Confidence',min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')
    
    st.subheader("Input Video")    
    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    #We get our input video here
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
     
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Frame Rate</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi1_text = st.markdown ("0")
    with kpi2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Detected Hands</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi2_text = st.markdown ("0")
    with kpi3:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Video Width</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi3_text = st.markdown("0")
    st.markdown ("<hr/>", unsafe_allow_html=True)
    st.subheader('Reload , if webpage hangs')
    st.markdown('---')
    st.subheader("Video Hangs in Browser works fine Locally like this : ")   
    data= 'sample.mp4'
    dat2= 'https://youtu.be/UT7gjebls4A'
    st.video(data, format="video/mp4", start_time=0)
    st.video(dat2)
    # video_file = open('sample.mp4', 'rb')
    # video_bytes = video_file.read()
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
    with mp_hand.Hands(max_num_hands=max_hands,min_detection_confidence=detection_confidence,
                       min_tracking_confidence=tracking_confidence) as hands:
    
        
        while vid.isOpened():
            
            i +=1
            ret, image = vid.read()
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
                    if total==5:
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
    st.subheader('Output Image')
    
    # st.video('streamlit-st2-2022-01-11-23-01-57.webm')
    # sample= 'streamlit-st2-2022-01-11-23-01-57.webm'
    # sampl= sample.read()
    # st.video(sampl)
    st.text('Video Processed')
    output_video = open('output1.mp4','rb')
    out_bytes= output_video.read()
    st.video(out_bytes)
    

    st.video(video_bytes) 
    vid.release()
    out.release()
 
# ---------------------------------------------------------------------- 
    
elif app_mode == 'Show Code':
    agree = st.checkbox('Show Only Game Code')

    if agree:
        st.subheader('Game Code') 
        uuu12='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=night-owl&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=import%2520cv2%250Aimport%2520mediapipe%2520as%2520mp%250Aimport%2520time%250Aimport%2520pyautogui%250Aimport%2520%2520pydirectinput%250Amp_draw%253D%2520mp.solutions.drawing_utils%250Amp_hand%253D%2520mp.solutions.hands%250Aimport%2520numpy%2520as%2520np%250A%250A%2523----------------------------------------------------------%250Avideo%253D%2520cv2.VideoCapture%280%29%250ApreviousTime%2520%253D%25200%250AcurrentTime%2520%253D%25200%250AtipIds%253D%2520%255B4%252C8%252C12%252C16%252C20%255D%250AclassNames%253D%2520%255B%27okay%27%252C%2520%27peace%27%252C%2520%27thumbs%2520up%27%252C%2520%27thumbs%2520down%27%252C%2520%27call%2520me%27%252C%2520%27stop%27%252C%2520%27rock%27%252C%2520%27live%2520long%27%252C%2520%27fist%27%252C%2520%27smile%27%255D%250Afrom%2520directkeys%2520import%2520ReleaseKey%252CPressKey%252C%2520W%252C%2520A%252C%2520S%252C%2520D%250A%250Awith%2520mp_hand.Hands%28max_num_hands%253D1%252Cmin_detection_confidence%253D0.5%252Cmin_tracking_confidence%253D0.5%29%2520as%2520hands%253A%250A%250A%2520%2520%2520%2520while%2520True%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520ret%252C%2520image%253D%2520video.read%28%29%250A%2520%2520%2520%2520%2520%2520%2520%2520image%253D%2520cv2.cvtColor%28image%252Ccv2.COLOR_BGR2RGB%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520image.flags.writeable%253DFalse%2520%2523To%2520improve%2520performance%252C%2520optionally%2520mark%2520the%2520image%2520as%2520not%2520writeable%2520to%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520pass%2520by%2520reference%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520results%253D%2520hands.process%28image%29%250A%2520%2520%2520%2520%2520%2520%2520%2520image.flags.writeable%253DTrue%250A%2520%2520%2520%2520%2520%2520%2520%2520image%253D%2520cv2.cvtColor%28image%252Ccv2.COLOR_RGB2BGR%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520Calculating%2520the%2520FPS%250A%2520%2520%2520%2520%2520%2520%2520%2520currentTime%2520%253D%2520time.time%28%29%250A%2520%2520%2520%2520%2520%2520%2520%2520fps%2520%253D%25201%2520%252F%2520%28currentTime-previousTime%29%250A%2520%2520%2520%2520%2520%2520%2520%2520previousTime%2520%253D%2520currentTime%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520lmList%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520lmList2forModel%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520Displaying%2520FPS%2520on%2520the%2520image%250A%2520%2520%2520%2520%2520%2520%2520%2520cv2.putText%28image%252C%2520str%28int%28fps%29%29%252B%2522%2520FPS%2522%252C%2520%2810%252C%252070%29%252C%2520cv2.FONT_HERSHEY_COMPLEX%252C%25201%252C%2520%280%252C255%252C0%29%252C%25202%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520if%2520results.multi_hand_landmarks%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520hand_landmark%2520in%2520results.multi_hand_landmarks%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520myHands%253Dresults.multi_hand_landmarks%255B0%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520id%252Clm%2520in%2520enumerate%28myHands.landmark%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520h%252Cw%252Cc%253Dimage.shape%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520cx%252Ccy%253Dint%28lm.x*w%29%252C%2520int%28lm.y*h%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList.append%28%255Bid%252Ccx%252Ccy%255D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList2forModel.append%28%255Bcx%252Ccy%255D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.draw_landmarks%28image%252Chand_landmark%252Cmp_hand.HAND_CONNECTIONS%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C0%252C255%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C0%252C255%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C255%252C0%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520"
  style="width: 1024px; height: 473px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu13='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=night-owl&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=%2523---------------------------------------------------------------------------------------------%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520len%28lmList%29%21%253D0%253A%2520%2523%2520No%2520Hand%2520in%2520BackGround%250A%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520lmList%255BtipIds%255B0%255D%255D%255B1%255D%2520%253E%2520lmList%255BtipIds%255B0%255D-1%255D%255B1%255D%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%280%29%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520id%2520in%2520range%281%252C5%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520lmList%255BtipIds%255Bid%255D%255D%255B2%255D%2520%253C%2520lmList%255BtipIds%255Bid%255D-1%255D%255B2%255D%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%280%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520total%253D%2520fingers.count%281%29%250A%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D0%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520print%28%2522Brake%2522%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28W%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28A%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28S%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520PressKey%28S%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520time.sleep%282%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28S%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520elif%2520total%253D%253D5%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520print%28%2522GAS%2522%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28S%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28A%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28S%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520PressKey%28W%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520time.sleep%282%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28W%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520pydirectinput.keyUp%28%27s%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520pydirectinput.keyDown%28%27w%27%29%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520elif%2520total%253D%253D4%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520print%28%2522Right%2522%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28W%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520PressKey%28D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520time.sleep%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520elif%2520total%253D%253D2%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520print%28%2522LEFT%2522%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520PressKey%28A%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520time.sleep%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ReleaseKey%28A%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520cv2.imshow%28%2522Frame%2522%252Cimage%29%250A%2520%2520%2520%2520%2520%2520%2520%2520k%253D%2520cv2.waitKey%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520if%2520k%253D%253Dord%28%27q%27%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520break%250Avideo.release%28%29%250Acv2.destroyAllWindows%28%29%250A%2520"
  style="width: 1024px; height: 473px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        st.components.v1.html(uuu12,height=470,scrolling=True)
        st.components.v1.html(uuu13,height=470,scrolling=True)
        
        
        # code = '''def hello():
        # print("Helliiiiiiiio, Streamlit!")'''
        # st.code(code, language='python')
        
    else:
        st.subheader('Streamlit Code')
        # 11898
        uuu= '''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17.5px&lh=133%25&si=false&es=2x&wm=false&code=from%2520bokeh.themes%2520import%2520theme%250Afrom%2520numpy.core.records%2520import%2520record%250Aimport%2520pandas%2520as%2520pd%250Aimport%2520numpy%2520as%2520np%250Aimport%2520cv2%250Aimport%2520mediapipe%2520as%2520mp%250Aimport%2520time%250Afrom%2520PIL%2520import%2520Image%250Aimport%2520tempfile%250Afrom%2520bokeh.models.widgets%2520import%2520Div%250Aimport%2520streamlit%2520as%2520st%250A%2523%2520---------------------------------------------------------------------%250Ast.set_page_config%28%250A%2520%2520%2520%2520page_title%253D%2522Game%2520Keys%2522%252C%250A%2520%2520%2520%2520page_icon%253D%2522%25F0%259F%258E%25AE%2522%252C%250A%2520%2520%2520%2520layout%253D%2522wide%2522%252C%250A%2520%2520%2520%2520initial_sidebar_state%253D%2522expanded%2522%250A%29%250A%250A%2523%2520---------------------------------------------------------------------%250A%250ADEMO_IMAGE%2520%253D%2520%27demo.jpg%27%250ADEMO_VIDEO%2520%253D%2520%27test3.mp4%27%250AprevTime%253D0%250AcurrTime%253D0%2520%250AtipIds%253D%2520%255B4%252C8%252C12%252C16%252C20%255D%250Ast.title%28%27Game%2520Keys%2520with%2520Hand%2520Tracking%2520Web-App%27%29%250Aurl%2520%253D%2520%27https%253A%252F%252Fgithub.com%252Fshashankanand13monu%252FGame-Automation%27%250A%250A%2523%2520----------------------------------------------------------------------%250A%250Amp_drawing%2520%253D%2520mp.solutions.drawing_utils%250Amp_draw%253D%2520mp.solutions.drawing_utils%250Amp_face_mesh%2520%253D%2520mp.solutions.face_mesh%250Amp_draw%253D%2520mp.solutions.drawing_utils%250Amp_hand%253D%2520mp.solutions.hands%250Amp_hands%2520%253D%2520mp.solutions.hands%250Amp_drawing_styles%2520%253D%2520mp.solutions.drawing_styles%250A%250A%2523%2520----------------------------------------------------------------------%250A%250At%2520%253D%2520st.empty%28%29%250Adef%2520draw%28str%29%253A%250A%2520%2520%2520%2520t.markdown%28f%27%253Cp%2520style%253D%2522font-family%253AArial%2520Black%253Bcolor%253A%2523FF0686%253Bfont-size%253A28px%253B%253B%2522%253E%257Bstr%257D%253C%252Fp%253E%27%252C%2520unsafe_allow_html%253DTrue%29%250A%250A%2523%2520----------------------------------------------------------------------%250A%250Ast.markdown%28%250A%2520%2520%2520%2520%2522%2522%2522%250A%2520%2520%2520%2520%253Cstyle%253E%250A%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522true%2522%255D%2520%253E%2520div%253Afirst-child%257B%250Awidth%253A%2520350px%250A%257D%250A%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522false%2522%255D%2520%253E%2520div%253Afirst-child%257B%250Awidth%253A%2520350px%250Amargin-left%253A%2520-350px%250A%253C%252Fstyle%253E%250A%2520%2520%2520%2520%2522%2522%2522%252Cunsafe_allow_html%253DTrue%252C%29%250A%2523%2520----------------------------------------------------------------------%250Aif%2520st.sidebar.button%28%27Github%27%29%253A%250A%2520%2520%2520%2520js%2520%253D%2520%2522window.open%28%27https%253A%252F%252Fgithub.com%252Fshashankanand13monu%252FGame-Automation%27%29%2522%2520%2520%2523%2520New%2520tab%2520or%2520window%250A%2520%2520%2520%2520%2523%2520js%2520%253D%2520%2522window.location.href%2520%253D%2520%27https%253A%252F%252Fwww.streamlit.io%252F%27%2522%2520%2520%2523%2520Current%2520tab%250A%2520%2520%2520%2520html%2520%253D%2520%27%253Cimg%2520src%2520onerror%253D%2522%257B%257D%2522%253E%27.format%28js%29%250A%2520%2520%2520%2520div%2520%253D%2520Div%28text%253Dhtml%29%250A%2520%2520%2520%2520st.bokeh_chart%28div%29"
  style="width: 1024px; height: 1500px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu2= '''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17.5px&lh=133%25&si=false&es=2x&wm=false&code=st.sidebar.title%28%27Menu%27%29%250Ast.sidebar.subheader%28%27Settings%27%29%250A%250A%2523%2520----------------------------------------------------------------------%250A%2540st.cache%2520%28%29%250A%2523%2520----------------------------------------------------------------------%250A%250Adef%2520image_resize%28image%252C%2520width%253DNone%252C%2520height%253DNone%252C%2520inter%2520%253Dcv2.INTER_AREA%29%253A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520dim%2520%253D%2520None%250A%2520%2520%2520%2520%28h%2520%252Cw%29%2520%253D%2520image.shape%255B%253A2%255D%250A%2520%2520%2520%2520if%2520width%2520is%2520None%2520and%2520height%2520is%2520None%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520return%2520image%250A%2520%2520%2520%2520if%2520width%2520is%2520None%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520r%253D%2520width%252Ffloat%28w%29%250A%2520%2520%2520%2520%2520%2520%2520%2520dim%2520%253D%2520%28int%28w*r%29%252C%2520height%29%250A%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520r%2520%253D%2520width%252Ffloat%28w%29%250A%2520%2520%2520%2520%2520%2520%2520%2520dim%2520%253D%2520%28width%252C%2520int%28h*r%29%29%250A%2520%2520%2520%2520%2523resize%2520the%2520image%250A%2520%2520%2520%2520resized%2520%253Dcv2.resize%28image%252C%2520dim%2520%252Cinterpolation%253Dinter%29%250A%2520%2520%2520%2520return%2520resized%250A%2523%2520----------------------------------------------------------------------%250A%250Aapp_mode%253D%2520st.sidebar.selectbox%28%27Choose%2520the%2520App%2520Mode%27%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%255B%27About%2520App%27%252C%27Run%2520on%2520Image%27%252C%27Run%2520On%2520Video%27%252C%27Show%2520Code%27%255D%29%250A%250A%2523%2520----------------------------------------------------------------------%250A%250Aif%2520app_mode%253D%253D%2520%27About%2520App%27%253A%250A%2520%2520%2520%2520st.markdown%28%27App%2520Made%2520using%2520**Mediapipe**%2520%2526%2520**Open%2520CV**%27%29%250A%250A%2520%2520%2520%2520st.markdown%28%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%253Cstyle%253E%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522true%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520%257D%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522false%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520margin-left%253A%2520-350px%250A%2520%2520%2520%2520%253C%252Fstyle%253E%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%252Cunsafe_allow_html%253DTrue%252C%29%250A%250A%2520%2520%2520%2520st.markdown%28%27%27%27%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520Tutorial%2520%255Cn%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%27%27%27%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%29%250A%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cpre%2520style%253D%2522font-family%253AAku%2520%2526%2520Kamu%253B%2520color%253A%2523FD0177%253B%2520font-size%253A%252025px%253Bfont-weight%253ABold%2522%253E%25F0%259F%2595%25B9%25EF%25B8%258F%2520W-%25205%2520Fingers%2520%2520%25F0%259F%2595%25B9%25EF%25B8%258F%2520A-%25202%2520or%25203%2520Fingers%253C%252Fpre%253E%27%250A%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cpre%2520style%253D%2522font-family%253AAku%2520%2526%2520Kamu%253B%2520color%253A%2523FD0177%253B%2520font-size%253A%252025px%253Bfont-weight%253ABold%253B%2522%253E%25F0%259F%2595%25B9%25EF%25B8%258F%2520S-%2520Fist%2520%2520%2520%2520%2520%2520%2520%25F0%259F%2595%25B9%25EF%25B8%258F%2520D-%25204%2520Fingers%253C%252Fpre%253E%27%250A%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2523%2520st.subheader%28%27%27%27W%2520-%25205%2520Fingers%27%27%27%29%250A%2520%2520%2520%2520%2523%2520st.subheader%28%27S%2520-%2520Fist%255Cn%2520A%2520-%25202%2520or%25203%2520Fingers%255Cn%2520D%2520-%25204%2520Fingers%27%29%250A%2520%2520%2520%2520st.image%28%27wsad.jpg%27%252Cwidth%253D200%29%250A%2520"
  style="width: 1024px; height: 1500px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu3='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17.5px&lh=133%25&si=false&es=2x&wm=false&code=%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cpre%2520style%253D%2522font-family%253AAku%2520%2526%2520Kamu%253B%2520color%253A%2523FD0101%2520%253B%2520font-size%253A%252028px%253Bfont-weight%253ABold%2522%253E*NOTE%253C%252Fpre%253E%27%250A%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520original_title%253D%2520%27%27%27%253Cpre%2520style%253D%2522font-family%253AAku%2520%2526%2520Kamu%253B%2520color%253A%2523270F40%253B%2520font-size%253A%252024px%253B%2522%253E%250A%2520%2520%2520%2520Video%2520Option%2520will%2520Experience%2520Lag%2520in%2520%2520Browsers.%250A%2520%2520%2520%2520If%2520It%27s%2520%253Cstrong%253ELagging%253C%252Fstrong%253E%2520just%2520%253Cstrong%253EReload%253C%252Fstrong%253E%2520%2526%2520Choose%2520your%2520option%2520%253Cstrong%253EASAP%253C%252Fstrong%253E%2520%250A%2520%2520%2520%2520eg%253A%2520%253Cstrong%253EChoosing%2520Max%2520Hands%253C%252Fstrong%253E%2520or%2520%253Cstrong%253EUsing%2520Webcam.%253C%252Fstrong%253E%2520%250A%2520%2520%2520%2520Webcam%2520Will%2520Take%2520about%2520%253Cstrong%253E20%2520Seconds%253C%252Fstrong%253E%2520to%2520Load%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520Update%2520%253A%250A%2520%2520%2520%25201%29%2520I%2520Discovered%2520that%2520you%2520can%27t%2520use%2520Webcam%2520Online%252C%250A%2520%2520%2520%2520Because%2520then%2520it%2520will%2520try%2520Access%2520Server%27s%2520Which%2520i%2520don%27t%2520Own.%250A%2520%2520%2520%2520%250A%2520%2520%2520%25202%29%2520Hand%2520Marks%2520are%2520not%2520showing%2520online%2520%252B%2520Video%2520freezes%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%253Cstrong%253ESolution%2520%253A%253C%252Fstrong%253E%250A%2520%2520%2520%2520Go%2520to%2520main%2520Streamlit%2520WebApp%2520Code%2520%2526%2520Run%2520it%2520Locally%2520by%2520typing%250A%2520%2520%2520%2520%253Cstrong%253Estreamlit%2520run%2520st2.py%253C%252Fstrong%253E%250A%2520%2520%2520%2520%253C%252Fpre%253E%27%27%27%250A%2520%2520%2520%2520%2523%2520st.markdown%28%27%27%27Video%2520Option%2520will%2520Experience%2520**Lag**%2520in%2520**Browsers**.%2520If%2520It%27s%2520**Lagging**%2520just%2520**Reload**%2520%2526%2520Choose%2520your%2520option%2520ASAP%2520eg%253A%2520**Choosing%2520Max%2520Hands**%2520or%2520**Using%2520Webcam**.%2520Webcam%2520Will%2520Take%2520about%2520**20%2520Seconds**%2520to%2520Load%2520%27%27%27%29%250A%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%250A%2523%2520----------------------------------------------------------------------%250A%2520%2520%2520%2520%250Aelif%2520app_mode%2520%253D%253D%2520%27Run%2520on%2520Image%27%253A%250A%2520%2520%2520%2520drawing_spec%2520%253D%2520mp_drawing.DrawingSpec%28thickness%253D2%252C%2520circle_radius%253D1%29%250A%2520%2520%2520%2520st.sidebar.markdown%2520%28%27---------%27%2520%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.markdown%28%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%253Cstyle%253E%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522true%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520%257D%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522false%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520margin-left%253A%2520-350px%250A%2520%2520%2520%2520%253C%252Fstyle%253E%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%252Cunsafe_allow_html%253DTrue%252C%29%250A%250A%2520%2520%2520%2520%2523%2520st.markdown%28%2522**Detected%2520Hands**%2522%29%250A%2520%2520%2520%2520st.header%28%2522**%2520%2520%2520Detected%2520Hands%2520%2520%2520**%2522%29%250A%2520%2520%2520%2520kpi1_text%2520%253D%2520st.markdown%28%25220%2522%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520max_hands%253D%2520st.sidebar.number_input%28%27Maximum%2520Number%2520of%2520Hands%27%252Cvalue%253D2%252Cmin_value%253D1%252Cmax_value%253D4%29%250A%2520%2520%2520%2520%2523%2520st.sidebar%28%27---%27%29%250A%2520%2520%2520"
  style="width: 1024px; height: 1500px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu4='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17.5px&lh=133%25&si=false&es=2x&wm=false&code=%2509detection_confidence%253D%2520st.sidebar.slider%28%27Detection%2520Confidence%27%252Cmin_value%253D0.0%252Cmax_value%253D1.0%252Cvalue%253D0.5%29%250A%2520%2520%2520%2520st.sidebar.markdown%28%27---%27%29%250A%2520%2520%2520%2520IMAGE_FILE%253D%255B%255D%250A%2520%2520%2520%2520count%253D0%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520img_file_buffer%2520%253D%2520st.sidebar.file_uploader%28%2522Upload%2520an%2520Image%2522%252C%2520type%253D%255B%2522jpg%2522%252C%2522jpeg%2522%252C%2520%2522png%2522%255D%29%250A%2520%2520%2520%2520if%2520img_file_buffer%2520is%2520not%2520None%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520file_bytes%2520%253D%2520np.asarray%28bytearray%28img_file_buffer.read%28%29%29%252C%2520dtype%253Dnp.uint8%29%250A%2520%2520%2520%2520%2520%2520%2520%2520opencv_image%2520%253D%2520cv2.imdecode%28file_bytes%252C%25201%29%250A%2520%2520%2520%2520%2520%2520%2520%2520image%2520%253D%2520opencv_image.copy%28%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520demo_image%253D%2520DEMO_IMAGE%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520image%2520%253D%2520%27demo.jpg%27%250A%2520%2520%2520%2520%2520%2520%2520%2520cap%2520%253D%2520cv2.imread%28%27demo.jpg%27%252C%2520cv2.IMREAD_UNCHANGED%29%250A%2520%2520%2520%2520%2520%2520%2520%2520image%2520%253D%2520cap.copy%28%29%250A%250A%2520%2520%2520%2520%2523%2520st.sidebar.text%28%27Input%2520Image%27%29%250A%2520%2520%2520%2520st.sidebar.subheader%28%27Input%2520Image%27%29%250A%2520%2520%2520%2520image%2520%253D%2520cv2.cvtColor%28image%252Ccv2.COLOR_BGR2RGB%29%250A%2520%2520%2520%2520st.sidebar.image%28image%29%2520%250A%2520%2520%2520%2520st.sidebar.subheader%28%27Demo%2520Images%27%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.sidebar.image%28%27dddemo.jpg%27%29%2520%2520%250A%2520%2520%2520%2520st.sidebar.image%28%27Screenshot%25202022-01-09%2520161732.png%27%29%250A%2520%2520%2520%2520st.sidebar.image%28%27woman-showing-four-fingers-white-background-woman-showing-four-fingers-white-background-closeup-hand-134504006.jpg%27%29%250A%250A%2520%2520%2520%2520st.sidebar.image%28%27demo.jpg%27%29%250A%2520%2520%2520%2520hand_count%2520%253D0%250A%2520%2520%2520%2520image%2520%253D%2520cv2.cvtColor%28image%252Ccv2.COLOR_RGB2BGR%29%250A%2520%2520%2520%2520cap%2520%253D%2520cv2.imread%28%27demo.jpg%27%252C%2520cv2.IMREAD_UNCHANGED%29%250A%2520%2520%2520%250A%2520%2520%2520%2520with%2520mp_hands.Hands%28%250A%2520%2520%2520%2520static_image_mode%253DTrue%252C%250A%2520%2520%2520%2520max_num_hands%253Dmax_hands%252C%250A%2520%2520%2520%2520min_detection_confidence%253Ddetection_confidence%29%2520as%2520hands%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520hand_count%252B%253D1%250A%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520results%2520%253D%2520hands.process%28cv2.cvtColor%28image%252C%2520cv2.COLOR_BGR2RGB%29%29%250A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520try%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520age_height%252C%2520image_width%252C%2520_%2520%253D%2520image.shape%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%2520%253D%2520image.copy%28%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList2forModel%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520hand_landmarks%2520in%2520results.multi_hand_landmarks%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520myHands%253Dresults.multi_hand_landmarks%255B0%255D%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520id%252Clm%2520in%2520enumerate%28myHands.landmark%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520h%252Cw%252Cc%253Dimage.shape%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520cx%252Ccy%253Dint%28lm.x*w%29%252C%2520int%28lm.y*h%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList.append%28%255Bid%252Ccx%252Ccy%255D%29"
  style="width: 1024px; height: 1470px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu5='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17.5px&lh=133%25&si=false&es=2x&wm=false&code=%2509%2509%2509%2509%2509lmList2forModel.append%28%255Bcx%252Ccy%255D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520lmList%255BtipIds%255B0%255D%255D%255B1%255D%2520%253C%2520lmList%255BtipIds%255B0%255D-1%255D%255B1%255D%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%280%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520id%2520in%2520range%281%252C5%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520lmList%255BtipIds%255Bid%255D%255D%255B2%255D%2520%253C%2520lmList%255BtipIds%255Bid%255D-1%255D%255B2%255D%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%280%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520total%253D%2520fingers.count%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D5%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522font-family%253AArial%2520Black%253B%2520color%253A%2523FD0177%253B%2520font-size%253A%252030px%253B%2522%253EAcclerate%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520st.markdown%28%27---%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.imread%28%27istockphoto-1179377734-612x612.jpg%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.resize%28overlay%252C%28100%252C100%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_offset%2520%253D%252080%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_offset%2520%253D%252010%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_end%2520%253D%2520x_offset%2520%252B%2520overlay.shape%255B1%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_end%2520%253D%2520y_offset%2520%252B%2520overlay.shape%255B0%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%255By_offset%253Ay_end%252Cx_offset%253Ax_end%255D%2520%253D%2520overlay%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D4%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520st.text%28%27Right%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522font-family%253AArial%2520Black%253B%2520color%253A%2523FD0177%253B%2520font-size%253A%252030px%253B%2522%253ERight%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.imread%28%27istockphoto-1179377734-612x612%2520%284%29.jpg%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.resize%28overlay%252C%28100%252C100%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_offset%2520%253D%2520120%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_offset%2520%253D%252050%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_end%2520%253D%2520x_offset%2520%252B%2520overlay.shape%255B1%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_end%2520%253D%2520y_offset%2520%252B%2520overlay.shape%255B0%255D%250A%2520%2520%2520%2520%2520%2520%2520"
  style="width: 1024px; height: 4328px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu6='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17.5px&lh=133%25&si=false&es=2x&wm=false&code=%2509%2509%2509%2509%2509annotated_image%255By_offset%253Ay_end%252Cx_offset%253Ax_end%255D%2520%253D%2520overlay%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D2%2520or%2520total%253D%253D3%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520st.text%28%27Left%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522font-family%253AArial%2520Black%253B%2520color%253A%2523FD0177%253B%2520font-size%253A%252030px%253B%2522%253ELeft%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.imread%28%27istockphoto-1179377734-612x612%2520%281%29.jpg%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.resize%28overlay%252C%28100%252C100%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_offset%2520%253D%252050%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_offset%2520%253D%252050%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_end%2520%253D%2520x_offset%2520%252B%2520overlay.shape%255B1%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_end%2520%253D%2520y_offset%2520%252B%2520overlay.shape%255B0%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%255By_offset%253Ay_end%252Cx_offset%253Ax_end%255D%2520%253D%2520overlay%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D0%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523%2520st.text%28%27Brake%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522font-family%253AArial%2520Black%253B%2520color%253A%2523FD0177%253B%2520font-size%253A%252030px%253B%2522%253EBrake%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.imread%28%27istockphoto-1179377734-612x612%2520%283%29.jpg%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520overlay%2520%253D%2520cv2.resize%28overlay%252C%28100%252C100%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_offset%2520%253D%252050%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_offset%2520%253D%252090%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520x_end%2520%253D%2520x_offset%2520%252B%2520overlay.shape%255B1%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520y_end%2520%253D%2520y_offset%2520%252B%2520overlay.shape%255B0%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%255By_offset%253Ay_end%252Cx_offset%253Ax_end%255D%2520%253D%2520overlay%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_drawing.draw_landmarks%28%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520hand_landmarks%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_hands.HAND_CONNECTIONS%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C0%252C255%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C255%252C0%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520kpi1_text.write%28f%2522%253Ch1%2520style%253D%27text-align%253A%2520center%253B%2520color%253Ared%253B%2520%27%253E%257Bhand_count%257D%253C%252Fh1%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520except%2520TypeError%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522font-family%253AArial%2520Black%253B%2520color%253ARed%253B%2520font-size%253A%252030px%253B%2522%253ESorry%252C%2520No%2520Hand%2520Found%21%21%253C"
  style="width: 923px; height: 5304px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu7='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=%2509%2509%2509st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520kpi1_text.write%28f%2522%253Ch1%2520style%253D%27text-align%253A%2520center%253B%2520color%253Ared%253B%2520%27%253E%257B0%257D%253C%252Fh1%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%250A%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520st.subheader%28%27Output%2520Image%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%2520%253D%2520cv2.cvtColor%28annotated_image%252Ccv2.COLOR_BGR2RGB%29%250A%2520%2520%2520%2520%2520%2520%2520%2520st.image%28annotated_image%252C%2520use_column_width%253DFalse%29%250A%250A%2523%2520----------------------------------------------------------------------%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250Aelif%2520app_mode%2520%253D%253D%2520%27Run%2520On%2520Video%27%253A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.set_option%28%27deprecation.showfileUploaderEncoding%27%252CFalse%29%250A%2520%2520%2520%2520use_webcam%2520%253D%2520st.sidebar.button%28%27Use%2520Webcam%27%29%250A%2520%2520%2520%2520record%253D%2520st.sidebar.checkbox%28%2522Record%2520Video%2522%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520if%2520record%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520st.checkbox%28%2522Recording%2522%252Cvalue%253DTrue%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.markdown%28%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%253Cstyle%253E%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522true%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520%257D%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522false%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520margin-left%253A%2520-350px%250A%2520%2520%2520%2520%253C%252Fstyle%253E%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%252Cunsafe_allow_html%253DTrue%252C%29%250A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520max_hands%253D%2520st.sidebar.number_input%28%27Maximum%2520Number%2520of%2520Hand%27%252Cvalue%253D1%252Cmin_value%253D1%252Cmax_value%253D4%29%250A%2520%2520%2520%2520detection_confidence%253D%2520st.sidebar.slider%28%27Detection%2520Confidence%27%252Cmin_value%253D0.0%252Cmax_value%253D1.0%252Cvalue%253D0.5%29%250A%2520%2520%2520%2520tracking_confidence%253D%2520st.sidebar.slider%28%27Tracking%2520Confidence%2520Confidence%27%252Cmin_value%253D0.0%252Cmax_value%253D1.0%252Cvalue%253D0.5%29%250A%2520%2520%2520%2520st.sidebar.markdown%28%27---%27%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.subheader%28%2522Input%2520Video%2522%29%2520%2520%2520%2520%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520stframe%2520%253D%2520st.empty%28%29%250A%2520%2520%2520%2520video_file_buffer%2520%253D%2520st.sidebar.file_uploader%28%2522Upload%2520a%2520Video%2522%252C%2520type%253D%255B%27mp4%27%252C%2520%27mov%27%252C%2520%27avi%27%252C%2520%27asf%27%252C%2520%27m4v%27%255D%29%250A%2520%2520%2520%2520tffile%2520%253D%2520tempfile.NamedTemporaryFile%28delete%253DFalse%29%250A%2520%2520%2520%2520%2523We%2520get%2520our%2520input%2520video%2520here%250A%2520%2520%2520%2520if%2520not%2520video_file_buffer%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520if%2520use_webcam%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520vid%2520%253D%2520cv2.VideoCapture%280%29%250A%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520vid%2520%253D%2520cv2.VideoCapture%28DEMO_VIDEO%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520tffile.name%2520%253D%2520DEMO_VIDEO%250A%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520tffile.write%28video_file_buffer.read%28%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520vid%2520%253D%2520cv2.VideoCapture%28tffile.name%29%250A%2520%2520%2520%2520width%2520%253D%2520int%28vid.get%28cv2.CAP_PROP_FRAME_WIDTH%29%29%250A%2520%2520%2520%2520height%2520%253D%2520int%28vid.get%28cv2.CAP_"
  style="width: 1024px; height: 4373px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''
        
        uuu8='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=%2509%2509%2509st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520kpi1_text.write%28f%2522%253Ch1%2520style%253D%27text-align%253A%2520center%253B%2520color%253Ared%253B%2520%27%253E%257B0%257D%253C%252Fh1%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%250A%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520st.subheader%28%27Output%2520Image%27%29%250A%2520%2520%2520%2520%2520%2520%2520%2520annotated_image%2520%253D%2520cv2.cvtColor%28annotated_image%252Ccv2.COLOR_BGR2RGB%29%250A%2520%2520%2520%2520%2520%2520%2520%2520st.image%28annotated_image%252C%2520use_column_width%253DFalse%29%250A%250A%2523%2520----------------------------------------------------------------------%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250Aelif%2520app_mode%2520%253D%253D%2520%27Run%2520On%2520Video%27%253A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.set_option%28%27deprecation.showfileUploaderEncoding%27%252CFalse%29%250A%2520%2520%2520%2520use_webcam%2520%253D%2520st.sidebar.button%28%27Use%2520Webcam%27%29%250A%2520%2520%2520%2520record%253D%2520st.sidebar.checkbox%28%2522Record%2520Video%2522%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520if%2520record%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520st.checkbox%28%2522Recording%2522%252Cvalue%253DTrue%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.markdown%28%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%253Cstyle%253E%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522true%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520%257D%250A%2520%2520%2520%2520%255Bdata-testid%253D%2522stSidebar%2522%255D%255Baria-expanded%253D%2522false%2522%255D%2520%253E%2520div%253Afirst-child%257B%250A%2520%2520%2520%2520width%253A%2520350px%250A%2520%2520%2520%2520margin-left%253A%2520-350px%250A%2520%2520%2520%2520%253C%252Fstyle%253E%250A%2520%2520%2520%2520%2520%2520%2520%2520%2522%2522%2522%252Cunsafe_allow_html%253DTrue%252C%29%250A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520max_hands%253D%2520st.sidebar.number_input%28%27Maximum%2520Number%2520of%2520Hand%27%252Cvalue%253D1%252Cmin_value%253D1%252Cmax_value%253D4%29%250A%2520%2520%2520%2520detection_confidence%253D%2520st.sidebar.slider%28%27Detection%2520Confidence%27%252Cmin_value%253D0.0%252Cmax_value%253D1.0%252Cvalue%253D0.5%29%250A%2520%2520%2520%2520tracking_confidence%253D%2520st.sidebar.slider%28%27Tracking%2520Confidence%2520Confidence%27%252Cmin_value%253D0.0%252Cmax_value%253D1.0%252Cvalue%253D0.5%29%250A%2520%2520%2520%2520st.sidebar.markdown%28%27---%27%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.subheader%28%2522Input%2520Video%2522%29%2520%2520%2520%2520%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520stframe%2520%253D%2520st.empty%28%29%250A%2520%2520%2520%2520video_file_buffer%2520%253D%2520st.sidebar.file_uploader%28%2522Upload%2520a%2520Video%2522%252C%2520type%253D%255B%27mp4%27%252C%2520%27mov%27%252C%2520%27avi%27%252C%2520%27asf%27%252C%2520%27m4v%27%255D%29%250A%2520%2520%2520%2520tffile%2520%253D%2520tempfile.NamedTemporaryFile%28delete%253DFalse%29%250A%2520%2520%2520%2520%2523We%2520get%2520our%2520input%2520video%2520here%250A%2520%2520%2520%2520if%2520not%2520video_file_buffer%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520if%2520use_webcam%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520vid%2520%253D%2520cv2.VideoCapture%280%29%250A%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520vid%2520%253D%2520cv2.VideoCapture%28DEMO_VIDEO%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520tffile.name%2520%253D%2520DEMO_VIDEO%250A%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520tffile.write%28video_file_buffer.read%28%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520vid%2520%253D%2520cv2.VideoCapture%28tffile.name%29%250A%2520%2520%2520%2520width%2520%253D%2520int%28vid.get%28cv2.CAP_PROP_FRAME_WIDTH%29%29%250A%2520%2520%2520%2520height%2520%253D%2520int%28vid.get%28cv2.CAP_PROP_FRAME_HEIGHT%29%29"
  style="width: 1024px; height: 1400px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu9='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=%250A%2520%2520%2520%2520fps_input%2520%253D%2520int%28vid.get%28cv2.CAP_PROP_FPS%29%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%2523Recording%2520Part%250A%2520%2520%2520%2520codec%2520%253D%2520cv2.VideoWriter_fourcc%28%27V%27%252C%2520%27P%27%252C%2520%270%27%252C%279%27%29%250A%2520%2520%2520%2520out%253D%2520cv2.VideoWriter%28%27output.mp4%27%252Ccodec%252Cfps_input%252C%28width%252Cheight%29%29%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520st.sidebar.text%28%27Input%2520Video%27%29%250A%2520%2520%2520%2520st.sidebar.video%28tffile.name%29%250A%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520fps%2520%253D%25200%250A%2520%2520%2520%2520i%2520%253D%25200%250A%2520%2520%2520%2520drawing_spec%2520%253D%2520mp_drawing.DrawingSpec%28thickness%253D2%252C%2520circle_radius%253D1%29%250A%2520%2520%2520%2520kpi1%252C%2520kpi2%252C%2520kpi3%2520%253D%2520st.columns%283%29%250A%2520%2520%2520%2520with%2520kpi1%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522text-align%253A%2520center%253B%2520font-size%253A%252020px%253B%2522%253E%253Cstrong%253EFrame%2520Rate%253C%252Fstrong%253E%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520kpi1_text%2520%253D%2520st.markdown%2520%28%25220%2522%29%250A%2520%2520%2520%2520with%2520kpi2%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522text-align%253A%2520center%253B%2520font-size%253A%252020px%253B%2522%253E%253Cstrong%253EDetected%2520Hands%253C%252Fstrong%253E%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520kpi2_text%2520%253D%2520st.markdown%2520%28%25220%2522%29%250A%2520%2520%2520%2520with%2520kpi3%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520original_title%2520%253D%2520%27%253Cp%2520style%253D%2522text-align%253A%2520center%253B%2520font-size%253A%252020px%253B%2522%253E%253Cstrong%253EVideo%2520Width%253C%252Fstrong%253E%253C%252Fp%253E%27%250A%2520%2520%2520%2520%2520%2520%2520%2520st.markdown%28original_title%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520kpi3_text%2520%253D%2520st.markdown%28%25220%2522%29%250A%2520%2520%2520%2520st.markdown%2520%28%2522%253Chr%252F%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520st.subheader%28%27Reload%2520%252C%2520if%2520webpage%2520hangs%27%29%250A%2520%2520%2520%2520drawing_spec%2520%253D%2520mp_drawing.DrawingSpec%28thickness%253D1%252C%2520circle_radius%253D1%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520with%2520mp_hand.Hands%28max_num_hands%253Dmax_hands%252Cmin_detection_confidence%253Ddetection_confidence%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520min_tracking_confidence%253Dtracking_confidence%29%2520as%2520hands%253A%250A%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520while%2520vid.isOpened%28%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520i%2520%252B%253D1%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520ret%252C%2520image%2520%253D%2520vid.read%28%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520not%2520ret%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520continue%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520image.flags.writeable%253DFalse%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520results%253D%2520hands.process%28image%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520image.flags.writeable%253DTrue%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520image%253D%2520cv2.cvtColor%28image%252Ccv2.COLOR_RGB2BGR%29%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList2forModel%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520hand_count%253D0%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520results.multi_hand_landmarks%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520hand_landmark%2520in%2520results.multi_hand_landmarks%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520"
  style="width: 1024px; height: 1500px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu10='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520hand_count%2520%252B%253D%25201%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520myHands%253Dresults.multi_hand_landmarks%255B0%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520id%252Clm%2520in%2520enumerate%28myHands.landmark%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520h%252Cw%252Cc%253Dimage.shape%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520cx%252Ccy%253Dint%28lm.x*w%29%252C%2520int%28lm.y*h%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList.append%28%255Bid%252Ccx%252Ccy%255D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520lmList2forModel.append%28%255Bcx%252Ccy%255D%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520lmList%255BtipIds%255B0%255D%255D%255B1%255D%2520%253E%2520lmList%255BtipIds%255B0%255D-1%255D%255B1%255D%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%281%29%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%280%29%250A%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520for%2520id%2520in%2520range%281%252C5%29%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520lmList%255BtipIds%255Bid%255D%255D%255B2%255D%2520%253C%2520lmList%255BtipIds%255Bid%255D-1%255D%255B2%255D%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%281%29%250A%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520else%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers.append%280%29%250A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520total%253D%2520fingers.count%281%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D5%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520sh%253D%2520%2522Acclerate%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520draw%28sh%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D2%2520or%2520total%253D%253D3%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520sh%253D%2520%2522Left%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520draw%28sh%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D4%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520sh%253D%2520%2522Right%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520draw%28sh%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520total%253D%253D0%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520sh%253D%2520%2522Brake%2522%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520draw%28sh%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.draw_landmarks%28image%252Chand_landmark%252Cmp_hand.HAND_CONNECTIONS%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C0%252C255%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%252C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520mp_draw.DrawingSpec%28color%253D%280%252C255%252C0%29%252C%2520thickness%253D2%252C%2520circle_radius%253D2%29%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2523FPS%2520Counter%2520Logic%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520currTime%2520%253D%2520time.time%28%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fps%2520%253D%25201%252F%2520%28currTime%2520-%2520prevTime%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520prevTime%2520%253D%2520currTime%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520fingers%253D%255B%255D%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520"
  style="width: 939px; height: 1129px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''

        uuu11='''<iframe
  src="https://carbon.now.sh/embed?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=seti&wt=none&l=python&ds=false&dsyoff=20px&dsblur=68px&wc=false&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=17px&lh=133%25&si=false&es=2x&wm=false&code=%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520if%2520record%253A%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520out.write%28image%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520image%253D%2520cv2.cvtColor%28image%252Ccv2.COLOR_BGR2RGB%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520kpi1_text.write%28f%2522%253Ch1%2520style%253D%27text-align%253A%2520center%253B%2520color%253Ared%253B%2520%27%253E%257Bint%28fps%29%257D%253C%252Fh1%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520kpi2_text.write%28f%2522%253Ch1%2520style%253D%27text-align%253A%2520center%253B%2520color%253Ared%253B%2520%27%253E%257Bhand_count%257D%253C%252Fh1%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520kpi3_text.write%28f%2522%253Ch1%2520style%253D%27text-align%253A%2520center%253B%2520color%253Ared%253B%2520%27%253E%257Bwidth%257D%253C%252Fh1%253E%2522%252C%2520unsafe_allow_html%253DTrue%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520image%2520%253D%2520cv2.resize%28image%252C%2520%280%252C0%29%252C%2520fx%2520%253D%25200.8%252C%2520fy%2520%253D0.8%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520image%2520%253D%2520image_resize%28image%2520%253D%2520image%252C%2520width%2520%253D%2520320%252Cheight%253D360%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520stframe.image%28image%252C%2520channels%2520%253D%2520%27BGR%27%252C%2520use_column_width%253DFalse%29%250A%2520%2520%2520%2520st.subheader%28%27Output%2520Image%27%29%250A%2520%2520%2520%2520st.text%28%27Video%2520Processed%27%29%250A%2520%2520%2520%2520output_video%2520%253D%2520open%28%27output1.mp4%27%252C%27rb%27%29%250A%2520%2520%2520%2520out_bytes%253D%2520output_video.read%28%29%250A%2520%2520%2520%2520st.video%28out_bytes%29%250A%2520%2520%2520%2520%2520%2520%2520%2520%250A%2520%2520%2520%2520vid.release%28%29%250A%2520%2520%2520%2520out.release%28%29%250A"
  style="width: 1024px; height: 586px; border:0; transform: scale(1); overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>'''
        st.components.v1.html(uuu,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu2,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu3,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu4,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu5,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu6,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu7,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu8,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu9,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu10,width=1024,height=1000,scrolling=True)
        st.components.v1.html(uuu11,width=1024,height=568,scrolling=True)
        
        
        
        
        # pl= "def hel()"    
        # code = st_ace(language='python',theme='dracula',placeholder=pl)
        # '''def hello():
        # print("Hello, Streamlit!")'''
        # st.st_ace(code, language='python',theme='cobalt')
    

            
        
        
        
            
        