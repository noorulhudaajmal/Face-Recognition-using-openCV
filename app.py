import numpy as np
import cv2 as cv
import face_recognition
import streamlit as st
import pickle
from streamlit_option_menu import option_menu as menu

st.set_page_config( page_title="Face Detection", layout="wide")


title_temp = """
<div style="background-color:{};padding:10px;border-radius:10px">
<h1 style="color:{};text-align:center;">{}</h1>
</div>
"""
sub_title_temp = """
<div style="background-color:{};padding:0.5px;border-radius:5px;">
<h4 style="color:{};text-align:center;">{}</h6>
</div>
"""
head_title_temp = """<h6 style="text-align:left;margin-top:2px">{}</h6>"""

# registered_people = {}
reg_face_names = []
reg_face_image_bytes = []
reg_face_encodings = []


def load_data():
    registered_people = {}
    with open("registered_people.dat", "rb")  as f:
        if f is not None:
            try:
                registered_people = pickle.load(f)
            except:
                registered_people = {}
            print("Data loaded successfully!")
        else:
            print("No person's data currently stored!")
    return registered_people

def update_data():
    data = {}
    global reg_face_encodings,reg_face_image_bytes,reg_face_names
    with open("registered_people.dat", "rb")  as f:
        if f is not None:
            try:
                data = pickle.load(f)
            except:
                data = {}
    names = list(data.keys())
    bytes = []
    encodings = []
    for i in data:
        bytes.append(data[i][0])
        encodings.append(data[i][1])
    reg_face_names = names ; reg_face_image_bytes = bytes ; reg_face_encodings = encodings


def register_person(name,image_bytes):
    person = load_data()
    image = cv.imdecode(image_bytes, 1)
    face_location = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image,known_face_locations=face_location)
    if len(face_encodings)==1:
        print("Face detected!")
        face = face_encodings[0]
        person[name] = [image_bytes,face]
        with open("registered_people.dat", "wb")  as f:
            pickle.dump(person,f)
        update_data()

        return 0
    else:
        return -1


def get_results(image):
    # image = face_recognition.load_image_file(img)
    # image = cv.imread(image)
    # image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    scale_x = 1
    scale_y = 1
    # if (image.shape[0] > 650) | (image.shape[1] > 550):
    #     scale_x = 0.3
    #     scale_y = 0.3
    copy = cv.resize(image,(0,0),None,scale_x,scale_y)
    faces_locations = face_recognition.face_locations(image)
    faces_encodings = face_recognition.face_encodings(image , known_face_locations=faces_locations)
    for locations in faces_locations:
        y,w,h,x = locations
        y,w,h,x = int(y*scale_y),int(w*scale_x),int(h*scale_y),int(x*scale_x)
        cv.rectangle(copy,(x,y),(w,h),(0,255,0),2)
        # cv.rectangle(copy,(locs[3]*4,locs[0]*4),(locs[1]*4,locs[2]*4),(255,0,0),2)
    for i in range(len(faces_encodings)):
        face = faces_encodings[i]
        found = face_recognition.compare_faces(reg_face_encodings,face,0.52)
        face_distances = face_recognition.face_distance(reg_face_encodings,face)

        if True in found:
            least_distance = face_distances.argmin()
            # index = found.index(True)
            index = least_distance
            print("Person = " , reg_face_names[index])
            box_loc = faces_locations[i]
            text = "{0} {1}".format(reg_face_names[index] , round(face_distances[index],2))
            y,w,h,x = box_loc
            y,w,h,x = int(y*scale_y),int(w*scale_x),int(h*scale_y),int(x*scale_x)
            cv.rectangle(copy,(x,y-20),(w,y),(0,255,0),cv.FILLED)
            cv.putText(copy ,text , (x,y-5),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        # else:
        #     print("No known faces detected!")

    return copy


def remove_person(ind):
    person = {}
    reg_face_names.pop(ind)
    reg_face_encodings.pop(ind)
    reg_face_image_bytes.pop(ind)
    for i in range(len(reg_face_names)):
        person[reg_face_names[i]] = [reg_face_image_bytes[i],reg_face_encodings[i]]
    with open("registered_people.dat", "wb")  as f:
        pickle.dump(person,f)
    update_data()


update_data()



st.markdown(title_temp.format('#66999B','white' , "ATTENDANCE USING FACIAL RECOGNITION"),unsafe_allow_html=True)
st.write("")
st.write("")

with st.sidebar:
    st.write("### REGISTERED PEOPLE")
    search_in = st.text_input("Search for person")
    if search_in is not None:
        search = search_in in reg_face_names
        if search:
            st.write("Record Found!")
            ind = reg_face_names.index(search_in)
            l = st.image(cv.resize(cv.imdecode(reg_face_image_bytes[ind],1),(200,200)) , channels="BGR" , caption=reg_face_names[ind])
            if st.button("Remove from list"):
                remove_person(ind)
        else:
            st.write("No record found")
    st.write("### List of registered people")
    for b,n in zip(reg_face_image_bytes,reg_face_names):
        x = cv.resize(cv.imdecode(b,1),(200,200))
        st.image(x,channels="BGR" , caption=n)


selected = menu(
    menu_title = None,
    options = ["Detect Presence" , "LIVE People Dectect" , "Register Newbies"],
    icons = ["image" , "camera-video","person-plus"],
    default_index = 0,
    orientation = "horizontal"
)

if selected == "Detect Presence":
    st.write("") ;st.write("") ;st.write("") ;st.write("") ;
    st.markdown(sub_title_temp.format('white','black' , "DETECT PRESENCE"),unsafe_allow_html=True)
    st.write("") ;st.write("") ;st.write("") ;
    col1 , col2 = st.columns((2,1))
    with col1:
        image = st.file_uploader("Upload image to detect registered people" , type=["png","jpg","jpeg"])
    with col2:
        pass

    if image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        result = get_results(opencv_image)
        # result = cv.resize(result,(600,400))
        st.image(result , channels="BGR")

if selected == "LIVE People Dectect":

    st.write("") ;st.write("") ;st.write("") ;st.write("") ;
    st.markdown(sub_title_temp.format('white'  ,"black", "START DETECTION USING LIVE CAMERA"),unsafe_allow_html=True)

    # st.write("### START DETECTION USING LIVE CAMERA")
    st.write("") ;st.write("") ;st.write("") ;
    start = st.checkbox("TURN ON LIVE STREAMING")
    FRAME_WIN = st.image([])
    cam = cv.VideoCapture(1)

    while start:
        success , image = cam.read()
        if image is None:
            st.write("Unable to render webcam, try changing camera!")
            break
        FRAME_WIN.image(get_results(image),channels="BGR")
    else:
        st.write("Live streaming is stopped!")
        del cam

if selected == "Register Newbies":

    st.write("") ;st.write("") ;st.write("") ;st.write("") ;
    st.markdown(sub_title_temp.format('white' ,"black", "REGISTER PEOPLE"),unsafe_allow_html=True)
    st.write("") ;st.write("") ;st.write("") ;
    start_registration = False
    col1,col2,col3 = st.columns((1,2,1))
    with col1:
        name_input = st.text_input("Enter the name of person without spaces [Convention : ID-NAME as 101-John]")
        button = st.button("Register")
    with col2:
        new_image = st.file_uploader("Upload image of person with face centered" , type=["png","jpg","jpeg"])
        # file_bytes = np.asarray(bytearray(new_image.read()), dtype=np.uint8)
    with col3:
        # pass
        if (name_input is not None) & (new_image is not None):
            file_bytes = np.asarray(bytearray(new_image.read()), dtype=np.uint8)
            img = cv.resize(cv.imdecode(file_bytes,1),(200,200))
            st.image(img,channels="BGR",caption=[name_input])

    if (name_input is not None) & (new_image is not None):
        start_registration = True

    if (start_registration) & (button):
        st.write("registration initiates...")
        # file_bytes = np.asarray(bytearray(new_image.read()), dtype=np.uint8)
        status = register_person(name_input,file_bytes)
        if status != 0 :
            st.warning("Error uploading the image, upload a proper image!")
        else:
            st.info("Image uploaded!")






