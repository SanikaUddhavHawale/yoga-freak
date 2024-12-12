import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlit app title
st.title("Yoga Pose Detection & Correction")
st.write("This application uses AI to detect and provide feedback on yoga poses in real-time.")

# Sidebar for user input
st.sidebar.title("Settings")
webcam_option = st.sidebar.radio("Select Video Source:", ("Webcam", "Upload Video"))

# Function to process video frames
def process_frame(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = pose.process(frame_rgb)
    
    # Draw landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return frame, results

# Handle webcam input
if webcam_option == "Webcam":
    st.write("Using webcam for real-time pose detection.")
    run_webcam = st.checkbox("Start Webcam")
    
    if run_webcam:
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.warning("Unable to access webcam.")
                break

            # Process the frame
            frame, results = process_frame(frame)
            
            # Display the frame
            stframe.image(frame, channels="BGR", use_container_width=True)

        video_capture.release()
        cv2.destroyAllWindows()

# Handle video upload
elif webcam_option == "Upload Video":
    st.write("Upload a video for yoga pose detection.")
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Load the video
        video_capture = cv2.VideoCapture(uploaded_file.name)
        stframe = st.empty()
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process the frame
            frame, results = process_frame(frame)

            # Display the frame
            stframe.image(frame, channels="BGR", use_container_width=True)

        video_capture.release()
        cv2.destroyAllWindows()

# Show instructions for yoga poses
st.sidebar.subheader("Pose Instructions")
st.sidebar.write(
    "Ensure you are visible to the camera and in a well-lit area for better pose detection."
)

# Footer
st.write("\n\n**Note:** This application is for educational purposes and is not a substitute for professional yoga guidance.")