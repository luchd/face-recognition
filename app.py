import streamlit as st
import pandas as pd
import numpy as np
import cv2
import glob
import datetime
from deepface import DeepFace
import time

def get_cv2_img(verify_picture) :
    bytes_data = verify_picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return cv2_img

def get_face_objs(cv2_img) :
    face_objs = DeepFace.extract_faces(img_path=cv2_img)
    return face_objs

def get_facial_areas(face_objs) :
    facial_areas = []
    for face in face_objs :
        facial_area = face['facial_area']
        facial_areas.append(facial_area)
    return facial_areas

def get_coordinates(facial_area) :
    start_row = facial_area['x']
    start_col = facial_area['y']
    end_row = facial_area['x'] + facial_area['w']
    end_col = facial_area['y'] + facial_area['h']
    return (start_row, start_col, end_row, end_col)

def draw_rect(cv2_img, start_row, start_col, end_row, end_col) :
    face_img = cv2.rectangle(cv2_img, (start_row, start_col), (end_row, end_col), color=(123,123,123), thickness=6)
    return face_img

def slice_img(face_img, start_row, start_col, end_row, end_col) :
    sliced_face_img = face_img[start_col + 6 : end_col - 6, start_row + 6 : end_row - 6]
    return sliced_face_img

def get_embedding_objs(sliced_face_img) :
    embedding_objs = DeepFace.represent(img_path=sliced_face_img, enforce_detection=False)
    return embedding_objs

def get_facial_embeddings(cv2_img, facial_areas) :
    facial_embeddings = []
    for facial_area in facial_areas :
        (start_row, start_col, end_row, end_col) = get_coordinates(facial_area)
        face_img = draw_rect(cv2_img, start_row, start_col, end_row, end_col)
        sliced_face_img = slice_img(face_img, start_row, start_col, end_row, end_col)
        embedding_objs = get_embedding_objs(sliced_face_img)
        embedding = embedding_objs[0]['embedding']
        facial_embeddings.append(np.array(embedding))
    return facial_embeddings

def calc_cosine_similarity(embedding, facial_embedding) :
    cosine_similarity = (np.array(embedding) @ facial_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(facial_embedding))
    return cosine_similarity

def get_names_similarity(facial_embeddings, all_pics) :
    result = []
    for facial_embedding in facial_embeddings :
        max_similarity = 0
        person_name = ''
        for img_file in all_pics :
            with open(img_file, 'rb') as f :
                embedding = np.load(f)
                cosine_similarity = calc_cosine_similarity(embedding, facial_embedding)
                if cosine_similarity > max_similarity :
                    max_similarity = cosine_similarity
                    person_name = img_file[img_file.rindex('\\') + 1 : -4]
        result.append([person_name, max_similarity])
    return result

def register_img(img_file_buffer, name_input) :
    cv2_img = get_cv2_img(img_file_buffer)
    face_objs = get_face_objs(cv2_img)
    facial_area = face_objs[0]['facial_area']
    (start_row, start_col, end_row, end_col) = get_coordinates(facial_area)
    face_img = draw_rect(cv2_img, start_row, start_col, end_row, end_col)
    st.image(face_img)
    sliced_face_img = slice_img(face_img, start_row, start_col, end_row, end_col)
    embedding_objs = get_embedding_objs(sliced_face_img)
    embedding = embedding_objs[0]['embedding']
    save_embedding(name_input, embedding)

def save_embedding(name_input, embedding) :
    np.save(f'./embeddings/{name_input}', np.array(embedding))


def main() :
    st.title('> face-recognition')
    register, verify, log = st.tabs(['Register', 'Verify', 'Log'])
    logs = []

    with register :
        st.header('Take a nice pic and input your name to register! 😊')
        name_input = st.text_input('Enter your name here: ')
        img_file_buffer = st.camera_input('Take a pic!', key='1')
        if img_file_buffer is not None :
            if (name_input is None) | (name_input == '') :
                st.error('Please enter your name_input')
            else :
                st.write("...And here's your nice pic!")
                register_img(img_file_buffer, name_input)

    with verify :
        st.write('Upload a photo or take one to verify!')

        verify_picture = st.camera_input('Take a pic!', key='2')
        
        if verify_picture is not None :
            cv2_img = get_cv2_img(verify_picture)
            face_objs = get_face_objs(cv2_img)
            facial_areas = get_facial_areas(face_objs)
            facial_embeddings = get_facial_embeddings(cv2_img, facial_areas)

            st.write("* * * Below are people detected from the input photo! * * *")
            time_start = time.perf_counter()
            all_pics = glob.glob('./embeddings/*')
            result = get_names_similarity(facial_embeddings, all_pics)

            for guess in result :
                st.success(f'{guess[0]}: {round(guess[1] * 100, 1)}%')
                logs.append([guess[0], datetime.datetime.now()])

            time_end = time.perf_counter()
            time_duration = time_end - time_start
            st.write(f'===>>> Took ~{time_duration: .3f} seconds')

    with log :
        st.header('This is the log tab!')
        if len(logs) :
            df = pd.DataFrame(np.array(logs), columns=['name', 'time'])
            st.table(df)

main()