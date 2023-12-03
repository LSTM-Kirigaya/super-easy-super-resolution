import importlib
import time
import os

import streamlit as st
import torch
esrgan = importlib.import_module('real-esrgan')

## make dir for image cache
cache_dir = 'streamlit_cache'
model_path = './model.bin'
if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

## make streamlit app
st.set_page_config(
    page_icon='🐳',
    page_title='Super Easy Super Resolution',
    layout='wide'
)


@st.cache_data
def save_image(file_id: int):
    save_path = os.path.join(cache_dir, file_id + '.png')
    open(save_path, 'wb').write(file_uploader.read())

## make side bar
with st.sidebar:
    st.title('esr-gan')
    st.markdown('⭐star my [project](https://github.com/LSTM-Kirigaya/super-easy-super-resolution)')
    file_uploader = st.file_uploader(label='upload your image', type=['.jpg', '.png', '.jpeg'])
    st.markdown('---')
    device = st.radio(label='device', options=['cpu', 'cuda'])
    
    
## handle file uploader
if file_uploader:
    file_id = file_uploader.file_id
    save_path = os.path.join(cache_dir, file_id + '.png')
    open(save_path, 'wb').write(file_uploader.read())

## check option
if device:
    if device == 'cuda' and not torch.cuda.is_available():
        st.error('cuda is not available in the server, please switch to cpu!')
        exit()
    if not os.path.exists(model_path):
        st.error('default weights named "model.bin" is not found in the server, please check!')
        exit()
    
## make main section (two column, one for input image render, one for output one)
left_section, right_section = st.columns(spec=2)

## section for input image
if file_uploader and os.path.exists(save_path):
    left_section.image(image=save_path, caption='input image', use_column_width='always')
else:
    left_section.info('Wait for image to upload. Click Browse files!')

## section for output image
if file_uploader and os.path.exists(save_path):
    with st.sidebar:
        st.markdown('---')
        output = esrgan.streamlit_main(save_path, device, 'model.bin')
    
    result_save_path_file = f'{save_path}.rebuild.png'
    # delete other images
    for image in os.listdir(cache_dir):
        if image != save_path:
            os.remove(os.path.join(cache_dir, image))
    output.save(result_save_path_file)
            
    right_section.image(result_save_path_file, caption='rebuild image', use_column_width='always')    
    with open(result_save_path_file, 'rb') as fp:
        st.sidebar.download_button(label='Download', data=fp, file_name='rebuild.png', mime='image/png')
else:
    right_section.info('Generate image is empty.')


