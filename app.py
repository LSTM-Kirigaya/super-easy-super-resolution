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
## progress bar
if file_uploader and os.path.exists(save_path):
    with st.sidebar:
        progress_bar = st.progress(value=0., text='rebuilding')

## section for input image
with left_section:
    if file_uploader and os.path.exists(save_path):
        st.image(image=save_path, caption='input image', use_column_width='always')
    else:
        st.info('Wait for image to upload. Click Browse files!')

## section for output image
with right_section:
    if file_uploader and os.path.exists(save_path):
        count = 0
        total_stages = 100
        for output in esrgan.streamlit_main(save_path, device, 'model.bin'):
            if isinstance(output, int):
                total_stages = output + 1
            if isinstance(output, torch.Tensor):
                count += 1
                percent = round(count / total_stages * 100, 3)
                progress_bar.progress(value=min(count / total_stages, 1.0), text=f'rebuilding ({percent * 100}%)')
                print(count, total_stages, min(count / total_stages, 1.0))
                
            if hasattr(output, 'save'):
                progress_bar.progress(value=1., text='bingo 😎')
                result_save_path_file = f'{save_path}.rebuild.png'
                output.save(result_save_path_file)
                st.image(result_save_path_file, caption='rebuild image', use_column_width='always')
                with open(result_save_path_file, 'rb') as fp:
                    st.download_button(label='Download', data=fp, file_name='rebuild.png', mime='image/png')
    
    else:
        st.info('Generate image is empty.')