import os
from pathlib import Path
import urllib
import tempfile

from tqdm import tqdm
import pandas as pd
import streamlit as st

DEFAULT_DATA_BASE_DIR = 'demo'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_DEMO_REPORT = "Select a Demo Report"
SIDEBAR_OPTION_UPLOAD_REPORT = "Upload a Report"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_REPORT, SIDEBAR_OPTION_UPLOAD_REPORT]

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/pierre-si/financial-summarisation/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def convert_to_csv(path):
    data=[]
    data_ids=[]
    data_long=[]

    with open(path) as fin:
        content = fin.read().replace('\x00', '').strip()
        data.append([content[:22000], 'X'])
        data_long.append(content)
        data_ids.append(id)
        
    data = pd.DataFrame(data)
    data.columns=['text','summary']
    data.to_csv('tet.csv',index=False)
    return data, data_ids, data_long

#        --model_name_or_path ./model \
def run_app(data, data_ids, data_long):
    os.system('''python transformers/examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path orzhan/t5-long-extract \
        --source_prefix "sum: " \
        --do_predict \
        --validation_file tet.csv \
        --test_file tet.csv \
        --output_dir output \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --predict_with_generate \
        --max_source_length 4096 \
        --max_target_length 64''')

    # Create summaries based on predicted sequences
    with open('output/generated_predictions.txt','r') as fin:
        preds=fin.readlines()
    summaries=[]
    aa=[]

    for i, row in tqdm(data.iterrows(), total=len(data)):
        pos=None
        # Split text into words
        tt=data_long[i]
        tt_words=['']
        tt_spaces=[]
        prev='a'
        for k in range(0,len(tt)):
            if tt[k].isspace() and prev.isspace():
                tt_spaces[-1] += tt[k]
            elif tt[k].isspace() and not prev.isspace():
                tt_spaces.append(tt[k])
            elif not tt[k].isspace() and prev.isspace():
                tt_words.append(tt[k])
            else:
                tt_words[-1] += tt[k]
            prev = tt[k]
        tt_spaces.append('')
        # Locate prediction in the text (best match)
        pp=preds[i].split()
        posmax=len(pp)
        for p in range(min(len(tt_words)-len(pp), 4000)):
            fail=0
            for j in range(len(pp)):
                if tt_words[p+j] != pp[j]:
                    fail+=1         # count different words
            if fail<posmax:     # found new best match
                pos=p
                posmax=fail
                if fail==0:
                    break
        if pos is None:
            pos=0
        aa.append(pos)
        predicted=''
        pred_length=1000
        for j in range(pos, pos+pred_length):             # take 1000 words from pos
            predicted += tt_words[j] + tt_spaces[j]
            if j == len(tt_words)-1:
                break
        summaries.append(predicted)
        with open('output/summary.txt', 'w') as fout:
            fout.write(predicted)

def main():
    st.sidebar.title("Explore the Following")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.sidebar.write("More About The Project")
        st.sidebar.write("Hi there! If you want to check out the source code, please visit my github repo: https://github.com/pierre-si/financial-summarisation/")

        st.write(get_file_content_as_string("README.md"))

    elif app_mode == SIDEBAR_OPTION_DEMO_REPORT:
        st.sidebar.write(" ------ ")

        directory = Path(DEFAULT_DATA_BASE_DIR)

        files = []
        for filepath in directory.iterdir():
            files.append(filepath.name)
        files.sort(key=lambda x: float(x[:-3]))

        option = st.sidebar.selectbox('Please select a sample report, then click Magic Time button', files)

        with open(directory/option) as f:
            content = f.read().replace('\x00','').strip()
        preview = st.text(content)

        pressed = st.sidebar.button('Magic Time')
        if pressed:
            preview.empty()
            st.empty()
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')

            path = Path(directory) / option
            data, data_ids, data_long = convert_to_csv(path)
            run_app(data, data_ids, data_long)
            with open("output/summary.txt", "r") as f:
                summary = f.read()#.replace('\x00','').strip()
            st.text("Generated summary:")
            st.text(summary)
            st.text("Original file:")
            st.text(content)


    elif app_mode == SIDEBAR_OPTION_UPLOAD_REPORT:
        # st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
        #     and discarded after the final results are displayed. ')
        f = st.sidebar.file_uploader("Please Select to Upload a report", type=['txt'])
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=True)
            content = f.read()
            tfile.write(content)
            st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
            data, data_ids, data_long = convert_to_csv(tfile.name)
            run_app(data, data_ids, data_long)
            with open("output/summary.txt", "r") as f:
                summary = f.read()#.replace('\x00','').strip()
            st.text("Generated summary:")
            st.text(summary)
            st.text("Original file:")
            with open(tfile.name) as f:
                content = f.read().replace('\x00','').strip()
            st.text(content)

main()
