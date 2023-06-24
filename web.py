import io
import re
from io import BytesIO
import requests
import pandas as pd
import docx2txt
import time
from time import sleep
from pandas_profiling import ProfileReport

import streamlit as st
from streamlit_tags import st_tags
import hydralit_components as hc
from streamlit_pandas_profiling import st_profile_report

import plotly.express as px
import plotly.graph_objects as go

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import spacy
from spacy.matcher import Matcher

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize

import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
st.set_option('deprecation.showPyplotGlobalUse', False)

import sys
sys.coinit_flags = 0

import pickle
from pickle import load

# Load the KNN model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfidf.pkl', 'rb') as f:
    model1 = pickle.load(f)

word = pd.read_csv('Resume word counts.csv').drop(columns=['Resume'])
clean = pd.read_csv('Resume cleaned.csv')

st.set_page_config(layout='wide',initial_sidebar_state='collapsed')
menu_data = [{'icon': "far fa-sticky-note", 'label':"About"},
    {'icon': "far fa-chart-bar", 'label':"Resume Data Analysis"},
    {'icon': "far fa-file-word", 'label':"Resume Classification"},] #no tooltip message]


over_theme = {'txc_inactive': 'white','menu_background':'purple','txc_active':'black','option_active':'white'}
font_fmt = {'font-class':'h2','font-size':'150%'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    login_name=None,
    hide_streamlit_markers=False, 
    sticky_nav=True,
    sticky_mode='pinned', )
 
if menu_id == 'Home':
    st.markdown("""<style>.stProgress .st-bo {color: purple;}</style>""", unsafe_allow_html=True)

    progress = st.progress(0)
    for i in range(100):
        progress.progress(i+1)
        sleep(0.001)
    
    st.markdown("<h1 style='text-align: center; color: black;'>RESUME SCREENING AND CLASSIFICATION</h1>", unsafe_allow_html=True)
    st.image("https://ursusinc.com/wp-content/uploads/2020/09/Resume-Blog-Animation-1080.gif")

if menu_id == "About":
    st.markdown("""<style>.stProgress .st-bo {color: purple;}</style>""", unsafe_allow_html=True)

    progress = st.progress(0)
    for i in range(100):
        progress.progress(i+1)
        sleep(0.001)

    st.markdown("<h1 style='text-align: center; color: black;'>BUSINESS OBJECTIVE </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: justify; font-size:180%; font-style: italic; color: black;'> The document classification solution should significantly reduce the manual human effort in the HRM. It should achieve a higher level of accuracy and automation with minimal human intervention.</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'> ABSTRACT </h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size:140%;'> With the growth of online recruitment, hiring platforms receives and stores an enormous amount of resumes data and job posts of various fields. Hence, applying through resume classification systems can reduce the time and labor required for classification, allowing recruiters to work efficiently & quickly. The recruiting process has several steps, but the first is resume categorization and verification. Automating the first stage would greatly assist the interview process in terms of speedy applicant selection. Classification of resumes is achieved by using Natural Language Processing (NLP) & Machine Learning (ML) Algorithm. NLP & ML models provide a high degree of automation in text analysis, being much more accurate and flexible than rule-based systems..</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'> INTRODUCTION </h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size:140%; '> The system make use of Machine Learning Algorithm & Natural Language Processing using Python as base langauge. A pure python-based utility to extract text from documents and a function to search for files that match a specific file pattern or nameText has been used to extract raw data and create features. Further, by using NLP techniques like stopwords identification, text tokenization, text lemmatization & stemming, tagging parts of speech(POS) by Name Entity Identification and text vectorization like TF-IDF vectorization to assign tfidf value indicating the importance of the word as per frequency is performed in order to create features for data training in the project. Once the preprocessing of words is done, then using various machine learning algorithms, the data is classified into different classes of job profile in accordance to the skillset.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size:140%; '> The phases to achieve the desired results were data collection, data cleaning, data analysis, data visualization, and building best ML model to get maximum accuracy in training and testing stage of process. The dataset created after preprocessing consists features as Name, Skills, Education, Experience in years, Contact Details as Contact number & Email ID, Job Category and a 'Resume' column that contain details of the candidate. The objective of this project is to make the e-recruitment process efficient and user-friendly. This approach will assist businesses and save time throughout the recruitment process. </p>", unsafe_allow_html=True)


if menu_id == 'Resume Data Analysis':
    st.markdown("""<style>.stProgress .st-bo {color: purple;}</style>""", unsafe_allow_html=True)
    progress = st.progress(0)
    for i in range(100):
        progress.progress(i+1)
        sleep(0.001)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h1 style='text-align: center; color: black;'> 🔍 Word Analysis </h1>", unsafe_allow_html=True)
        with st.expander('**Explore Data**'):
            x_axis = st.selectbox('**X-Axis**',options=[None]+list(word.columns),index=0)
            y_axis = st.selectbox('**Y-Axis**',options=[None]+list(word.columns),index=0)
            
            if x_axis and y_axis:
                if (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes != 'object'):
                    plots = ['Scatter Plot']
                elif (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes == 'object'):
                    plots = ['Bar Plot']
                elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes != 'object'):
                    plots = ['Bar Plot']
                elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes == 'object'):
                    plots = ['Bar Plot']

            elif x_axis and not y_axis:
                if word[x_axis].dtypes != 'object':
                    plots = ['Histogram','Box Plot']
                else :
                    plots = ['Bar Plot','Pie Plot']
            elif not x_axis and y_axis:
                if word[y_axis].dtypes != 'object':
                    plots = ['Histogram','Box Plot']
                else :
                    plots = ['Bar Plot','Pie Plot']
            else :
                plots = []
            if plots :
                disPlot = False
            else :
                disPlot = True

            disp = st.selectbox('**Plots**',options=plots,disabled=disPlot)
            if disp in ['Bar Plot','Pie Plot']:
                lim_dis = False
            else :
                lim_dis = True
            
            plot = st.button('**Plot**')
        
        if disPlot:
            st.warning('No Plots Available.')
        else :
            if plot :
                # plot here 
                if x_axis and not y_axis:
                    if disp == 'Histogram':
                        fig = px.histogram(word,x=[x_axis],title=f'<b>{x_axis}</b>')
                        st.plotly_chart(fig)
                    elif disp == 'Box Plot':
                        fig = px.box(word,x=[x_axis],title=f'<b>{x_axis}</b>')
                        st.plotly_chart(fig)
                    elif disp == 'Bar Plot':
                        emp = word[x_axis].value_counts().head()
                        fig = px.bar(x=emp.index,y=emp.values)
                        st.plotly_chart(fig)
                    elif disp == 'Pie Plot':
                        emp = word[x_axis].value_counts().head()
                        fig = px.pie(values=emp.values,names=emp.index,title=f'<b>{x_axis}</b>')
                        st.plotly_chart(fig)

                elif y_axis and not x_axis:
                    if disp == 'Histogram':
                        fig = px.histogram(word,x=[y_axis],title=f'<b>{y_axis}</b>')
                        st.plotly_chart(fig)
                    elif disp == 'Box Plot':
                        fig = px.box(word,x=[y_axis],title=f'<b>{y_axis}</b>')
                        st.plotly_chart(fig)
                    elif disp == 'Bar Plot':    
                        emp = word[y_axis].value_counts().head()
                        fig = px.bar(x=emp.index,y=emp.values)
                        st.plotly_chart(fig)
                    elif disp == 'Pie Plot':
                        emp = word[y_axis].value_counts().head()
                        fig = px.pie(values=emp.values,names=emp.index,title=f'<b>{y_axis}</b>')
                        st.plotly_chart(fig)    

                elif x_axis and y_axis:
                    if (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes != 'object'):
                        if disp == 'Scatter Plot':
                            fig = px.scatter(word,x=x_axis,y=y_axis,title=f'{y_axis} Vs {x_axis}')
                            st.plotly_chart(fig)
                    elif (word[x_axis].dtypes != 'object') and (word[y_axis].dtypes == 'object'):
                        if disp == 'Bar Plot':
                            emp = word[[y_axis,x_axis]].groupby(by=[y_axis]).mean()
                            fig = px.bar(x=emp.values.ravel(),y=emp.index,title=f'{y_axis} Vs mean({x_axis})')
                            st.plotly_chart(fig)
                    elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes != 'object'):
                        if disp == 'Bar Plot':
                            emp = word[[y_axis,x_axis]].groupby(by=[x_axis]).mean()
                            fig = px.bar(x=emp.index,y=emp.values.ravel(),title=f'{y_axis} Vs {x_axis}')
                            st.plotly_chart(fig)
                    elif (word[x_axis].dtypes == 'object') and (word[y_axis].dtypes == 'object'):
                        if disp == 'Bar Plot':
                            
                            #st.write(word[[x_axis,y_axis]].pivot_table(index=[x_axis,y_axis]).index)
                            word['dummy'] = np.ones(len(word))
                            emp = word[[x_axis,y_axis,'dummy']].pivot_table(index=[x_axis,y_axis],values=['dummy'],aggfunc=np.sum)
                            emp = emp.reset_index((0,1))

                            fig = px.bar(emp,x=x_axis,y='dummy',color=y_axis)
                            st.plotly_chart(fig)
                            
                else :
                    st.warning('No Plots Available.')
            else :
                st.info('Click Plot Button')

    with col2:
        # Web App Title
        st.markdown("<h1 style='text-align: center; color: black;'> Dataset EDA </h1>", unsafe_allow_html=True)
        
        # Upload CSV data
        uploaded_file = st.file_uploader(" ", type=["csv"])
        
        # Pandas Profiling Report
        if uploaded_file is not None:
            def load_data():
                csv = pd.read_csv(uploaded_file)
                return csv

            df = load_data()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr) 

if menu_id == 'Resume Classification':
    st.markdown("""<style>.stProgress .st-bo {color: purple;}</style>""", unsafe_allow_html=True)
    progress = st.progress(0)
    for i in range(100):
        progress.progress(i+1)
        sleep(0.001)

    def tokenText(extText):
        punc = r'''!()-[]{};:'"\,.<>/?@#$%^&*_~'''
        for ele in extText:
            if ele in punc:
                puncText = extText.replace(ele, "")
        stop_words = set(stopwords.words('english'))
        puncText.split()
        word_tokens = word_tokenize(puncText)
        TokenizedText = [w for w in word_tokens if not w.lower() in stop_words]
        TokenizedText = []
      
        for w in word_tokens:
            if w not in stop_words:
                TokenizedText.append(w)
        return(TokenizedText)            
    def extract_name(Text):
        name = ''  
        for i in range(0,3):
            name = " ".join([name, Text[i]])
        return(name)
    STOPWORDS = set(stopwords.words('english'))

    def extract_skills(resume_text):
        nlp_text = nlp(resume_text)
        noun_chunks = nlp_text.noun_chunks
        # removing stop words and implementing word tokenization
        tokens = [token.text for token in nlp_text if not token.is_stop]
        data = pd.read_csv("skillset.csv") 
        skills = list(data.columns.values)
        skillset = []
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)
        for token in noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)
        return [i.capitalize() for i in set([i.lower() for i in skillset])]
    def string_found(string1, string2):
            if re.search(r"\b" + re.escape(string1) + r"\b", string2):
                return True
            return False
    def extract_text_from_docx(path):
        if path.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            temp = docx2txt.process(path)
            return temp
    def display(docx_path):
        txt = docx2txt.process(docx_path)
        if txt:
            return txt.replace('\t', ' ')
    def preprocess(sentence):
        sentence=str(sentence)
        sentence = sentence.lower()
        sentence=sentence.replace('{html}',"") 
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        rem_url=re.sub(r'http\S+', '',cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)  
        filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
        return " ".join(lemma_words)
    df = pd.DataFrame(columns=['Name','Skills'], dtype=object)
    st.markdown("<h1 style='text-align: center; color: black;'> RESUME ANALYSIS </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: black;'>Upload Resume here </h3>", unsafe_allow_html=True)
    upload_file = st.file_uploader('', type= ['docx'], accept_multiple_files=False) 
    if upload_file is not None:
        displayed=display(upload_file)
        i=0
        text = extract_text_from_docx(upload_file)
        tokText = tokenText(text)
        df.loc[i,'Name']=extract_name(tokText)
        df.loc[i,'Skills']=extract_skills(text)
        col1, col2 = st.columns(2)
        with col1:
            displayed=extract_text_from_docx(upload_file)
            cleaned=preprocess(display(upload_file))
            predicted= model.predict(model1.transform([cleaned]))
            st.subheader(upload_file.name +" is entitled to work in  "+ " " + predicted + " " + "Profile")
            if predicted == 'Work Day':
                st.image("https://www.workday.com/content/dam/web/en-us/images/social/workday-og-theme.png",width=480)
            elif predicted == 'SQL Developer':
                st.image("https://assets.leetcode.com/static_assets/others/SQL_2.gif",width=480)
            elif predicted == 'React JS Developer':
                st.image("https://miro.medium.com/v2/resize:fit:1400/0*EitUXT-pqbaQSCTt.gif",width=480)
            elif predicted == 'Peoplesoft':
                st.image("https://www.orbitanalytics.com/wp-content/uploads/2021/01/oracle-by-peoplesoft-logo.png",width=480)
            expander = st.expander("View Resume")
            expander.write(displayed)    
        with col2:
            st.markdown("<h1 style='text-align: center; font-size:140%; color: black;'> TECHNICAL SKILLS </h1>", unsafe_allow_html=True)
            st.subheader(upload_file.name + ' has expertise in')
            keywords = st_tags(label = ' ', text = '--', value= df['Skills'][0])