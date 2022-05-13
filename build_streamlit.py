# imputing libraries

from tkinter.tix import COLUMN
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Ellipse
import pandas as pd
import seaborn as sns
import numpy as np
#from train_n_test import benchmark, feature,hyper_param

#input data into dataframe
df = pd.read_csv("./dataset_5secondWindow/dataset_5secondWindow.csv")
df4 = df.sort_values(by='user', ascending=True)
benchmark= pd.read_csv('benmarch_model.csv')
feature= pd.read_csv('feature_engineer_model.csv')
hyper_param = pd.read_csv('hyper_param_model.csv')




# objectives of work
def objectives():
    st.header('*The objectives of the project is to:* \n' )  
    st.markdown('### 1.To develop a fitness software that is able to be used in plug and play style into most apps and smart watches \n' 
                '### 2.To build a model that  facilitate the Transport mode detection and calorie counting and make it more precise. \n'
                '### 3.This Model incoporates  different measurements on the phone to classify the activity the athletes are taking part on accurately.\n'
                '### 4. the activities are grouped into three classes, which are : class 0 (Train,car, bus), class 1(still), class 2 (walking)'

                )
        
# outline   
def outline():
    st.markdown('### 1. Project Objectives \n ' 

                '### 2. Methodology of the Work \n'

                '### 3. Analysis \n'

                '### 4. Main Analysis \n'

                '### 5. Final Result \n'

                '### 6. Recommendation')
        
    
# methods and show data
def methodology():
    # st.markdown('#### Data used was London Hotels, Restaurants and Pubs, scraped from [yelp](https://www.yelp.co.uk/search?find_desc=&find_loc=London%2C+United+Kingdom&ns=1) website \n')
    # #st.header("Converted Price into Number")
    # st.markdown("### Converting Pounds:  £ = 1, ££ = 2, £££ = 3, ££££ = 4")
    
    

    # st.markdown('### ***Restaurants Data***')
    # st.dataframe(df_restaurant)

    # st.markdown('### ***Pubs Data***')
    # st.dataframe(df_pub)

    # st.markdown('### ***Hotels Data***')
    # st.dataframe(df_hotel)
   pass
def preprocessing():
    fig, ax = plt.subplots(figsize=(16, 8))

    ax = sns.countplot(x='user', hue='target', data=df4) 
    st.plotly_chart(ax)


def Benchmark_model():
    
    fig = px.bar(
                benchmark, x = "Model", y = "Accuracy",
                                
                                template = 'seaborn',
                                title = 'accuracy graph of the Raw data', 
                                
                                        )

    st.plotly_chart(fig) 




def Enhanced_data_model():
    

    fig = px.bar(
                feature, x = "Model", y = "Accuracy",
                               
                                template = 'seaborn',
                                title = 'accuracy graph of enhanced data ', 
                                
                                        )

    st.plotly_chart(fig) 
    st.plotly_chart(fig) 


#### best food

def completion_time():
    

    fig = px.bar(
                feature, x = "Model", y = "Time",hue="Model",
                                template = 'seaborn',
                                title = 'time of completion of each model per seconds', 
                                
                                        )

    st.plotly_chart(fig) 
  

def hyper_param_model():
    

    fig = px.bar(
                hyper_param, x = "Model", y =  "Accuracy",hue="Model",
                                template = 'seaborn',
                                title = 'Accuracy with Hyperameters', 
                                
                                        )

    st.plotly_chart(fig) 

def completion_time_hyper_param():
    

    fig = px.bar(hyper_param, x = "Model", y = "Time",hue="Model",
                                template = 'seaborn',
                                title = 'time of completion of each model per seconds'
                                
                                        )

    st.plotly_chart(fig) 







#  Output in the Streamlit App.
def main():
    #st.title('Kojo')
    page = st.sidebar.selectbox(
        "Main Content", 
        [
            "Title",
            "Presentation Outline",
            "Objectives",
            "Methodology",
            "Analysis of Results",
            "Demonstration"
        ],
        
    )
    
    if page=='Title':
       st.title("Apple Watch FIt Tracker")
       st.markdown("Team - Apple Watch \n"
                    '1.Omolara\n'
                    '2.Kingleys\n'
                    '3.Islom\n')
       #st.image("Downloads\\london.jpg", use_column_width = True)
    

    #First Page
    elif page == "Presentation Outline":
        st.title("Presentation Outline")
        #st.image("Downloads\\pre1.jpg", use_column_width = True)
        outline()


    #Second Page
    elif page == "Objectives":
       
        objectives()
       # st.image("Downloads\\food.jpg", use_column_width = True)
        
    
    #Third Page
    elif page == "Methodology":

        preprocessing()
    
        

    


    # fifth page
    elif page == "Analysis of Results":
        Benchmark_model()
        Enhanced_data_model()
        #     page2 = st.sidebar.selectbox(
        #      "Sub Content", 
        #     [
        #         ""
        #     ],
            
        # )

        #    if page2 == "Top Location in worst case":
        #        st.title("Top Location in worst case")
              

        #    elif page2 == "Top Cuisines in worst case":
        #        st.title("Top Cuisines in worst case")


        #    elif page2 == "Top Rated 10 Location":
        #           st.title("Top Rated 10 Location")
                  

        #    elif page2 == "Top Rated 10 Cuisines":
        #           st.title("Top Rated 10 Cuisines")
                  


    #Fifth page
    elif page == "Final Result":
        st.title("Final Result")
        st.header("Restaurants")
        #st.image("Downloads\\res.jpg", use_column_width = True)
        st.balloons()


    elif page == "Recommendation":
        st.title("Recommended Location")
        
         
    

if __name__ == "__main__":
    main()