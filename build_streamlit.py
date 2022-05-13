# imputing libraries

from tkinter.tix import COLUMN
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Ellipse
import pandas as pd
import seaborn as sns
import numpy as np

#input data into dataframe

    





# objectives of work
def objectives():
    st.header('*The objectives of the project is to:* \n' )  
    st.markdown('### 1.To develop a fitness software that is able to be used in plug and play style into most apps and smart watches \n' 
                '### 2.To build a model that  facilitate the Transport mode detection and calorie counting and make it more precise. \n'
                '### 3.This Model incoporates  different measurements on the phone to classify the activity the athletes are taking part on accurately.'
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


def Benchmark_model():
    
    fig = px.bar(
                benchmark, x = "Model", y = "Accuracy",
                                hue="Model",
                                template = 'seaborn',
                                title = 'accuracy graph of the Raw data', 
                                
                                        )

    st.plotly_chart(fig) 




def Enhanced_data_model():
    

    fig = px.bar(
                feature, x = "Model", y = "Accuracy",
                                hue="Model",
                                template = 'seaborn',
                                title = 'accuracy graph of enhanced data ', 
                                
                                        )

    st.plotly_chart(fig) 
    st.plotly_chart(fig) 


#### best food

def completion_time():
    

    fig = px.bar(
                feature, x = "Model", y = "Time",hue="Model"
                                template = 'seaborn',
                                title = 'time of completion of each model per seconds', 
                                
                                        )

    st.plotly_chart(fig) 
    pass

def hyper_param_model():
    

    fig = px.bar(
                hyper_param, x = "Model", y =  "Accuracy",hue="Model"
                                template = 'seaborn',
                                title = 'Accuracy with Hyperameters', 
                                
                                        )

    st.plotly_chart(fig) 

def completion_time_hyper_param():
    

    fig = px.bar(
                hyper_param, x = "Model", y = "Time",hue="Model"
                                template = 'seaborn',
                                title = 'time of completion of each model per seconds', 
                                
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
       st.markdown("Team - Apple Watch"
                    '1.Omolara'
                    '2.Kingleys'
                    '3.Islom')
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

        methodology()
        

    


    # fifth page
    elif page == "Analysis of Results":
           
           page2 = st.sidebar.selectbox(
             "Sub Content", 
            [
                ""
            ],
            
        )

           if page2 == "Top Location in worst case":
               st.title("Top Location in worst case")
              

           elif page2 == "Top Cuisines in worst case":
               st.title("Top Cuisines in worst case")


           elif page2 == "Top Rated 10 Location":
                  st.title("Top Rated 10 Location")
                  

           elif page2 == "Top Rated 10 Cuisines":
                  st.title("Top Rated 10 Cuisines")
                  


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