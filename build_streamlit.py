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
    st.markdown('### 1. \n' 

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


def bar_chart_location():
    
    fig = px.bar(
                pd.DataFrame, x = "Location Name", y = "Total Number",
                                template = 'seaborn',
                                title = '', 
                                
                                        )

    st.plotly_chart(fig) 




# def bar_chart_best_location():
    

#     fig = px.bar(
#                 pd.DataFrame, x = "Location Name", y = "Total Number",
#                                 template = 'seaborn',
#                                 title = 'Top Rated 10 Location', 
                                
#                                         )

#     st.plotly_chart(fig) 


#### best food

def bar_chart_best_food():
    

    # fig = px.bar(
    #             pd.DataFrame, x = "Cuisines Name", y = "Total Number",
    #                             template = 'seaborn',
    #                             title = 'Top Rated 10 Cuisines', 
                                
    #                                     )

    # st.plotly_chart(fig) 
    pass







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
       st.title("")
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