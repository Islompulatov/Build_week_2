from tokenize import group
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button, ButtonBehavior 
from datetime       import datetime
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from matplotlib.pyplot import text
import joblib
import pandas as pd

df_test=pd.read_csv('To_test.csv')
df=df_test.drop('Unnamed: 0', axis=1)
class CalorieCalculator(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols=1

        self.top_grid=GridLayout()
        self.top_grid.cols=2

        # self.check=GridLayout()
        # self.check.cols=2
        #  # Add checkbox, widget and labels
        # self.check.add_widget(Label(text ='Male'))
        # self.gender= CheckBox(active = True)
        # self.check.add_widget(self.gender)
 
        # self.check.add_widget(Label(text ='Female'))
        # self.gender = CheckBox(active = True)
        # self.check.add_widget(self.gender)

        

        self.top_grid.add_widget(Label(text='Age '))
        self.age=TextInput(multiline=False)
        self.top_grid.add_widget(self.age)

        
        self.top_grid.add_widget(Label(text='Height ()'))
        self.high=TextInput(multiline=False)
        self.top_grid.add_widget(self.high)

       
        self.top_grid.add_widget(Label(text='Weight(kg)'))
        self.weight=TextInput(multiline=False)
        self.top_grid.add_widget(self.weight)

        #self.add_widget(self.check)
        self.add_widget(self.top_grid)

        
        self.submit=Button(text='SUBMIT', font_size=32)
        self.submit.bind(on_press=self.cal_calorie)
        self.add_widget(self.submit)

    
    
    def cal_calorie(self, instance):
        age=int(self.age.text)
        height= int(self.high.text)
        weight= int(self.weight.text)
        model=joblib.load("model.pkl")
        prediction=model.predict(df)
        #gender=self.gender.text
        # if gender=='Female':
        #      BMR = (9.56*weight)+(1.8*height)-(4.68*age)+655
        # else:
        #     BMR = (13.75*weight)+(5*height)-(6.76*age)+66
        BMR = (9.56*weight)+(1.8*height)-(4.68*age)+655
        for i in range(prediction.shape[0]):
        
            if i==0:

                calorie=BMR*1.1
                act= f'You are either on a train or Bus\n and your Calories Burnt is: {str(int(calorie))}cals '

                self.add_widget(Label(text = act))
            elif i==1:

                calorie=BMR*1.2
                act= f'You are Still and doing Nothing\n and your Calories Burnt is: {str(int(calorie))}cals '

                self.add_widget(Label(text = act))

            #self.add_widget(Label(text = "You are " + str(int(calorie)) + ""))
            else: 
                calorie=BMR*1.372
                act= f'You are walking\n and your Calories Burnt is: {str(int(calorie))}cals '

                self.add_widget(Label(text = act))

    
class MyApp(App):

    def build(self):
        return CalorieCalculator()



if __name__ == "__main__":
    MyApp().run()