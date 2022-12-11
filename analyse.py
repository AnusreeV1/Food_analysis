from flask import Flask, render_template,redirect, url_for
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html',adv='',dis='')

import numpy as np #standard
import plotly.express as px  #plots and graphing lib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def dic_maker(arr):
  """ dis takes in arr [[prob(1),prob(2),prob(3)......prob(n)]]
   and outputs [(1,prob(1)),(2,prob(2))]
   (basically some formatting to make life easier)"""
  dict_ = {}
  for ind in range(len(arr[0])):
    dict_[ind] = arr[0][ind]
  return sorted(dict_.items(), key=lambda x: x[1],reverse=True)[:3]


def dic_maker_tuple(tuple_arr,target_dict):
  """ takes in [(x,y),(a,b)]
      outputs {x:y,a:b} (basically some formatting to make life easier)
  """
  dict_ = {}
  for tuple_ in tuple_arr:
    dict_[target_dict[tuple_[0]]] = tuple_[1]
  return dict_


def inception_no_gen(image,model_saved,target_dict):
  """ 
  prediction happens in this function
  super important, takes in image_path (/content/test_1/test/111.jpg)
  outputs: {1:prob(1),2:prob(2)}
  """
  #image_1 = tensorflow.keras.preprocessing.image.load_img(image_path)

  input_arr = tensorflow.keras.preprocessing.image.img_to_array(image)
  input_arr = preprocess_input(input_arr)
  input_arr = tensorflow.image.resize(input_arr,size = (256,256))
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model_saved.predict(input_arr)
  print("\nPredictions probability: ")
  print(predictions)
  pred = predictions[0]
  pred_dict = {v:k for v,k in enumerate(pred)}
  pred_dict = sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)
  print("\nSorted list based on probability: ")
  print(pred_dict)
  print("\nPredicted food id: ")
  print(pred_dict[0])
  preds = dic_maker_tuple(dic_maker(predictions),target_dict)
  #res = list(pred_dict.keys())[0]
  #print(res)
  return (preds,pred_dict[0])

def plot_pred_final(test_imgs,model_saved,target_dict):
  """
  dis takes in {1:prob(1),2:prob(2)}
  and plots a SUPER NORMIE PLOT to make it easier for SRM FACULTY(or they might flip out like the bunch of idiots they are)
  """
  #test_imgs = glob(image_path_custom + '/*/*.jpeg')
  fig = make_subplots(rows = 2, cols = 2)
  pred_list,ingred = inception_no_gen(test_imgs,model_saved,target_dict)
  fig.append_trace(go.Image(z = np.array(test_imgs)),1,1)
  fig.append_trace(go.Bar(y = list(pred_list.keys()), x = list(pred_list.values()),orientation = 'h'),1,2)
  fig.update_layout(width = 1750, height = 800,title_text = "Predictions",showlegend = False)
  return fig,ingred

def predictions(id):
  adv_dict = {0:["Contains Fiber","Has a Prebiotic Effect","Low in Fat"],
              1:["Source of nutrients like calcium, vitamin D, potassium, and protein","These are all essential to good bone health"],
              2:["Lowers your blood pressure","Inspires You To Be Active"],
              3:["Contains vitamins, protein, calcium","Helps maintain your eyesight"],
              4:["Minimal loss of food quality as hot fat seals the food surface"],
              5:["Good source of protein, vitamins and several other growth factors which are essential for the cell growth and immunity",],
              6:["Sustained Energy: Carbohydrates like pasta provide glucose, the crucial fuel for your brain and muscles","Low Sodium and Cholesterol Free: If you're watching your cholesterol levels, pasta is perfect for you, being very low in sodium and cholesterol free"],
              7:["Natural Anti-Inflammatory and Gluten Free","Improves Nervous System Health","Good Source of Energy"],
              8:["Low in saturated fats, high in protein, and packed full of important nutrients including omega-3 fatty acids, vitamin A, and B vitamins"],
              9:["Helps Lose Weight","Good For Digestion","Vitamins and Minerals Stay Intact"],
              10:["Great source of vitamins and minerals","Lots of fiber","Low-calorie and low-fat"]
              }
  
  dis_dict = {0:["Celiac disease (Over consumption of gluten)","Allergic reaction (Moldy bread)","Respiratory issues (Moldy bread)"],
              1:["High in saturated fats(Risk for coronary heart disease",],
              2:["Diabetes(Increase in insulin levels)","Excess weight gain and obesity"],
              3:["High consumption of egg lead to heart disease, Diabetes, etc due to higher cholesterol and saturated fat",],
              4:["Fried foods are typically high in trans fats which are associated with an increased risk of many diseases, including heart disease, cancer, diabetes and obesity"],
              5:["Heart disease","Cancer","Diabetes"],
              6:["Pasta is high in carbs, which can be bad for you when consumed in large amounts.","It also contains gluten, a type of protein that causes issues for those who are gluten-sensitive"],
              7:["The rice plant accumulates more arsenic which is linked to an increased risk of heart disease and some types of cancer"],
              8:["Seafood sometimes may contain high mercury content","Unknown parasites get attached to seafood"],
              9:["High in sodium - High sodium diets are associated with health risks such as high blood pressure"],
              10:["Diarrhea, reflux and bloating are all potential side effects of eating too much fruit","High blood sugar is another side effect of fruit consumption"]
              }
  print("\nAdvantages:")
  print(adv_dict[id])
  print("\nDisdvantages:")
  print(dis_dict[id])
  return(adv_dict[id],dis_dict[id])

@app.route('/food_analyse/<path:id>')
def food_analyse(id):
    print("heyy")
    model_saved = tensorflow.keras.models.load_model("C:/Users/Anusree/Documents/Sem/Sem 7/4.SNA/Project/2/Food_analysis/inception_food_rec_50epochs.h5")
    target_dict = {0:"Bread",1:"Dairy product",2:"Dessert",3:"Egg",4:"Fried food",
                    5:"Meat",6:"Noodles/Pasta",7:"Rice",8:"Seafood",9:"Soup",10:"Veggies/Fruits"}

    #image_path = ss.file_uploader("Drop the image file here: ", type = ["jpg"])
    image_path = id
    if image_path:
        image = Image.open(image_path)
        preds,ingred = plot_pred_final(image,model_saved,target_dict)
        print("\nPredicted food: ")
        print(target_dict[ingred[0]])
        adv, dis = predictions(ingred[0])
        #ss.plotly_chart(preds,use_container_width=True)
    return render_template('modal.html',food=target_dict[ingred[0]],adv=adv,dis=dis)
    #return redirect(url_for('index')+'#popup1')
  
@app.route('/redirect')
def redirect():
      return render_template('index.html')

if __name__ == '__main__':
      app.run(debug=True)




