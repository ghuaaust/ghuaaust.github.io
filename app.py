import flask
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use pickle to load in the pre-trained model
model = pickle.load(open('model/CircularCylinder_mCp.pkl', 'rb'))   
scalermodel = pickle.load(open('model/scalermodel_mCp.pkl', 'rb'))   

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Diameter = flask.request.form['Diameter']
        TurbulenceIntensity = flask.request.form['TurbulenceIntensity']
        Windspeed = flask.request.form['Windspeed']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[Diameter, TurbulenceIntensity, Windspeed]],
                               columns=['Diameter', 'TurbulenceIntensity', 'Windspeed'],
                               dtype=float,
                               index=['input'])

        TurbulenceIntensity=input_variables['TurbulenceIntensity'][0]
        Re=input_variables['Windspeed'][0]*input_variables['Diameter'][0]/1.5111E-5

        # prediction = input_variables['Diameter'][0]+input_variables['TurbulenceIntensity'][0]+input_variables['Windspeed'][0]

        x_new=[]
        y_gbrt=[]
        for ang in range(0, 181, 1):
            x_new.append([Re,TurbulenceIntensity,ang])
        # x_new=np.array([Re,TurbulenceIntensity,90]).reshape(1,3)
        x_new1 = scalermodel.transform(x_new)


       # Get the model's prediction
        prediction = model.predict(x_new1)
        
        plt.figure(1, figsize=(5,4))
        plt.rcParams["font.family"] = "times new roman"
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams['lines.markersize'] = 7
        plt.rcParams['axes.labelsize'] = 12
        ang=np.arange(0,181,1)
        plt.plot(ang, prediction, 'b-')
        plt.ylabel('Mean Cp')
        plt.xlabel(r"$\theta (^o)$")  
        plt.axis([0, 180, -3.0, 1])
        plt.xticks(np.arange(0,210,30))
        plt.yticks(np.arange(-3.0,1.5,0.5))
        plt.savefig("model/meanCp.png", dpi=600, bbox_inches='tight')

        prediction = np.sum(prediction)  
        # Render the form again, but add in the prediction and remind user of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Diameter':Diameter,
                                                     'TurbulenceIntensity':TurbulenceIntensity,
                                                     'Windspeed':Windspeed},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
    