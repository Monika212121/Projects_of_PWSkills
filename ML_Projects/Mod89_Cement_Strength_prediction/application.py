from flask import Flask, request, render_template

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline 

application = Flask(__name__)

app = application

@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        new_data = CustomData(
            cement = float(request.form.get('Cement (component 1)(kg in a m^3 mixture)')),
            blast_furnace_slag = float(request.form.get('Blast Furnace Slag (component 2)(kg in a m^3 mixture)')),
            fly_ash = float(request.form.get('Fly Ash (component 3)(kg in a m^3 mixture)')),
            water = float(request.form.get('Water (component 4)(kg in a m^3 mixture)')),
            superplasticizer = float(request.form.get('Superplasticizer (component 5)(kg in a m^3 mixture)')),
            coarse_aggregate = float(request.form.get('Coarse Aggregate (component 6)(kg in a m^3 mixture)')),
            fine_aggregate = float(request.form.get('Fine Aggregate (component 7)(kg in a m^3 mixture)')),
            age = int(request.form.get('Age (day)')),
        )

        new_data_df = new_data.get_data_as_dataframe()

        prediction_pipeline_obj = PredictPipeline()
        predicted_cement_strength = prediction_pipeline_obj.predict_cement_strength(new_data_df)

        return render_template('form.html', final_result = predicted_cement_strength)



if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)


# Cement (component 1)(kg in a m^3 mixture),Blast Furnace Slag (component 2)(kg in a m^3 mixture),Fly Ash (component 3)(kg in a m^3 mixture),Water  (component 4)(kg in a m^3 mixture),Superplasticizer (component 5)(kg in a m^3 mixture),Coarse Aggregate  (component 6)(kg in a m^3 mixture),Fine Aggregate (component 7)(kg in a m^3 mixture),Age (day),"Concrete compressive strength(MPa, megapascals) "
