from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods= ['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('form.html')
    
        else:
            # Extract input values from form
            data = CustomData(
                mean_radius=float(request.form['mean_radius']),
                mean_texture=float(request.form['mean_texture']),
                mean_perimeter=float(request.form['mean_perimeter']),
                mean_area=float(request.form['mean_area']),
                mean_smoothness=float(request.form['mean_smoothness']),
                mean_compactness=float(request.form['mean_compactness']),
                mean_concavity=float(request.form['mean_concavity']),
                mean_concave_points=float(request.form['mean_concave_points']),
                mean_symmetry=float(request.form['mean_symmetry']),
                mean_fractal_dimension=float(request.form['mean_fractal_dimension']),
                radius_error=float(request.form['radius_error']),
                texture_error=float(request.form['texture_error']),
                perimeter_error=float(request.form['perimeter_error']),
                area_error=float(request.form['area_error']),
                smoothness_error=float(request.form['smoothness_error']),
                compactness_error=float(request.form['compactness_error']),
                concavity_error=float(request.form['concavity_error']),
                concave_points_error=float(request.form['concave_points_error']),
                symmetry_error=float(request.form['symmetry_error']),
                fractal_dimension_error=float(request.form['fractal_dimension_error']),
                worst_radius=float(request.form['worst_radius']),
                worst_texture=float(request.form['worst_texture']),
                worst_perimeter=float(request.form['worst_perimeter']),
                worst_area=float(request.form['worst_area']),
                worst_smoothness=float(request.form['worst_smoothness']),
                worst_compactness=float(request.form['worst_compactness']),
                worst_concavity=float(request.form['worst_concavity']),
                worst_concave_points=float(request.form['worst_concave_points']),
                worst_symmetry=float(request.form['worst_symmetry']),
                worst_fractal_dimension=float(request.form['worst_fractal_dimension'])
        )

        new_data = data.get_data_as_dataframe()
        prediction_pipeline = PredictPipeline()

        predicted_value = prediction_pipeline.predict_disease(new_data)

        result = "Malignant" if predicted_value[0] == 0 else "Benign"
        return render_template('result.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")


if __name__=="__main__":
    app.run(host= '0.0.0.0', debug = True)


