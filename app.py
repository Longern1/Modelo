from flask import Flask, render_template, request
import pandas as pd
from io import BytesIO
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np  # Agregar esta línea para importar numpy
import matplotlib
matplotlib.use('Agg')  # 👈 Esto evita el error de NSWindow
import matplotlib.pyplot as plt
import io
import base64

import os



app = Flask(__name__)

model = None

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/explicacion')
def explicacion():
    return render_template('explicacion.html')

@app.route('/diarios', methods=['GET', 'POST'])
def diarios():
    global model  # Accedemos al modelo global

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or not file.filename.endswith('.csv'):
            return render_template('diarios.html', error="Por favor, sube un archivo CSV válido.")

        try:
            df = pd.read_csv(file, skiprows=10, header=None, encoding='ISO-8859-1')

            if df.shape[1] < 4:
                return render_template('diarios.html', error="El archivo no tiene suficientes columnas de datos.")

            df.columns = ['YEAR', 'MO', 'DY', 'ALLSKY_SFC_UVA']
            df['YEAR'] = df['YEAR'].astype(int)
            df['MO'] = df['MO'].astype(int)
            df['DY'] = df['DY'].astype(int)

            df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(
                columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}
            ), errors='coerce')
            df = df.dropna(subset=['DATE'])
            df['ALLSKY_SFC_UVA'] = df['ALLSKY_SFC_UVA'].replace(-999.0, np.nan)
            df = df.dropna(subset=['ALLSKY_SFC_UVA'])

            df['UVA_BIN'] = (df['ALLSKY_SFC_UVA'] > 20).astype(int)

            # Forzar al menos dos clases
            if df['UVA_BIN'].nunique() < 2:
                first_row = df.iloc[0].copy()
                first_row['UVA_BIN'] = 1 - first_row['UVA_BIN']
                df = pd.concat([df, pd.DataFrame([first_row])], ignore_index=True)

            X = df[['YEAR', 'MO', 'DY']]
            y = df['UVA_BIN']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Entrenamiento del modelo solo si no está entrenado
            if model is None:
                model = LogisticRegression()
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Gráfico de matriz de confusión
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title("Matriz de Confusión")
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Realidad')
            plt.colorbar(im)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            # Análisis detallado
            count_high = df['UVA_BIN'].sum()
            count_low = len(df) - count_high

            summary_by_year = df.groupby('YEAR')['UVA_BIN'].mean().reset_index()
            summary_by_year.columns = ['Año', 'Porcentaje de Alta Radiación']
            summary_by_year['Porcentaje de Alta Radiación'] *= 100
            yearly_summary = summary_by_year.to_dict(orient='records')

            coefs = dict(zip(X.columns, model.coef_[0]))

            # Interpretación segura de la matriz de confusión
            if cm.shape == (2, 2):
                confusion_explained = (
                    f"- {cm[0][0]} días correctamente clasificados como BAJA radiación UVA.\n"
                    f"- {cm[1][1]} días correctamente clasificados como ALTA radiación UVA.\n"
                    f"- {cm[0][1]} días mal clasificados como ALTA radiación.\n"
                    f"- {cm[1][0]} días mal clasificados como BAJA radiación."
                )
            else:
                confusion_explained = (
                    f"La matriz de confusión no tiene el tamaño esperado (2x2). "
                    f"Esto puede deberse a que los datos de prueba contienen solo una clase. "
                    f"La matriz resultante fue: {cm.tolist()}"
                )

            return render_template(
                'diarios.html',
                accuracy=accuracy,
                img_str=img_str,
                confusion_matrix=cm.tolist(),
                confusion_explained=confusion_explained,
                count_high=count_high,
                count_low=count_low,
                coefs=coefs,
                yearly_summary=yearly_summary
            )

        except Exception as e:
            return render_template('diarios.html', error=f"Error al procesar el archivo: {str(e)}")

    return render_template('diarios.html')


@app.route('/predict_uv', methods=['POST'])
def predict_uv():
    global model  # Usamos el modelo global cargado previamente

    try:
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        input_data = [[year, month, day]]
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]  # Probabilidad de clase 1 (Alta UVA)

        result = 'Alta' if prediction[0] == 1 else 'Baja'
        enfermedad_piel = 'Riesgo de enfermedad en la piel' if prediction[0] == 1 else 'Sin riesgo'
        proba_text = f"{prediction_proba * 100:.2f}% de probabilidad de alta radiación UVA"

        prediction_data = {
            'fecha': f'{day}-{month}-{year}',
            'prediction': result,
            'enfermedad_piel': enfermedad_piel,
            'proba': proba_text
        }

        return render_template('diarios.html', prediction=prediction_data)

    except Exception as e:
        return render_template('diarios.html', error=f"Error al procesar la predicción: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True)
