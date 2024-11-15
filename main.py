import numpy as np
import pandas as pd
import pennylane as qml
import pickle
import folium
from folium import plugins
import branca.colormap as cm
from datetime import datetime
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import logging
from pathlib import Path

# logging as it's really annoying failing halfway through and not realising why...
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumModel:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.weights = None
        self.circuit = None
        self.base_year = None
    
    @staticmethod
    def model_already_exists(filepath: str) -> bool:
        return Path(filepath).exists()

    def _create_circuit(self):

        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev)
        def circuit(inputs, weights):
            # encode inputs
            for i in range(self.n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
            
            # variational layers
            for layer in range(self.n_layers):
                # entangling layer
                for i in range(self.n_qubits):
                    qml.RZ(weights[layer, i], wires=i)
                    
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # rotation layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer + self.n_layers, i], wires=i)
            
            return qml.expval(qml.PauliZ(0))

        logger.info(f"Circuit created with n_qubits: {self.n_qubits} and layers: {self.n_layers}")
        return circuit

    def _get_climate_data(self, lat: float, lon: float, year: int) -> Dict:

        # getting rate limited so for now, just returning some mock data whilst I try and find a dataset with lat/long


        # base_url = "https://archive-api.open-meteo.com/v1/archive"
        # start_date = f"{year}-01-01"
        # end_date = f"{year}-12-31"
        
        # params = {
        #     "latitude": lat,
        #     "longitude": lon,
        #     "start_date": start_date,
        #     "end_date": end_date,
        #     "daily": "temperature_2m_mean,precipitation_sum"
        # }
        
        # try:
        #     response = requests.get(base_url, params=params)
        #     data = response.json()
            
        #     # yearly averages
        #     temp_mean = np.mean(data['daily']['temperature_2m_mean'])
        #     precip_sum = np.sum(data['daily']['precipitation_sum'])

        #     logger.info(f"fetched average climate data for year {year}. temp_mean: {temp_mean} precip_sum: {precip_sum}")
        #     return {
        #         "temperature": temp_mean,
        #         "precipitation": precip_sum
        #     }
        # except Exception as e:
        #     logger.error(f"Error fetching climate data: {e}")
        #     return {"temperature": 0, "precipitation": 0}

        return {"temperature": 10, "precipitation": 10}

    def prepare_data(self, data_df: pd.DataFrame, include_climate: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        features = ['LATITUDE', 'LONGITUDE', 'YEAR']
        processed_data = data_df[features + ['sum.allrawdata.ABUNDANCE']].dropna()
        
        self.base_year = processed_data['YEAR'].min()
        processed_data['YEARS_SINCE_START'] = processed_data['YEAR'] - self.base_year
        
        if include_climate:
            climate_data = []
            for _, row in processed_data.iterrows():
                climate = self._get_climate_data(row['LATITUDE'], row['LONGITUDE'], int(row['YEAR']))
                climate_data.append(climate)
            
            processed_data['TEMPERATURE'] = [d['temperature'] for d in climate_data]
            processed_data['PRECIPITATION'] = [d['precipitation'] for d in climate_data]
            features.extend(['TEMPERATURE', 'PRECIPITATION'])
        
        X = processed_data[features]
        y = processed_data['sum.allrawdata.ABUNDANCE']
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
        
        return X_scaled, y_scaled

    def train(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 100) -> Tuple[np.ndarray, List[float]]:

        self.circuit = self._create_circuit()
        
        self.weights = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=(2 * self.n_layers, self.n_qubits)
        )
        
        opt = qml.GradientDescentOptimizer(stepsize=0.01)
        losses = []
        
        for epoch in range(n_epochs):
            batch_loss = 0
            for X_batch, y_batch in zip(X_train, y_train):
                prediction = self.circuit(X_batch, self.weights)
                loss = np.abs(prediction - y_batch[0])
                
                self.weights = opt.step(lambda w: self.circuit(X_batch, w), self.weights)
                batch_loss += loss
                
            avg_loss = batch_loss / len(X_train)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return self.weights, losses

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.circuit is None or self.weights is None:
            raise ValueError("Model needs to be trained first")
        
        predictions = []
        for x in X:
            pred = self.circuit(x, self.weights)
            predictions.append(pred)
        
        return np.array(predictions)

    def predict_future_abundance(self, location: List[float], years_ahead: int, 
                               include_climate: bool = True) -> float:

        if self.circuit is None or self.weights is None:
            raise ValueError("Model needs to be trained first")
        
        future_year = datetime.now().year + years_ahead
        base_input = [
            location[0],  # lat
            location[1],  # long
            future_year - self.base_year  # scaled future year
        ]
        
        if include_climate:
            climate = self._get_climate_data(location[0], location[1], future_year)
            base_input.extend([climate['temperature'], climate['precipitation']])
        
        scaled_input = self.feature_scaler.transform([base_input])[0]
        prediction = self.predict([scaled_input])[0]
        
        # get actual abundance
        return self.target_scaler.inverse_transform([[prediction]])[0][0]

    def save_model(self, filepath: str):
        model_state = {
            'weights': self.weights,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'base_year': self.base_year
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

    @classmethod
    def load_model(cls, filepath: str) -> 'QuantumModel':

        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(n_qubits=model_state['n_qubits'], n_layers=model_state['n_layers'])
        model.weights = model_state['weights']
        model.feature_scaler = model_state['feature_scaler']
        model.target_scaler = model_state['target_scaler']
        model.base_year = model_state['base_year']
        model.circuit = model._create_circuit()
        
        return model


class BiodiversityMap:

    def __init__(self, data_df: pd.DataFrame):

        self.data_df = data_df
        self.default_lat = data_df['LATITUDE'].mean()
        self.default_lon = data_df['LONGITUDE'].mean()
        
    def create_3d_map(self, year: int, species_filter: str = 'All Species') -> folium.Map:

        filtered_data = self.data_df[self.data_df['YEAR'] == year].copy()
        
        if species_filter != 'All Species':
            filtered_data = filtered_data[filtered_data['GENUS_SPECIES'] == species_filter]
        
        if len(filtered_data) == 0:
            map_center = [self.default_lat, self.default_lon]
        else:
            map_center = [filtered_data['LATITUDE'].mean(), filtered_data['LONGITUDE'].mean()]
        
        m = folium.Map(
            location=map_center,
            zoom_start=4,
            tiles='CartoDB positron',
            attr='CartoDB'
        )
        
        # colourmap for abundance
        max_abundance = filtered_data['sum.allrawdata.ABUNDANCE'].max()
        colourmap = cm.LinearColormap(
            colors=['yellow', 'orange', 'red'],
            vmin=0,
            vmax=max_abundance
        )
        m.add_child(colourmap)
        
        # marker clusters
        marker_cluster = plugins.MarkerCluster().add_to(m)
        for idx, row in filtered_data.iterrows():
            popup_content = f"""
                <b>Species:</b> {row['GENUS_SPECIES']}<br>
                <b>Abundance:</b> {row['sum.allrawdata.ABUNDANCE']:.2f}<br>
                <b>Location:</b> ({row['LATITUDE']:.4f}, {row['LONGITUDE']:.4f})<br>
                <b>Year:</b> {row['YEAR']}
            """
            
            folium.CircleMarker(
                location = [row['LATITUDE'], row['LONGITUDE']],
                radius = np.sqrt(row['sum.allrawdata.ABUNDANCE'])/2,
                popup = folium.Popup(popup_content, max_width=300),
                color = colourmap(row['sum.allrawdata.ABUNDANCE']),
                fill = True,
                fill_opacity = 0.7
            ).add_to(marker_cluster)
        
        self._add_title(m, f"Biodiversity Distribution {year} ({len(filtered_data)} locations)")
        return m

    def create_simple_2d_map(self, output_path: str = 'simple_biodiversity_map.html'):

        m = folium.Map(
            location=[self.default_lat, self.default_lon],
            zoom_start=4,
            tiles='CartoDB positron',
            attr='CartoDB'
        )

        # heatmap layer
        heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['sum.allrawdata.ABUNDANCE']] 
                    for _, row in self.data_df.iterrows()]
        
        plugins.HeatMap(heat_data).add_to(m)
        
        self._add_title(m, f"Overall Biodiversity Distribution (Heatmap)")
        m.save(output_path)
        logger.info(f"Simple 2D map saved to {output_path}")
        return m

    def create_prediction_comparison_map(self, model: QuantumModel, years_ahead: int = 5, output_path: str = 'prediction_map.html'):
                    
        m = folium.Map(
            location=[self.default_lat, self.default_lon],
            zoom_start=4,
            tiles='CartoDB positron',
            attr='CartoDB'
        )

        current_year = self.data_df['YEAR'].max()
        current_data = self.data_df[self.data_df['YEAR'] == current_year]
        
        # feature groups for toggling
        current_fg = folium.FeatureGroup(name=f'Current Data ({current_year})')
        future_fg = folium.FeatureGroup(name=f'Predictions ({current_year + years_ahead})')
        
        # current data points
        for _, row in current_data.iterrows():
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=np.sqrt(row['sum.allrawdata.ABUNDANCE'])/2,
                color='blue',
                popup=f"Current Abundance: {row['sum.allrawdata.ABUNDANCE']:.2f}",
                fill=True,
                fill_opacity=0.7
            ).add_to(current_fg)
        
        lat_range = np.linspace(self.data_df['LATITUDE'].min(), self.data_df['LATITUDE'].max(), 20)
        lon_range = np.linspace(self.data_df['LONGITUDE'].min(), self.data_df['LONGITUDE'].max(), 20)
        
        for lat in lat_range:
            for lon in lon_range:
                try:
                    predicted_abundance = model.predict_future_abundance([lat, lon], years_ahead)
                    if predicted_abundance > 0:  # show positive predictions - might split out negative ones and display too
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=np.sqrt(predicted_abundance)/2,
                            color='red',
                            popup=f"Predicted Abundance: {predicted_abundance:.2f}",
                            fill=True,
                            fill_opacity=0.5
                        ).add_to(future_fg)
                except Exception as e:
                    logger.warning(f"Prediction failed for location [{lat}, {lon}]: {str(e)}")
        
        current_fg.add_to(m)
        future_fg.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        self._add_title(m, f"Current vs Predicted Biodiversity ({years_ahead} years ahead)")
        m.save(output_path)
        logger.info(f"Prediction comparison map saved to {output_path}")
        return m


    def create_time_slider_map(self, output_path: str = 'basic_biodiversity_map.html'):

        years = sorted(self.data_df['YEAR'].unique())
        
        m = folium.Map(
            location=[self.default_lat, self.default_lon],
            zoom_start=4,
            tiles='CartoDB positron',
            attr='CartoDB'
        )
        
        features = []
        style_dict = {}
        
        # group data by year -> features for each location
        for year in years:
            year_data = self.data_df[self.data_df['YEAR'] == year]
            
            for idx, row in year_data.iterrows():
                feature_id = f"location_{idx}"
                
                if feature_id not in [f['id'] for f in features]:
                    feature = {
                        'type': 'Feature',
                        'id': feature_id,
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [row['LONGITUDE'], row['LATITUDE']]
                        },
                        'properties': {
                            'times': []
                        }
                    }
                    features.append(feature)
                
                if feature_id not in style_dict:
                    style_dict[feature_id] = {}
                
                # colour based on abundance
                abundance = row['sum.allrawdata.ABUNDANCE']
                opacity = min(0.8, abundance / self.data_df['sum.allrawdata.ABUNDANCE'].max())
                
                style_dict[feature_id][str(year)] = {
                    'color': 'red',
                    'fillColor': 'red',
                    'fillOpacity': opacity,
                    'radius': np.sqrt(abundance)/2,
                    'weight': 1
                }
                
                for feature in features:
                    if feature['id'] == feature_id:
                        feature['properties']['times'].append(str(year))
                        break
        
        geojson_data = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        plugins.TimeSliderChoropleth(
            geojson_data,
            styledict=style_dict
        ).add_to(m)
        
        # play button
        plugins.FloatImage(
            'https://cdn-icons-png.flaticon.com/512/0/375.png',
            bottom=5,
            left=5
        ).add_to(m)
        
        self._add_title(m, "Biodiversity Distribution Over Time")
        m.save(output_path)
        logger.info(f"Time slider map saved to {output_path}")
        return m

    @staticmethod
    def _add_title(m: folium.Map, title: str):
        title_html = f'''
            <div style="position: fixed; 
                        top: 10px; 
                        left: 50px; 
                        width: 300px; 
                        height: 30px; 
                        z-index:9999; 
                        background-color: white; 
                        font-size:16px;
                        font-weight: bold;
                        padding: 5px;
                        border-radius: 5px;
                        border: 2px solid gray;">
                    {title}
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))


def main():
    data_path = "BioTIMEQuery_24_06_2021.csv"
    model_path = "biodiversity_model.pkl"
    number_of_years = 1 # edit this if it is taking too long
    
    try:
        # prevent dtype warning - probably should set to true on your machine Winnie
        data_df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
        logger.info("data loaded successfully")
        logger.info(f"number of rows loaded {len(data_df.index)}")

        data_df.columns = data_df.columns.str.strip()
        max_year = data_df['YEAR'].max()
        data_df = data_df[data_df['YEAR'] >= (max_year - number_of_years)]
        data_df = data_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        logger.info(f"filtered data to last {number_of_years} years from {max_year}")
        logger.info(f"number of rows to be processed {len(data_df.index)}")

        bio_map = BiodiversityMap(data_df)
        
        # basic time slider map
        bio_map.create_time_slider_map('biodiversity_timeline_map.html')
        logger.info("timeline map created")
        
        # simple 2D heatmap for abundance
        bio_map.create_simple_2d_map('biodiversity_heatmap.html')
        logger.info("heatmap created")
        
        if QuantumModel.model_already_exists(model_path):
            logger.info("loading existing model...")
            model = QuantumModel.load_model(model_path)
        else:
            logger.info("training new model...")
            model = QuantumModel()
            X_scaled, y_scaled = model.prepare_data(data_df, include_climate=True)
            
            # split data
            X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=123)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)
            
            # train model
            weights, losses = model.train(X_train, y_train)
            model.save_model(model_path)
            logger.info("model trained and saved successfully")
        
        # test prediction
        location = [47.4, -95.12]
        years_ahead = 5
        prediction = model.predict_future_abundance(location, years_ahead)
        logger.info(f"Predicted abundance in {years_ahead} years at location "
                   f"{location}: {prediction:.2f}")

         # prediction comparison map
        bio_map.create_prediction_comparison_map(model, years_ahead=5, output_path='biodiversity_prediction_map.html')
        logger.info("comparison map created")
        
    except Exception as e:
        logger.error(f"An error occurred, don't panic: {str(e)}")
        raise

if __name__ == "__main__":
    main()