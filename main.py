import numpy as np
import pandas as pd
import pennylane as qml
import pickle
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import BallTree
from folium import plugins
from folium.plugins import TimestampedGeoJson, MarkerCluster
import branca.colormap as cm
from datetime import datetime
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict
import logging
from pathlib import Path
from geopy.distance import geodesic

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

    def prepare_data(self, data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        features = [
            'LATITUDE', 'LONGITUDE', 'YEAR',
            'avg_temp_c', 'precipitation_mm'
        ]
        
        processed_data = data_df[features + ['sum.allrawdata.ABUNDANCE']].copy()
        processed_data = processed_data.dropna()
        
        self.base_year = processed_data['YEAR'].min()
        processed_data['YEARS_SINCE_START'] = processed_data['YEAR'] - self.base_year
        
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
        
    def create_climate_biodiversity_map(self, climate_variable: str = 'avg_temp_c', output_path: str = 'climate_biodiversity_map.html'):

        m = folium.Map(
            location=[self.default_lat, self.default_lon],
            zoom_start=4,
            tiles='CartoDB positron'
        )
        
        # feature groups for toggling
        biodiversity_fg = folium.FeatureGroup(name='Biodiversity')
        climate_fg = folium.FeatureGroup(name='Climate Data')
        

        heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['sum.allrawdata.ABUNDANCE']] 
                    for _, row in self.data_df.iterrows()]
        plugins.HeatMap(heat_data, name='Biodiversity').add_to(biodiversity_fg)
        

        climate_data = [[row['LATITUDE'], row['LONGITUDE'], row[climate_variable]] 
                       for _, row in self.data_df.iterrows() if pd.notna(row[climate_variable])]
        

        climate_values = [d[2] for d in climate_data]
        colormap = cm.LinearColormap(
            colors=['blue', 'yellow', 'red'],
            vmin=min(climate_values),
            vmax=max(climate_values)
        )
        
        for point in climate_data:
            folium.CircleMarker(
                location=[point[0], point[1]],
                radius=8,
                color=None,
                fill=True,
                fill_color=colormap(point[2]),
                fill_opacity=0.6,
                popup=f"{climate_variable}: {point[2]:.1f}"
            ).add_to(climate_fg)
        
        # layers
        biodiversity_fg.add_to(m)
        climate_fg.add_to(m)
        colormap.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        self._add_title(m, f"Biodiversity and {climate_variable} Distribution")
        m.save(output_path)
        return m

    def create_future_timeline_map(self, model: QuantumModel, years_ahead: int = 5, prediction_grid_size: int = 20, output_path: str = 'future_biodiversity_map.html'):

        m = folium.Map(
            location=[self.default_lat, self.default_lon],
            zoom_start=4,
            tiles='CartoDB positron'
        )
        
        # prep existing data
        years = sorted(self.data_df['YEAR'].unique())
        current_year = max(years)
        future_years = range(current_year + 1, current_year + years_ahead + 1)
        all_years = years + list(future_years)
        
        # prediction grid
        lat_range = np.linspace(
            self.data_df['LATITUDE'].min(), 
            self.data_df['LATITUDE'].max(), 
            prediction_grid_size
        )
        lon_range = np.linspace(
            self.data_df['LONGITUDE'].min(), 
            self.data_df['LONGITUDE'].max(), 
            prediction_grid_size
        )
        
        features = []
        style_dict = {}
        
        # historical data points
        logger.info("Processing historical data points...")
        for year in years:
            year_data = self.data_df[self.data_df['YEAR'] == year]
            
            for idx, row in year_data.iterrows():
                feature_id = f"historical_{idx}"
                
                if feature_id not in [f['id'] for f in features]:
                    feature = {
                        'type': 'Feature',
                        'id': feature_id,
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [row['LONGITUDE'], row['LATITUDE']]
                        },
                        'properties': {
                            'times': [str(year)],
                            'popup': (f"Historical Data<br>"
                                    f"Year: {year}<br>"
                                    f"Abundance: {row['sum.allrawdata.ABUNDANCE']:.2f}")
                        }
                    }
                    features.append(feature)
                
                abundance = row['sum.allrawdata.ABUNDANCE']
                max_abundance = self.data_df['sum.allrawdata.ABUNDANCE'].max()
                opacity = min(0.8, abundance / max_abundance)
                
                style_dict[feature_id] = {
                    str(year): {
                        'color': 'blue',
                        'fillColor': 'blue',
                        'fillOpacity': opacity,
                        'radius': np.sqrt(abundance)/2,
                        'weight': 1
                    }
                }
        
        logger.info("Generating future predictions...")
        prediction_counter = 0
        for lat in lat_range:
            for lon in lon_range:
                feature_id = f"prediction_{prediction_counter}"
                prediction_counter += 1
                
                # climate data for this location - or as close as possible
                nearest_weather = self._get_nearest_weather_data(lat, lon)
                
                try:
                    # predictions for each future year
                    predictions = {}
                    for year in future_years:
                        predicted_abundance = model.predict_future_abundance(
                            location=[lat, lon],
                            years_ahead=year - current_year,
                            climate_data=nearest_weather
                        )
                                                
                        predictions[year] = predicted_abundance
                    
                    if predictions:  # create if we have valid predictions
                        feature = {
                            'type': 'Feature',
                            'id': feature_id,
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [lon, lat]
                            },
                            'properties': {
                                'times': [str(year) for year in predictions.keys()],
                                'popup': 'Predicted Data<br>' + '<br>'.join(
                                    f"Year {year}: {abundance:.2f}"
                                    for year, abundance in predictions.items()
                                )
                            }
                        }
                        features.append(feature)
                        
                        style_dict[feature_id] = {
                            str(year): {
                                'color': 'red',
                                'fillColor': 'red',
                                'fillOpacity': min(0.8, abundance / max_abundance),
                                'radius': np.sqrt(abundance)/2,
                                'weight': 1
                            }
                            for year, abundance in predictions.items()
                        }
                
                except Exception as e:
                    logger.warning(f"Prediction failed for location [{lat}, {lon}]: {str(e)}")
        
        logger.info("Creating time slider visualisation...")
        geojson_data = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        time_slider = TimestampedGeoJson(
            geojson_data,
            period='P1Y',  # one year per step - may scale thid depending on accuracy
            add_last_point=False,
            auto_play=True,
            loop=True,
            max_speed=1,
            loop_button=True,
            date_options='YYYY',
            time_slider_drag_update=True,
            duration='P1Y'
        )
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; 
                    right: 50px; 
                    z-index: 1000; 
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    border: 2px solid gray;">
            <p><strong>Legend</strong></p>
            <p><span style="color: blue;">●</span> Historical Data</p>
            <p><span style="color: red;">●</span> Predicted Data</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        time_slider.add_to(m)
        self._add_title(m, f"Biodiversity Distribution: Historical to {max(future_years)}")
        
        m.save(output_path)
        logger.info(f"Future timeline map saved to {output_path}")
        return m
    
    def _get_nearest_weather_data(self, lat: float, lon: float) -> dict:

        if not hasattr(self, 'data_df') or 'avg_temp_c' not in self.data_df.columns:
            return {}
            
        # distances to all weather stations - pythag dist
        distances = np.sqrt(
            (self.data_df['LATITUDE'] - lat)**2 + 
            (self.data_df['LONGITUDE'] - lon)**2
        )
        
        # nearest station's data
        nearest_idx = distances.argmin()
        nearest_row = self.data_df.iloc[nearest_idx]
        
        return {
            'temperature': nearest_row.get('avg_temp_c', None),
            'precipitation': nearest_row.get('precipitation_mm', None)
        }

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


class ModelEvaluator:
    def __init__(self, model: QuantumModel):
        self.model = model
        
    def evaluate_regression_metrics(self, X_true: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        # predictions
        y_pred = self.model.predict(X_true)
        
        # original scale
        y_true_orig = self.model.target_scaler.inverse_transform(y_true)
        y_pred_orig = self.model.target_scaler.inverse_transform(y_pred.reshape(-1, 1))
        
        metrics = {
            'mse': mean_squared_error(y_true_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_true_orig, y_pred_orig),
            'r2': r2_score(y_true_orig, y_pred_orig)
        }
        
        return metrics
    
    def create_performance_plot(self, X_test: np.ndarray, y_test: np.ndarray, year: int, output_dir: str = './') -> None:

        y_pred = self.model.predict(X_test)
        
        # transform to original scale
        y_test_orig = self.model.target_scaler.inverse_transform(y_test)
        y_pred_orig = self.model.target_scaler.inverse_transform(y_pred.reshape(-1, 1))
        
        errors = y_pred_orig.flatten() - y_test_orig.flatten()

        df = pd.DataFrame({
            'Latitude': self.model.feature_scaler.inverse_transform(X_test)[:, 0],
            'Longitude': self.model.feature_scaler.inverse_transform(X_test)[:, 1],
            'Error': errors
        })

        df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
        geo_df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326") 

        world = gpd.read_file('./earth/ne_110m_admin_0_countries.shp')

        fig, ax = plt.subplots(figsize=(15, 10))
        world.boundary.plot(ax=ax, linewidth=1)
        geo_df.plot(column='Error', ax=ax, legend=True, cmap='coolwarm', markersize=20, legend_kwds={'label': "Prediction Error", 'orientation': "horizontal"})

        plt.title(f'Prediction Error on World Map - Year {year}')
        plt.savefig(f'{output_dir}/error_heatmap_{year}.png', bbox_inches='tight', dpi=300)
        plt.close()
    

def get_nearest_stations(bio_df: pd.DataFrame, weather_df: pd.DataFrame, k: int = 1) -> pd.DataFrame:

    # haversine was taking too long and wanted 24 tb of disk space so trying sklearn BallTree for approx nearest station

    weather_coords = np.radians(weather_df[['latitude', 'longitude']].values)
    tree = BallTree(weather_coords, metric='haversine')
    
    # convert coordinates to radians
    bio_coords = np.radians(bio_df[['LATITUDE', 'LONGITUDE']].values)

    # k nearest neighbors for each location
    distances, indices = tree.query(bio_coords, k=k)
    
    # convert to km
    distances_km = distances * 6371.0 # radius of earth in km
    
    merged_records = []
    
    for i in range(len(bio_df)):
        bio_row = bio_df.iloc[i]
        
        for j in range(k):
            weather_row = weather_df.iloc[indices[i, j]]
            
            merged_record = {
                **bio_row.to_dict(),
                'station_id': weather_row['station_id'],
                'station_distance_km': distances_km[i, j],
                'station_latitude': weather_row['latitude'],
                'station_longitude': weather_row['longitude'],
                'avg_temp_c': weather_row['avg_temp_c'],
                'precipitation_mm': weather_row['precipitation_mm'],
            }
            merged_records.append(merged_record)
    
    result_df = pd.DataFrame(merged_records)
    
    return result_df


def print_performance_summary(metrics: Dict[str, float]) -> None:
    print("\nModel Performance Summary")
    print("========================")
    print(f"R² Score: {metrics['r2']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")

def main():

    data_path = "BioTIMEQuery_24_06_2021.csv"
    model_path = "biodiversity_model.pkl"    
    #number_of_years = 1 # edit this if it is taking too long
    year = 2012

    try:
        # prevent dtype warning - probably should set to true on your machine Winnie
        data_df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
        logger.info("bio data loaded successfully")
        logger.info(f"number of rows loaded {len(data_df.index)}")

        weather_df = pd.read_parquet('daily_weather.parquet', engine='fastparquet')
        logger.info("weather data loaded successfully")
        logger.info(f"number of rows loaded {len(weather_df.index)}")

        # join weather data to stations/cities as we need the lat/long
        stations_df = pd.read_csv('cities.csv')
        weather_df = weather_df.merge(stations_df[['station_id', 'latitude', 'longitude']], on='station_id', how='left')
        logger.info(f"joined weather table {weather_df.head(10)}")

        data_df.columns = data_df.columns.str.strip()
        weather_df.columns = weather_df.columns.str.strip()

        data_df = data_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        weather_df = weather_df.dropna(subset=['latitude', 'longitude'])
        
        #max_year = data_df['YEAR'].max()
        data_df = data_df[data_df['YEAR'] == year]
        #logger.info(f"filtered data to last {number_of_years} years from {max_year}")
        logger.info(f"number of biodata rows to be processed {len(data_df.index)}")

        if not pd.api.types.is_datetime64_any_dtype(weather_df['date']):
            weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        weather_df = weather_df[weather_df['date'].dt.year == year]
        logger.info(f"number of weather rows to be processed {len(weather_df.index)}")

        merged_df = get_nearest_stations(data_df, weather_df)
        logger.info(f"Merged dataset contains {len(merged_df)} records")
        
        # create maps with climate data - avg temp and precipitation, could also do min/max temp or windspeed? 
        # depending on how much data we have
        bio_map = BiodiversityMap(merged_df)
        # bio_map.create_climate_biodiversity_map(climate_variable='avg_temp_c', output_path='temperature_biodiversity_map.html')
        # bio_map.create_climate_biodiversity_map(climate_variable='precipitation_mm', output_path='precipitation_biodiversity_map.html')
        
        # train quantum model with climate data
        model = QuantumModel()
        
        X_scaled, y_scaled = model.prepare_data(merged_df)
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=123)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)
        if QuantumModel.model_already_exists(model_path):
            model = QuantumModel.load_model(model_path)
            logger.info(f"model loaded from file")
        else:
            logger.info(f"model being trained")
            weights, losses = model.train(X_train, y_train)
            model.save_model(model_path)
            model.save_model(model_path)
            logger.info(f"model saved to file")

        # bio_map.create_future_timeline_map(
        #     model=model,
        #     years_ahead=5,
        #     prediction_grid_size=20,
        #     output_path='future_biodiversity_timeline.html')

        evaluator = ModelEvaluator(model)
    
        metrics = evaluator.evaluate_regression_metrics(X_test, y_test)
        print_performance_summary(metrics)
        
        # plot performance
        evaluator.create_performance_plot(
            X_test=X_test,
            y_test=y_test,
            year=year,
            output_dir='./model_evaluation'
        )
        
    except Exception as e:
        logger.error(f"An error occurred, don't panic: {str(e)}")
        raise

if __name__ == "__main__":
    main()