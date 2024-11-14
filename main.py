# %%
# Install dependencies
%pip install pennylane
%pip install qiskit qiskit_machine_learning
%pip install numpy pandas matplotlib scipy scikit-learn
%pip install folium
%pip install ipywidgets

# %% [markdown]
# # Project - Quantum Approach to Biodiversity Mapping & Predictions
# 
# Figma [here](https://www.figma.com/board/YJkl666NgYY9lzeGnKZ1lw/Quantum-Hackathon-2024?node-id=0-1&node-type=canvas&t=7ul1ZMUwhKcWZdU6-0)
# 
# Goals:  
# 1.  
# 2.  

# %%
# Import libraries
import numpy as np
import pandas as pd
import pennylane as qml
import qiskit
import zipfile
import io
import os
import folium
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQR
from qiskit_machine_learning.circuit.library import RawFeatureVector
from folium.plugins import TimestampedGeoJson, MarkerCluster
from IPython.display import display
from ipywidgets import interact, IntSlider, Select, Layout

# [BioTIME database](https://zenodo.org/records/5026943#.Y9ZAKdJBwUE)
metadata_df  = pd.read_csv("BioTIMEMetadata_24_06_2021.csv", encoding='latin1')
metadata_df.columns = metadata_df.columns.str.strip()

with zipfile.ZipFile("BioTIMEQuery_24_06_2021.zip") as z:
    print(z.namelist())
    with z.open("BioTIMEQuery_24_06_2021.csv") as f:
        data_df = pd.read_csv(f, encoding='latin1')
# data_df = pd.read_csv("BioTIMEQuery_24_06_2021.csv", encoding='latin1')

data_df.columns = data_df.columns.str.strip()

# Filter the dataframe to include only the latest 10 years
latest_year = data_df['YEAR'].max()
filtered_data_df = data_df[data_df['YEAR'] >= (latest_year - 9)]

# Drop the DAY and MONTH columns if they exist
columns_to_drop = ['DAY', 'MONTH']
filtered_data_df = filtered_data_df.drop(columns=[col for col in columns_to_drop if col in filtered_data_df.columns])


default_lat = filtered_data_df['LATITUDE'].mean()
default_lon = filtered_data_df['LONGITUDE'].mean()

def update_map(year, species_filter='All Species'):

    filtered_data = filtered_data_df[filtered_data_df['YEAR'] == year].copy()
    
    # species filter if not "All Species"
    if species_filter != 'All Species':
        filtered_data = filtered_data[filtered_data['GENUS_SPECIES'] == species_filter]
    
    # Use the mean of valid coordinates, if no valid data points -> use default center
    if len(filtered_data) == 0:
        map_center = [default_lat, default_lon]
    else:
        map_center = [filtered_data['LATITUDE'].mean(), filtered_data['LONGITUDE'].mean()]

    m = folium.Map(location=map_center, zoom_start=4)
    
    # add markers with data
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in filtered_data.iterrows():
        # for debugging for now, can make it look nicer later
        popup_content = f"""
            <b>Species:</b> {row['GENUS_SPECIES']}<br>
            <b>Abundance:</b> {row['sum.allrawdata.ABUNDANCE']}<br>
            <b>Biomass:</b> {row['sum.allrawdata.BIOMASS']}<br>
            <b>Plot:</b> {row['PLOT']}<br>
            <b>Location:</b> ({row['LATITUDE']}, {row['LONGITUDE']})
        """
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=row['GENUS_SPECIES']
        ).add_to(marker_cluster)
    
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
                Species Distribution Map {year} ({len(filtered_data)} locations)
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    return m
        
def create_map(data_df):

    year_slider = IntSlider(
        min=int(data_df['YEAR'].min()),
        max=int(data_df['YEAR'].max()),
        step=1,
        description='Year',
        value=int(data_df['YEAR'].min()),
        layout=Layout(width='800px') 
    )
    
    # species filter - ignore rows that have no valid lat/long
    valid_data = data_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    species_list = ['All Species'] + sorted(valid_data['GENUS_SPECIES'].unique().tolist())
    species_dropdown = Select(
        options=species_list,
        description='Species:',
        value='All Species'
    )
    
    return interact(update_map, year=year_slider, species_filter=species_dropdown)


n_qubits = 4
n_layers = 2
scaler = MinMaxScaler()

def prepare_data(data_df):
    # scaling/select relevant features
    features = ['LATITUDE', 'LONGITUDE', 'YEAR','sum.allrawdata.ABUNDANCE']
    
    # clean
    processed_data = data_df[features].dropna()
    
    base_year = processed_data['YEAR'].min()
    processed_data['YEARS_SINCE_START'] = processed_data['YEAR'] - base_year
    
    # normalise
    X = processed_data[['LATITUDE', 'LONGITUDE', 'YEARS_SINCE_START']]
    y = processed_data['sum.allrawdata.ABUNDANCE']
    
    X_scaled = scaler.fit_transform(X)
    y_scaled = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))
    
    return X_scaled, y_scaled


def create_pennylane_circuit():
    # a PennyLane quantum circuit (variational model) for classification
    # see https://pennylane.ai/qml/demos/tutorial_variational_classifier/

    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(inputs, weights):
        # encode inputs
        for i in range(n_qubits):
            qml.RY(inputs[i % len(inputs)], wires=i)
        
        # variational layers
        for layer in range(n_layers):
            # entangling layer
            for i in range(n_qubits):
                qml.RZ(weights[layer, i], wires=i)
                
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # rotation layer
            for i in range(n_qubits):
                qml.RY(weights[layer + n_layers, i], wires=i)
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit


def train_pennylane_model(X_train, y_train, n_epochs=100):

    # train the model
    circuit = create_pennylane_circuit()

    weights = np.random.uniform(
        low=-np.pi, 
        high=np.pi, 
        size=(2 * n_layers, n_qubits)
    )

    # optimiser https://docs.pennylane.ai/en/stable/code/api/pennylane.GradientDescentOptimizer.html
    opt = qml.GradientDescentOptimizer(stepsize=0.01)

    # training loop
    losses = []
    for epoch in range(n_epochs):
        batch_loss = 0
        for X_batch, y_batch in zip(X_train, y_train):
            # forwards
            prediction = circuit(X_batch, weights)
            loss = np.abs(prediction - y_batch[0])
            
            # back
            weights = opt.step(lambda w: circuit(X_batch, w), weights)
            
            batch_loss += loss
            
        avg_loss = batch_loss / len(X_train)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
    return weights, losses
  

def predict_future_abundance(trained_model, location, years_ahead):
        
    future_predictions = []
    base_input = np.array([
        location[0],  # lat
        location[1],  # long
        scaler.transform([[years_ahead]])[0][0]  # scaled future year
    ])
    
    prediction = trained_model.predict([base_input])[0]
    return prediction


X_scaled, y_scaled = prepare_data(filtered_data_df)

# split into test/train subsets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=123)

# train
penny_lane_model = train_pennylane_model(X_train, y_train)

# prediction
location = [47.4, -95.12] # random lat /long - need to plot these probably
years_ahead = 5
prediction = predict_future_abundance(penny_lane_model, location, years_ahead)

print(prediction)
