import numpy as np
import time
import requests
import threading
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.feature_selection import SelectFromModel

#########################################
############ Data properties #############
##########################################
sampling_rate = 20  # Sampling rate in Hz of the input data
window_time = 0.5  # Window size in seconds for each trial window
window_samples = int(window_time * sampling_rate)  # Number of samples in each window

# Cargar datos
data = np.loadtxt("activity_data.txt")
x = data[:, 1:]
y = data[:, 0]

# Eliminar filas con NaNs
mask = ~np.isnan(x).any(axis=1)
x_clean = x[mask]
y_clean = y[mask]

# Entrenar modelo base Random Forest para selección de características
rf_base = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf_base.fit(x_clean, y_clean)

# Selección automática de características
sfm = SelectFromModel(rf_base, prefit=True, threshold='mean')
x_selected = sfm.transform(x_clean)
selected_indices = sfm.get_support(indices=True)

# Definir grid de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Búsqueda de hiperparámetros en todo el dataset usando solo las características seleccionadas
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(x_selected, y_clean)

# Guardar mejor modelo encontrado
best_rf = grid_search.best_estimator_

##########################################
##### Data acquisition configuration #####
##########################################

# Communication parameters
IP_ADDRESS = '10.43.98.116'
COMMAND = 'accX&accY&accZ&acc_time'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data buffer (circular buffer)
max_samp_rate = 5000  # Maximum possible sampling rate
n_signals = 3  # Number of signals (accX, accY, accZ)
buffer_size = max_samp_rate * 5  # Buffer size (number of samples to store)
buffer = np.zeros((buffer_size, n_signals + 1), dtype='float64')  # Buffer for storing data
buffer_index = 0  # Index for the next data point to be written
last_sample_time = 0.0  # Last sample time for the buffer

# Flag for stopping the data acquisition
stop_recording_flag = threading.Event()

# Mutex for thread-safe access to the buffer
buffer_lock = threading.Lock()

# Function for continuously fetching data from the mobile device
def fetch_data():
    sleep_time = 1. / max_samp_rate
    while not stop_recording_flag.is_set():
        try:
            response = requests.get(BASE_URL, timeout=0.5)
            response.raise_for_status()
            data = response.json()

            global buffer, buffer_index, last_sample_time
            with buffer_lock:  # Ensure thread-safe access to the buffer
                buffer[buffer_index, 0] = data["buffer"]["acc_time"]["buffer"][0]
                buffer[buffer_index, 1] = data["buffer"]["accX"]["buffer"][0]
                buffer[buffer_index, 2] = data["buffer"]["accY"]["buffer"][0]
                buffer[buffer_index, 3] = data["buffer"]["accZ"]["buffer"][0]
                buffer_index = (buffer_index + 1) % buffer_size
                last_sample_time = data["buffer"]["acc_time"]["buffer"][0]

        except Exception as e:
            print(f"Error fetching data: {e}")

        time.sleep(sleep_time)

# Function for stopping the data acquisition
def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()

# Start data acquisition
recording_thread = threading.Thread(target=fetch_data, daemon=True)
recording_thread.start()

##########################################
######### Online classification ##########
##########################################

# Mapeo de clases numéricas a actividades
activity_map = {
    1: 'Nada',
    2: 'Sacudirte',
    3: 'Saltar de arriba a abajo',
    4: 'Sentadillas',
    5: 'Saltar de lado a lado',
    6: 'Girar'
}

update_time = 0.25
ref_time = time.time()

def extract_features(window_data, fs):
    feat = []
    rms_total = 0.0

    n_signals = window_data.shape[1]

    for s in range(n_signals):
        sig = window_data[:, s]

        # Estadísticas temporales
        feat.append(np.mean(sig))
        feat.append(np.std(sig))
        feat.append(stats.kurtosis(sig))
        feat.append(stats.skew(sig))
        feat.append(np.min(sig))
        feat.append(np.max(sig))
        feat.append(np.median(sig))
        feat.append(np.ptp(sig))
        feat.append(np.percentile(sig, 25))
        feat.append(np.percentile(sig, 75))
        feat.append(np.sum(sig**2))  # energía eje individual

        # Estadísticas espectrales
        fft_vals = np.abs(np.fft.rfft(sig))
        fft_freqs = np.fft.rfftfreq(len(sig), d=1/fs)

        feat.append(np.mean(fft_vals))
        feat.append(np.std(fft_vals))
        feat.append(fft_freqs[np.argmax(fft_vals)])  # frecuencia dominante

        rms_total += np.sum(sig**2)

    # Magnitud total del vector aceleración
    acc_total = np.linalg.norm(window_data, axis=1)
    feat.append(np.mean(acc_total))
    feat.append(np.std(acc_total))
    feat.append(np.max(acc_total))
    feat.append(np.sum(acc_total**2))  # energía total

    rms_total = np.sqrt(rms_total)
    feat.append(rms_total)

    return np.array(feat).reshape(1, -1)

while True:
    time.sleep(update_time)

    if buffer_index > 2 * sampling_rate:
        ref_time = time.time()

        ##### Get last data samples #####
        end_index = (buffer_index - 1) % buffer_size
        start_index = (buffer_index - 2) % buffer_size

        with buffer_lock:
            while (buffer[end_index, 0] - buffer[start_index, 0]) <= window_time:
                start_index = (start_index - 1) % buffer_size
            indices = (buffer_index - np.arange(buffer_size, 0, -1)) % buffer_size
            last_raw_data = buffer[indices, :]  # Get last data samples from the buffer

        # Calculate time vector for interpolation
        t = last_raw_data[:, 0]  # Time vector from the buffer
        t_uniform = np.linspace(last_sample_time - window_time, last_sample_time, int(window_time * sampling_rate))

        # Interpolate each signal to a uniform time vector
        last_data = np.zeros((len(t_uniform), n_signals))  # Array with interpolated data
        for i in range(n_signals):
            interp_x = interp1d(t, last_raw_data[:, i + 1], kind='linear', fill_value="extrapolate")
            last_data[:, i] = interp_x(t_uniform)  # Interpolate signal i to the uniform time vector

       #print("Window data:\n", last_data)

        #######################################################
        ##### Calculate features of the last data samples #####
        #######################################################
        feature_vector = extract_features(last_data, sampling_rate)
        feature_vector_selected = feature_vector[:, selected_indices]
        
        
        #################################################################
        ##### Evaluate classifier here with the calculated features #####
        #################################################################
        predicted_class = best_rf.predict(feature_vector_selected)[0]
        activity_name = activity_map.get(predicted_class, "Desconocida")
        print(f"Actividad: {activity_name}")

# Stop data acquisition (this would only be reached with an exception or external stop)
stop_recording()



