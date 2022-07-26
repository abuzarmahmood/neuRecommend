def normalize_spikes(spike_array):
    standard_array = spike_array.copy()
    min_vals = np.abs(np.min(spike_array,axis=-1)[:,np.newaxis])
    max_vals = np.max(spike_array,axis=-1)[:,np.newaxis]
    standard_array[spike_array<0] = (standard_array / min_vals)[spike_array<0] 
    standard_array[spike_array>0] = (standard_array / max_vals)[spike_array>0] 
    return standard_array, min_vals, max_vals

def convert_to_8bit(array):
    temp_array = array.copy()
    temp_array *= 127
    temp_array = temp_array.astype('int8')
    return temp_array
