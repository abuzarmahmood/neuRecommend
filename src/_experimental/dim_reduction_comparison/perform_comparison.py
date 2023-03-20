import sys
import os
return_data_path = '/media/bigdata/projects/neuRecommend/src/create_pipeline/return_data.py'
sys.path.append(os.path.dirname(return_data_path))

from return_data import return_data
from compare_methods import dim_red_comparison

data, labels = return_data()

comparison_handler = dim_red_comparison(data, labels)
fit_times, transform_times, variances = \
        comparison_handler.iterate_over_components(2,8, 'pca') 

#fit_time, transform_time, variance = \
#        comparison_handler.run_methods_for_n_components(2, 'pca') 
