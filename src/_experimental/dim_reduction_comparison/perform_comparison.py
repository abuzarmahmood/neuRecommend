import seaborn as sns
import pandas as pd
from itertools import combinations
import numpy as np
from umap.parametric_umap import ParametricUMAP
from sklearn.neural_network import MLPRegressor
from matplotlib.colors import LogNorm
import pylab as plt
from compare_methods import dim_red_comparison
from return_data import return_data
from tqdm import tqdm
import sys
import os


return_data_path = '/media/bigdata/projects/neuRecommend/src/create_pipeline/return_data.py'
sys.path.append(os.path.dirname(return_data_path))

save_path = '/media/bigdata/projects/neuRecommend/src/_experimental/dim_reduction_comparison/out_frame.csv'
save_dir = os.path.dirname(save_path)

data, labels = return_data()

comparison_handler = dim_red_comparison(data, labels)

model_names = ['pca', 'autoencoder', 'umap']

############################################################
# Run loop over chosen models
############################################################

#counter = 0
#fin_out_frame = pd.DataFrame()
# for this_model in tqdm(model_names):
#    min_components, max_components = [2, 8]
#    # fit_times, transform_times, variances = \
#    #        comparison_handler.iterate_over_components(
#    #                min_components, max_components, this_model)
#    for n_components in trange(min_components, max_components):
#        outs = comparison_handler.run_methods_for_n_components(
#            n_components, this_model)
#        outs = np.round(outs, 3)
#        out_frame = pd.DataFrame(
#            dict(
#                fit_times=outs[0],
#                transform_times=outs[1],
#                variances=outs[2],
#                model_name=this_model,
#                n_components=n_components,
#            ),
#            index=[counter],
#        )
#        fin_out_frame = pd.concat([fin_out_frame, out_frame])
#        fin_out_frame.to_csv(save_path)
#        counter += 1
#        #outs.append([fit_times, transform_times, variances])
#    #fin_out_frame = pd.concat([fin_out_frame, out_frame])
#    #fin_out_frame.to_csv(save_path)


# Read in the output frame
fin_out_frame = pd.read_csv(save_path, index_col=0)
mean_out_frame = fin_out_frame.groupby('model_name').mean()

############################################################
# Plot the results
############################################################

# Plot fit times vs number of components
sns.lineplot(
    x='n_components',
    y='fit_times',
    hue='model_name',
    data=fin_out_frame,
    linewidth=2.5,
    markers=True,
    style='model_name',
)
ax = plt.gca()
for this_model in mean_out_frame.index:
    this_mean = mean_out_frame.loc[this_model, 'fit_times']
    ax.axhline(this_mean, color='black', linestyle='--', zorder=-10)
    ax.text(
        fin_out_frame['n_components'].max() - 0.5,
        this_mean*1.1,
        str(np.round(this_mean, 2)),
    )
plt.ylabel('Fit Time (s)')
plt.yscale('log')
fig = plt.gcf()
fig.suptitle('Fit Time vs. Number of Components')
fig.savefig(os.path.join(save_dir, 'fit_time_vs_n_components.png'))
plt.close(fig)
# plt.show()

# Plot transform times vs number of components
sns.lineplot(
    x='n_components',
    y='transform_times',
    hue='model_name',
    data=fin_out_frame,
    linewidth=2.5,
    markers=True,
    style='model_name',
)
ax = plt.gca()
for this_model in mean_out_frame.index:
    this_mean = mean_out_frame.loc[this_model, 'transform_times']
    ax.axhline(this_mean, color='black', linestyle='--', zorder=-10)
    ax.text(
        fin_out_frame['n_components'].max() - 0.5,
        this_mean*1.2,
        str(np.round(this_mean, 2)),
    )
plt.ylabel('Fit Time (s)')
plt.yscale('log')
fig = plt.gcf()
plt.legend(loc='lower left')
fig.suptitle('Transform Time vs. Number of Components')
fig.savefig(os.path.join(save_dir, 'transform_time_vs_n_components.png'))
plt.close(fig)

# Plot variance explained vs number of components
sns.lineplot(
    x='n_components',
    y='variances',
    hue='model_name',
    data=fin_out_frame,
    linewidth=2.5,
    markers=True,
    style='model_name',
)
ax = plt.gca()
plt.ylabel('Variance Explained Ratio')
fig = plt.gcf()
fig.suptitle('Variance Explained vs. Number of Components')
fig.savefig(os.path.join(save_dir, 'variance_vs_n_components.png'))
plt.close(fig)
plt.show()

############################################################
# Comare UMAP and Parametric UMAP
############################################################
n_components = 2

parametric_umap = ParametricUMAP(
    n_components=n_components,
    verbose=True,
)
parametric_umap, p_umap_fit_time = comparison_handler.fit_method(
    parametric_umap)
param_umap_transformed_data, p_umap_transform_time = comparison_handler.transform_method(
    parametric_umap)

umap = comparison_handler.umap(n_components=n_components)
umap, umap_fit_time = comparison_handler.fit_method(umap)
umap_transformed_data, umap_transform_time = comparison_handler.transform_method(
    umap)

# Something is wrong with parametric umap inverse transform
# But embedding looks good
# Regress both embeddings and check match
regressor = MLPRegressor().fit(param_umap_transformed_data, umap_transformed_data)
r2_score = regressor.score(param_umap_transformed_data, umap_transformed_data)
param_umap_transform_custom = regressor.predict(param_umap_transformed_data)

if n_components == 2:
    # 2D version
    # ==============================
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    for num, this_dat in enumerate(dat_list):
        scatter = ax[num, 0].scatter(this_dat[:, 0],
                                     this_dat[:, 1],
                                     c=comparison_handler.y_test)
        im = ax[num, 1].hist2d(this_dat[:, 0],
                               this_dat[:, 1],
                               bins=100,
                               norm=LogNorm(),
                               )
        fig.colorbar(im[3], ax=ax[num, 1])
        ax[num, 0].set_ylabel(name_list[num])
        ax[num, 0].legend(
            handles=scatter.legend_elements()[0],
            labels=['False', 'True'],
        )
    fig.suptitle('UMAP vs Parametric UMAP' + '\n' +
                 'R2 Score: ' + str(np.round(r2_score, 3)))
    fig.savefig(save_dir + '/umap_vs_parametric_umap.png',
                dpi=300,
                #bbox_inches = 'tight',
                )
    plt.show()

else:
    # 3D version
    # ==============================

    name_list = ['umap', 'parametric_umap']
    dat_list = [umap_transformed_data, param_umap_transform_custom]
    # Make subplots of all combinations of first 3 dimensions as 2D scatter plots
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    inds = list(combinations(np.arange(3), 2))
    for num, (this_name, this_dat) in enumerate(zip(name_list, dat_list)):
        ax[0, num].set_title(this_name)
        for num2, (this_ind1, this_ind2) in enumerate(inds):
            print(num2, num)
            scatter = ax[num2, num].scatter(
                this_dat[:, this_ind1],
                this_dat[:, this_ind2],
                c=comparison_handler.y_test)
            ax[num2, num].set_xlabel('dim' + str(this_ind1))
            ax[num2, num].set_ylabel('dim' + str(this_ind2))
            ax[num2, num].legend(
                handles=scatter.legend_elements()[0],
                labels=['False', 'True'],
            )
    fig.suptitle('UMAP vs Parametric UMAP' + '\n' +
                 'r2_score = ' + str(r2_score))
    fig.savefig(os.path.join(save_dir, 'umap_vs_parametric_umap_3D.png'))
    plt.close(fig)
