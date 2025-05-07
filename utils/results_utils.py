import json
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns  # Optional, for smoother KDE plot
from matplotlib.ticker import MultipleLocator


MODEL_COLORS={'pretrained':'tab:blue','finetuned':'tab:orange','geo':'tab:green','sift':'tab:red', 'geo_ctx_v2':'tab:pink',
              'pretrained_gam':'tab:blue','finetuned_gam':'tab:orange','geo_gam':'tab:green','sift_gam':'tab:red', 'geo_ctx_v2_gam':'tab:pink'}
MODEL_LABELS={
    "pretrained": "Pre-trained LoFTR",
    "finetuned": "Fine-tuned LoFTR",
    "geo": "Geo-LoFTR",
    "sift":"SIFT",
    "geo_ctx_v2":"Geo-LoFTR, low res.",
    ######################### + GAM: 
    "pretrained_gam": "Pre-trained LoFTR + GAM",
    "finetuned_gam": "Fine-tuned LoFTR + GAM",
    "geo_gam": "Geo-LoFTR + GAM",
    "sift_gam":"SIFT + GAM",
    "geo_ctx_v2_gam":"Geo-LoFTR, low res. + GAM"
}
MODEL_LINESTYLE={
    "pretrained": "-",
    "finetuned": "-",
    "geo": "-",
    "sift":"-",
    "geo_ctx_v2":"-",
    ######################### + GAM: 
    "pretrained_gam": "--",
    "finetuned_gam": "--",
    "geo_gam": "--",
    "sift_gam":"--",
    "geo_ctx_v2_gam":"--"
}
#### DATA LOADING #########################

def find_result_npz(path):
    for file in os.listdir(path):
        if "_results_" in file and file.endswith(".npz"):
            return file
    return None

def find_per_query_npz(path):
    for file in os.listdir(path):
        if "_per_query_" in file and file.endswith(".npz"):
            return file
    return None

def find_json(path):
    for file in os.listdir(path):
        if file.endswith(".json"):
            return file
    return None

def load_data(base_path, model_type, el, az, obs_el=40, obs_az=180, pose_prior=False, uncertainty="low"):
    print(f"base_pat: {base_path}")
    sun_comb_name = f"map_{el}_{az}_obsv_{obs_el}_{obs_az}"
    if pose_prior:
        pose_prior_str = f"pose_uncertainty_{uncertainty}"
    else:
        pose_prior_str = "wo_pose_prior"

    if "sift" in model_type:
        method_str = f"sift_gam" if "gam" in model_type else "sift"
    else :
        method_str = f"loftr/{model_type}"
   
    target_path = os.path.join(base_path,
                    os.path.join(sun_comb_name,
                            os.path.join(f"{pose_prior_str}/{method_str}", "accuracy") ) )
    print(f"target_path: {target_path}")
    filename = find_result_npz(target_path)
    # print(f"filename: {filename}")
    
    data = np.load(os.path.join(target_path, filename), allow_pickle=True)
    return data

def load_data_per_query(base_path, model_type, map_el, map_az, obs_el=40, obs_az=180, pose_prior=False, uncertainty="low"):
    
    sun_comb_name = f"map_{map_el}_{map_az}_obsv_{obs_el}_{obs_az}"
    if pose_prior:
        pose_prior_str = f"pose_uncertainty_{uncertainty}"
    else:
        pose_prior_str = "wo_pose_prior"

    if "sift" in model_type:
        method_str = f"sift_gam" if "gam" in model_type else "sift"
    else:
        method_str = f"loftr/{model_type}"  
   
    target_path = os.path.join(base_path,
                    os.path.join(sun_comb_name,
                            os.path.join(f"{pose_prior_str}/{method_str}", "accuracy") ) )
    filename = find_per_query_npz(target_path)
    data = np.load(os.path.join(target_path, filename), allow_pickle=True)['per_query_res']
    return data

def load_json_data(base_path, model_type, map_el, map_az, obs_el=40, obs_az=180, pose_prior=False, uncertainty="low"):
    
    sun_comb_name = f"map_{map_el}_{map_az}_obsv_{obs_el}_{obs_az}"
    if pose_prior:
        pose_prior_str = f"pose_uncertainty_{uncertainty}"
    else:
        pose_prior_str = "wo_pose_prior"
    
    if model_type == "sift":
        method_str = f"sift"
    else:
        method_str = f"loftr/{model_type}"    
    
    target_path = os.path.join(base_path,
                    os.path.join(sun_comb_name,
                            os.path.join(f"{pose_prior_str}/{method_str}", "accuracy") ) )
    filename = find_json(target_path)
    with open(os.path.join(target_path, filename), 'r') as f:
        data = json.load(f)
    return data


########## PLOTTING ########################

# Localization accuracy
def plot_loc_cum_acc(base_path, model_types,  map_el=40, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=1,pose_prior=False, uncertainty="low", filedest=None, show_2m_precision=True, show_title=False):
    
    """
    Plots the cumulative localization accuracy for different image matching models over specified altitude ranges.
    Parameters:
    -----------
    base_path (str): The base directory path where the data is stored.
    model_types (list of str): A list of strings representing the types of models to be evaluated 
                                (e.g., 'pretrained', 'finetuned', 'geo', 'sift').
    map_el (int, optional): Sun Elevation of the map in degrees. 
    map_az (int, optional): Sun Azimuth of the map in degrees. 
    obs_el (int, optional): Sun Elevation of the observation in degrees.
    obs_az (int, optional): Sun Azimuth of the observation in degrees.
    n_alt_ranges (int, optional): Number of altitude ranges to divide the data into.
    pose_prior (bool, optional): Whether to use pose prior information.
    uncertainty (str, optional): The level of uncertainty to consider if pose_prior is enabled (e.g., "low", "medium", "high").
    filedest (str or None, optional): Destination directory to save the plots. If None, the plots are displayed instead of being saved.
    show_title (bool, optional): Whether to display the title on the plots. 
    
    Returns:
    --------
    None
        The function either saves the plots to the specified directory or displays them.
    
    Notes:
    ------
    - The function computes cumulative location accuracy for each model type and altitude range.
    - The cumulative accuracy is calculated as the percentage of location errors below a threshold.
    - The function supports splitting the data into multiple altitude ranges and plotting results for each range.
    - The plots include labels with additional information about accuracy at 1m and 2m thresholds.
    - The function uses predefined model labels and colors (`MODEL_LABELS` and `MODEL_COLORS`) for visualization.
    """
    

    for i in range(n_alt_ranges):
        fig, axs = plt.subplots()
        if filedest:
            filepath = f"{filedest}/loc_cum_acc_alt_subrange_{i}.png"
        else:
            filepath = None
        for model_type in model_types:
            
            data = load_data(base_path, model_type, map_el, map_az, obs_el, obs_az, pose_prior=pose_prior, uncertainty=uncertainty)
            
            loc_error_list = data['location_err_acc']
            altitude_list = data['altitude_acc']

            # Convert lists to numpy arrays for easier manipulation
            loc_error = np.array(loc_error_list)
            altitude = np.array(altitude_list)
            
            # Sort the arrays by altitude
            sorted_indices = np.argsort(altitude)
            altitude = altitude[sorted_indices]
            loc_error = loc_error[sorted_indices]
            
            # Split data into three equal parts
            altitude_splits = np.array_split(altitude, n_alt_ranges)
            loc_error_splits = np.array_split(loc_error, n_alt_ranges)

            bins = np.arange(0.1, 10, 0.1)
            # Define quantities in each altitude range
            loc_error_range=loc_error_splits[i]               
            altitude_range=altitude_splits[i]
        
            # Compute cumulative distribution
            loc_err_cum = [100*np.sum(loc_error_range < Th) / len(loc_error_range) for Th in bins]
            corr_loc = 100*np.sum(loc_error_range < 1) / len(loc_error_range)
            # fail_loc = 100*np.sum(loc_error_range > 10) / len(loc_error_range)
            
            # Create label with additional info     
            label = f"{MODEL_LABELS[model_type]}\n({'{:.1f}'.format(corr_loc)}% @1m)"
            if show_2m_precision:
                corr_loc_2m = 100*np.sum(loc_error_range < 2) / len(loc_error_range)
                label += f", {'{:.1f}'.format(corr_loc_2m)}% @2m)"
            color = MODEL_COLORS[model_type]
            linestyle = MODEL_LINESTYLE[model_type]
            # Cumulative pose accuracy vs bins
            axs.plot(bins, loc_err_cum, label=label, color=color, linestyle=linestyle, linewidth=4)
            axs.set_ylim(0, 101)
            axs.set_xlim(0, 10)
            # axs[row, col].legend(fontsize=10)

        # Make the legend, labels, and ticks larger
        axs.set_ylabel('Cumulative Accuracy [%]', fontsize=36)
        axs.set_xlabel('Localization error [m]', fontsize=36)  
        axs.legend(fontsize=28)
        # axs.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.35, 0.4))
        axs.tick_params(axis='both', which='major', labelsize=32)
        axs.yaxis.set_major_locator(MultipleLocator(10))
        fig.set_size_inches(10, 10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        axs.grid(True)
        if show_title:
            axs.set_title(f"Altitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=24)  

        if filepath:
            fig.savefig(filepath)


def plot_loc_cum_acc_alt_range(base_path, model_types, min_alt=64, max_alt=200, map_el=40, map_az=180, obs_el=40, obs_az=180, pose_prior=False, uncertainty="low", title=None, filepath=None):
    """
    Plots the cumulative localization accuracy for different image matching models and a specified altitude range
    Args:
        base_path (str): The base directory path where the data is stored.
        model_types (list of str): List of model types to include in the plot.
        min_alt (float, optional): Minimum altitude for filtering data. Defaults to 64.
        max_alt (float, optional): Maximum altitude for filtering data. Defaults to 200.
        map_el (float, optional): Sun Elevation angle of the map. Defaults to 40.
        map_az (float, optional): Sun Azimuth angle of the map. Defaults to 180.
        obs_el (float, optional): Sun Elevation angle of the observer. Defaults to 40.
        obs_az (float, optional): Sun Azimuth angle of the observer. Defaults to 180.
        pose_prior (bool, optional): Whether to use pose prior in the data. Defaults to False.
        uncertainty (str, optional): Level of uncertainty in the data ("low", "medium", "high"). Defaults to "low".
        title (str, optional): Title of the plot. If None, a default title is generated based on the altitude range. Defaults to None.
        filepath (str, optional): Filepath to save the plot. If None, the plot is not saved. Defaults to None.
    Returns:
        None: The function generates and optionally saves a plot but does not return any value.
    Notes:
        - The function filters data based on the specified altitude range (`min_alt` to `max_alt`).
        - The cumulative accuracy is computed for location errors in the range of 0.1m to 10m.
        - The plot includes a legend with accuracy at 1m and 2m thresholds for each model.
        - The function uses predefined `MODEL_LABELS` and `MODEL_COLORS` for labeling and coloring the models.
    """
    
    fig, axs = plt.subplots()
    for model_type in model_types:
        data = load_data(base_path, model_type, map_el, map_az, obs_el, obs_az, pose_prior=pose_prior, uncertainty=uncertainty)

        loc_error = data['location_err_acc']
        altitude = data['altitude_acc']

        loc_err_range = []
        altitude_range = []
        for i in range(len(altitude)):
            if altitude[i] >= min_alt and altitude[i] <= max_alt:
                loc_err_range.append(loc_error[i])
                altitude_range.append(altitude[i])
            
        loc_err_range = np.array(loc_err_range)
        altitude_range = np.array(altitude_range)
        

        bins = np.arange(0.1, 10, 0.1)
        # Define quantities in each altitude range 
        # Compute cumulative distribution
        loc_err_cum = [100*np.sum(loc_err_range < Th) / len(loc_err_range) for Th in bins]
        corr_loc = 100*np.sum(loc_err_range < 1) / len(loc_err_range)
        corr_loc_2m = 100*np.sum(loc_err_range < 2) / len(loc_err_range)
        fail_loc = 100*np.sum(loc_err_range > 10000) / len(loc_err_range)
        
        # Create label with additional info     
        # umulative pose accuracy vs bins
        label = f"{MODEL_LABELS[model_type]}\n(@1m: {'{:.1f}'.format(corr_loc)}%, @2m: {'{:.1f}'.format(corr_loc_2m)}%)"
        color = MODEL_COLORS[model_type]
        linestyle = MODEL_LINESTYLE[model_type]
        axs.plot(bins, loc_err_cum, label=label, linestyle = linestyle, color=color, linewidth=4)
        axs.set_ylim(0, 101)
        axs.set_xlim(0, 10)

        # Make the legend, labels, and ticks larger     
        axs.set_ylabel('Cumulative Accuracy [%]', fontsize=36)
        axs.set_xlabel('Localization error [m]', fontsize=36)
        # axs.legend(fontsize=28, loc='upper left', bbox_to_anchor=(0.35, 0.4))
        axs.legend(fontsize=26)
        axs.tick_params(axis='both', which='major', labelsize=32)
        axs.grid(True)
        axs.yaxis.set_major_locator(MultipleLocator(10))
        fig.set_size_inches(10, 10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if title is None:
            title = f"altitude range: {min_alt}-{max_alt} m"
        axs.set_title(f"{title}", fontsize=18) 
           
    # fig.suptitle(f'Cumulative Pose Accuracy \nAZ diff: {fixed_az - obs_az} deg, EL diff: {fixed_el - obs_el} deg \nAltitude range: 64-200m', fontsize=22)
    if filepath:
        fig.savefig(filepath)

def plot_loc_err_density_vs_alt(base_path, model_type, map_el=40, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=1, show_inliers=False, filepath=None):
    """
    Plots the density of localization error as a function of altitude for a given dataset.
    This function visualizes the relationship between altitude and localization error
    using a scatter plot with density-based coloring. The altitude range can be split
    into multiple sub-ranges, and the localization error is plotted for each range.
    Args:
        base_path (str): Path to the base directory containing the data.
        model_type (str): Type of the model used for generating the data.
        map_el (int, optional): Sun Elevation angle of the map in degrees. Defaults to 40.
        map_az (int, optional): Sun Azimuth angle of the map in degrees. Defaults to 180.
        obs_el (int, optional): Sun Elevation angle of the observations in degrees. Defaults to 40.
        obs_az (int, optional): Sun Azimuth angle of the observations in degrees. Defaults to 180.
        n_alt_ranges (int, optional): Number of altitude ranges to split the data into. Defaults to 1.
        show_inliers (bool, optional): Whether to include RANSAC inliers in the plot. Defaults to False.
        filepath (str, optional): Filepath to save the plot. If None, the plot is displayed. Defaults to None.
    Returns:
        None: The function either displays the plot or saves it to the specified filepath.
    Notes:
        - The function uses Gaussian Kernel Density Estimation (KDE) to compute the density
          of points in the scatter plot.
        - If `n_alt_ranges` is greater than 1, the altitude range is divided into equal parts,
          and a separate subplot is created for each range.
    """
    
    fig, axs = plt.subplots(n_alt_ranges, 1, figsize=(12,12))
    for row in range(n_alt_ranges):        
        data = load_data_per_query(base_path, model_type, map_el, map_az, obs_el, obs_az)
        loc_error_list = []
        inliers_list = []
        altitude_list = []
        # nr_pnp_fails = data[-1]['Nr of PNP Fails']
        for query_id in range(len(data)-1):
            altitude_list.append(data[query_id]['altitude'])
            if "Location error" in data[query_id].keys():
                loc_error_list.append(data[query_id]["Location error"])
                inliers_list.append(data[query_id]["RANSAC Inliers"])
            else:
                loc_error_list.append(1000000)

        # Convert lists to numpy arrays for easier manipulation
        loc_error = np.array(loc_error_list)
        altitude = np.array(altitude_list)

        # Sort the arrays by altitude
        sorted_indices = np.argsort(altitude)
        altitude = altitude[sorted_indices]
        loc_error  = loc_error[sorted_indices]
        

        # Split data into three equal parts (low, medium, high altitude ranges)
        altitude_splits = np.array_split(altitude, n_alt_ranges)
        loc_error_splits = np.array_split(loc_error, n_alt_ranges)

        # Define quantities in each altitude range
        loc_error_range=loc_error_splits[row]
        altitude_range=altitude_splits[row]

        # Convert them to list
        loc_errorc_range = loc_error_range.tolist()
        altitude_range = altitude_range.tolist() 

        
        xy = np.vstack([altitude_range, loc_error_range])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = np.array(altitude_range)[idx], np.array(loc_error_range)[idx], z[idx]
        if n_alt_ranges == 1:
            axs.scatter(x, y, c=z, s=50, edgecolor='face')
            axs.grid(axis='x', alpha=0.75)
            axs.grid(axis='y', alpha=0.75)
            axs.set_xlabel('Altitude [m]', fontsize=22)
            axs.set_ylabel('Localization error [m]', fontsize=22)
            # axs.set_title('Correct matches vs altitude')
            axs.set_ylim(0, 10)
            axs.tick_params(axis='both', which='major', labelsize=22)
        else:
            axs[row].scatter(x, y, c=z, s=50, edgecolor='face')
            axs[row].grid(axis='x', alpha=0.75)
            axs[row].grid(axis='y', alpha=0.75)
            axs[row].set_ylabel('Localization error [m]', fontsize=22)
            # axs[row].set_title('Correct matches vs altitude')
            axs[row].set_ylim(0, 10)
            axs[row].tick_params(axis='both', which='major', labelsize=22)
            if row == n_alt_ranges - 1:
                axs[row].set_xlabel('Altitude [m]', fontsize=22)
            

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if filepath:
            fig.savefig(filepath)


#   Matching accuracy
def plot_match_cum_acc(base_path, model_types,  map_el=40, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=1,pose_prior=False, uncertainty="low", filedest=None, show_title=False):
    
    """
    Plots the cumulative matching accuracy for different image matching models over specified altitude ranges.
    Parameters:
    -----------
    base_path (str): The base directory path where the data is stored.
    model_types (list of str): A list of strings representing the types of models to be evaluated 
                                (e.g., 'pretrained', 'finetuned', 'geo', 'sift').
    map_el (int, optional): Sun Elevation of the map in degrees. 
    map_az (int, optional): Sun Azimuth of the map in degrees. 
    obs_el (int, optional): Sun Elevation of the observation in degrees.
    obs_az (int, optional): Sun Azimuth of the observation in degrees.
    n_alt_ranges (int, optional): Number of altitude ranges to divide the data into.
    pose_prior (bool, optional): Whether to use pose prior information.
    uncertainty (str, optional): The level of uncertainty to consider if pose_prior is enabled (e.g., "low", "medium", "high").
    filedest (str or None, optional): Destination directory to save the plots. If None, the plots are displayed instead of being saved.
    show_title (bool, optional): Whether to display the title on the plots. 
    
    Returns:
    --------
    None
        The function either saves the plots to the specified directory or displays them.
    
    Notes:
    ------
    - The function computes cumulative matches reprojection accuracy for each model type and altitude range.
    - The cumulative accuracy is calculated as the percentage of reprojection errors below a threshold.
    - The function supports splitting the data into multiple altitude ranges and plotting results for each range.
    - The plots include labels with additional information about accuracy at 1m and 2m thresholds.
    - The function uses predefined model labels and colors (`MODEL_LABELS` and `MODEL_COLORS`) for visualization.
    """
    

    for i in range(n_alt_ranges):
        fig, axs = plt.subplots()
        if filedest:
            filepath = f"{filedest}/loc_cum_acc_alt_subrange_{i}.png"
        else:
            filepath = None
        for model_type in model_types:
            
            data = load_data(base_path, model_type, map_el, map_az, obs_el, obs_az, pose_prior=pose_prior, uncertainty=uncertainty)
            
            rep_error_list = data['err_1to0_list']
            altitude_list = data['altitude_acc']

            # Convert lists to numpy arrays for easier manipulation
            rep_error = np.array(rep_error_list)
            altitude = np.array(altitude_list)
            
            # Sort the arrays by altitude
            sorted_indices = np.argsort(altitude)
            altitude = altitude[sorted_indices]
            rep_error = rep_error[sorted_indices]
            
            # Split data into three equal parts
            altitude_splits = np.array_split(altitude, n_alt_ranges)
            rep_error_splits = np.array_split(rep_error, n_alt_ranges)

            bins = np.arange(0.1, 10, 0.1)
            # Define quantities in each altitude range
            rep_error_range=rep_error_splits[i]               
            altitude_range=altitude_splits[i]
        
            # Compute cumulative distribution
            rep_err_cum = [100*np.sum(rep_error_range < Th) / len(rep_error_range) for Th in bins]
            corr_match = 100*np.sum(rep_error_range < 1) / len(rep_error_range)
            corr_match_2px = 100*np.sum(rep_error_range < 2) / len(rep_error_range)
            # fail_loc = 100*np.sum(loc_error_range > 10) / len(loc_error_range)
            
            # Create label with additional info     
            label = f"{MODEL_LABELS[model_type]}\n(@1px: {'{:.1f}'.format(corr_match)}%, @2px: {'{:.1f}'.format(corr_match_2px)}%)"
            color = MODEL_COLORS[model_type]
            linestyle = MODEL_LINESTYLE[model_type]
            # Cumulative pose accuracy vs bins
            axs.plot(bins, rep_err_cum, label=label, color=color, linestyle=linestyle, linewidth=4)
            axs.set_ylim(0, 101)
            axs.set_xlim(0, 5)
            # axs[row, col].legend(fontsize=10)

        # Make the legend, labels, and ticks larger
        axs.set_ylabel('Cumulative Accuracy [%]', fontsize=36)
        axs.set_xlabel('Reprojection error [px]', fontsize=36)  
        axs.legend(fontsize=28)
        # axs.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.35, 0.4))
        axs.tick_params(axis='both', which='major', labelsize=32)
        axs.yaxis.set_major_locator(MultipleLocator(10))
        fig.set_size_inches(10, 10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        axs.grid(True)
        if show_title:
            axs.set_title(f"Altitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=24)  

        if filepath:
            fig.savefig(filepath)

def plot_rep_err_density_vs_alt(base_path, model_type, map_el=40, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=1, show_title=True, show_inliers=False, filepath=None):
    # COMPUTATIONALLY EXPENSIVE 
    fig, axs = plt.subplots(n_alt_ranges, 1, figsize=(12,12))
    for row in range(n_alt_ranges):        
        data = load_data_per_query(base_path, model_type, map_el, map_az, obs_el, obs_az)
        err_list = []
        altitude_list = []
        # nr_pnp_fails = data[-1]['Nr of PNP Fails']
        for query_id in range(len(data)-1):
            
            if "Location error" in data[query_id].keys():
                if show_inliers:
                    err_1to0_per_query = data[query_id]["err_1to0_inliers"]
                else:
                    err_1to0_per_query = data[query_id]["err_1to0"]
                err_list.extend(err_1to0_per_query)
                altitude_list.extend(np.full_like(err_1to0_per_query, data[query_id]['altitude']))

        # Convert lists to numpy arrays for easier manipulation
        err = np.array(err_list)
        altitude = np.array(altitude_list)

        # Sort the arrays by altitude
        sorted_indices = np.argsort(altitude)
        altitude = altitude[sorted_indices]
        err  = err[sorted_indices]
        

        # Split data into three equal parts (low, medium, high altitude ranges)
        altitude_splits = np.array_split(altitude, n_alt_ranges)
        err_splits = np.array_split(err, n_alt_ranges)

        # Define quantities in each altitude range
        err_range=err_splits[row]
        altitude_range=altitude_splits[row]

        # Convert them to list
        err_range = err_range.tolist()
        altitude_range = altitude_range.tolist() 

        
        xy = np.vstack([altitude_range, err_range])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = np.array(altitude_range)[idx], np.array(err_range)[idx], z[idx]
        if n_alt_ranges == 1:
            axs.scatter(x, y, c=z, s=50, edgecolor='face')
            axs.grid(axis='x', alpha=0.75)
            axs.grid(axis='y', alpha=0.75)
            axs.set_xlabel('Altitude [m]', fontsize=22)
            axs.set_ylabel('Matches Reprojection Error [px]', fontsize=22)
            axs.tick_params(axis='both', which='major', labelsize=22)
            axs.set_ylim(0, 10)
            if show_title:
                axs.set_title(MODEL_LABELS[model_type])
        else:
            axs[row].scatter(x, y, c=z, s=50, edgecolor='face')
            axs[row].grid(axis='x', alpha=0.75)
            axs[row].grid(axis='y', alpha=0.75)
            axs[row].set_ylabel('Matches Reprojection Error [px]', fontsize=22)
            axs[row].tick_params(axis='both', which='major', labelsize=22)
            # axs[row].set_title('Correct matches vs altitude')
            axs[row].set_ylim(0, 10)
            if row == n_alt_ranges - 1:
                axs[row].set_xlabel('Altitude [m]', fontsize=22)
            if show_title:
                axs[row].set_title(MODEL_LABELS[model_type])

        # Create a secondary y-axis for altitude distribution
        ax_altitude = axs.twinx() if n_alt_ranges == 1 else axs[row].twinx
        
        # Plot altitude distribution as a density or histogram
        # Calculate histogram for altitude_range
        # alt_hist, alt_bins = np.histogram(altitude_range, bins=50, density=True)  # Adjust the number of bins as needed
        # alt_hist = alt_hist / np.max(alt_hist) * 100  # Scale to percentage
        # Generate KDE for altitude data and scale it to 100
        kde = gaussian_kde(altitude_range)
        kde_values = kde(altitude_range)
        kde_values = kde_values * (100 / np.sum(kde_values))  # Scale to area 100

        # sns.kdeplot(altitude_range, ax=ax_altitude, color="black", fill=True, alpha=0.1, linewidth=2, hue_norm=(0,100))
        # ax_altitude.plot(alt_bins[:-1], alt_hist, color="black", linestyle="--", linewidth=2)
        ax_altitude.plot(altitude_range, kde_values, color="black", linewidth=2)
        ax_altitude.set_ylabel("Altitude Density [%]", color="black", fontsize=14)
        ax_altitude.tick_params(axis="y", labelcolor="black",  labelsize=22)
        ax_altitude.set_ylim(0, 1)  # Adjust scale if needed

    fig.set_size_inches(10, 10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if filepath:
        fig.savefig(filepath)

def plot_1px_rep_err_density_vs_alt(base_path, model_type, map_el=40, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=1, show_title=True, show_suptitle=True, show_inliers=False, filepath=None):
       
    fig, axs = plt.subplots(n_alt_ranges, 1, figsize=(12,12))
    for row in range(n_alt_ranges):        
        data = load_data_per_query(base_path, model_type, map_el, map_az, obs_el, obs_az)
        corr_match_acc_list = []
        altitude_list = []
        # nr_pnp_fails = data[-1]['Nr of PNP Fails']
        for query_id in range(len(data)-1):
            
            if "Location error" in data[query_id].keys():
                if show_inliers:
                    err_1to0_per_query = data[query_id]["err_1to0_inliers"]
                else:
                    err_1to0_per_query = data[query_id]["err_1to0"]
                # inliers_list.append(data[query_id]["RANSAC Inliers"])
                
                corr_match_acc = 100*np.sum(err_1to0_per_query < 1) / len(err_1to0_per_query)
                corr_match_acc_list.append(corr_match_acc)
                altitude_list.append(data[query_id]['altitude'])

        # Convert lists to numpy arrays for easier manipulation
        corr_match_acc = np.array(corr_match_acc_list)
        altitude = np.array(altitude_list)

        # Sort the arrays by altitude
        sorted_indices = np.argsort(altitude)
        altitude = altitude[sorted_indices]
        corr_match_acc  = corr_match_acc[sorted_indices]
        

        # Split data into three equal parts (low, medium, high altitude ranges)
        altitude_splits = np.array_split(altitude, n_alt_ranges)
        corr_match_acc_splits = np.array_split(corr_match_acc, n_alt_ranges)

        # Define quantities in each altitude range
        corr_match_acc_range=corr_match_acc_splits[row]
        altitude_range=altitude_splits[row]

        # Convert them to list
        corr_match_acc_range = corr_match_acc_range.tolist()
        altitude_range = altitude_range.tolist() 

        
        xy = np.vstack([altitude_range, corr_match_acc_range])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = np.array(altitude_range)[idx], np.array(corr_match_acc_range)[idx], z[idx]
        if n_alt_ranges == 1:
            axs.scatter(x, y, c=z, s=50, edgecolor='face')
            axs.grid(axis='x', alpha=0.75)
            axs.grid(axis='y', alpha=0.75)
            axs.set_xlabel('Altitude [m]')
            axs.set_ylabel('Correct matches (@1px) [%]')
            axs.set_ylim(0, 100)
            if show_title:
                axs.set_title(f"altitude range: {np.min(altitude_range)}-{np.max(altitude_range)} m")
        else:
            axs[row].scatter(x, y, c=z, s=50, edgecolor='face')
            axs[row].grid(axis='x', alpha=0.75)
            axs[row].grid(axis='y', alpha=0.75)
            axs[row].set_ylabel('Correct matches (@1px) [%]')
            axs[row].set_ylim(0, 100)
            if row == n_alt_ranges - 1:
                axs[row].set_xlabel('Altitude [m]')
            if show_title:
                axs[row].set_title(f"altitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m")
    if show_suptitle:
        fig.suptitle(MODEL_LABELS[model_type])


# Varying azimuth 
def plot_loc_acc_vs_az(base_path, model_types, precision=1, map_el=40, obs_el=40, obs_az=180, n_alt_ranges=1, filepath=None):
    
    fig, axs = plt.subplots(1, n_alt_ranges)

    azimuths = [0, 45, 90, 135, 180, 180, 225, 270, 315]
    azimuths4plot = np.array([0, 45, 90, 135, 180, -180, -135, -90, -45])
    ref_az_plot = obs_az if obs_az < 180 else obs_az - 360



    print(f"\nPrecision level: {precision} m")
    for row in range(n_alt_ranges):
        print(f"\nAltitude range: {row}")
        for model_type in model_types:
            corr_loc_list = []
            fail_loc_list = []

            for _, az in enumerate(azimuths):
                data = load_data(base_path, model_type, map_el, az, obs_el, obs_az)
                loc_error_list = data['location_err_acc']
                altitude_list = data['altitude_acc']

                # Convert lists to numpy arrays for easier manipulation
                loc_error = np.array(loc_error_list)
                altitude = np.array(altitude_list)      

                # Sort the arrays by altitude
                sorted_indices = np.argsort(altitude)
                altitude = altitude[sorted_indices]
                loc_error = loc_error[sorted_indices]             

                # Split data into three equal parts (low, medium, high altitude ranges)
                altitude_splits = np.array_split(altitude, n_alt_ranges)
                loc_error_splits = np.array_split(loc_error, n_alt_ranges)
                
                # Define quantities in each altitude range
                loc_error_range=loc_error_splits[row]               
                altitude_range=altitude_splits[row]

                # Compute cumulative distribution
                corr_loc = 100*float(np.sum(np.array(loc_error_range )< precision))/len(loc_error_range)
                fail_loc = 100*np.sum(loc_error_range > 10) / len(loc_error_range)
                # avg_inliers = int(np.sum(inliers_range)/len(inliers_range))
                
                corr_loc_list.append(corr_loc)
                fail_loc_list.append(fail_loc)

            


            # Plot loc err CDF
            label = MODEL_LABELS[model_type]
            color = MODEL_COLORS[model_type]
            linestyle = MODEL_LINESTYLE[model_type]
            sorted_indices = np.argsort(azimuths4plot)
            sorted_azimuth = azimuths4plot[sorted_indices]
            corr_loc4plot = np.array(corr_loc_list)
            fail_loc4plot = np.array(fail_loc_list)
            sorted_corr_loc = corr_loc4plot[sorted_indices]
            sorted_fail_loc = fail_loc4plot[sorted_indices]

            
            # print(f"Model: {model_type}, Accuracy [%]: {[f'{val:.1f}' for val in sorted_corr_loc]}")
            print(f"Model: {model_type}, Fail [%]: {[f'{val:.1f}' for val in sorted_fail_loc]}")
            if n_alt_ranges == 1:
                
                axs.plot(sorted_azimuth, sorted_corr_loc, label=label, linestyle=linestyle, marker='o', linewidth=4, color=color)
                axs.axvline(x=ref_az_plot, color='black', linewidth=4, linestyle='--')
                axs.legend(fontsize=20)
                axs.set_ylabel("Pose accuracy [%]", fontsize=20)
                axs.set_xlabel(r'Sun Azimuth[$^{\circ}$]', fontsize=20)
                axs.tick_params(axis='both', which='major', labelsize=20)
                axs.set_ylim(0, 101)
                axs.set_xlim(-180, 181)
                axs.set_xticks(np.arange(-180, 181, 45))
                axs.yaxis.set_major_locator(MultipleLocator(10))   # Major ticks every 10
                fig.set_size_inches(12, 12)
                axs.grid(True)
                # axs.set_title(f"Altitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=20)    
                
            else:
                axs[row].plot(sorted_azimuth, sorted_corr_loc, label=label, linewidth=4, linestyle=linestyle, marker='o', color=color)
                axs[row].axvline(x=ref_az_plot, color='black', linewidth=4, linestyle='--')
                if row == n_alt_ranges -1 :
                    axs[row].set_title(f"{int(np.min(altitude_range))}-{200} m", fontsize=40)
                else: 
                    axs[row].set_title(f"{int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=40)
                # axs[row].legend(fontsize=28) if row == 2 else None
                axs[row].legend(fontsize=28, loc='upper left', bbox_to_anchor=(0.01, 0.95)) if row == 2 else None    
                axs[row].set_ylabel("Localization accuracy [%]", fontsize=36) if row==0 else None                
                axs[row].set_ylim(0, 101)
                axs[row].set_xlim(-180, 181)
                axs[row].set_xlabel(r'Sun Azimuth [deg]', fontsize=36)
                axs[row].set_xticks(np.arange(-180, 181, 45))
                axs[row].yaxis.set_major_locator(MultipleLocator(10))   # Major ticks every 1
                axs[row].grid(True)
                # Get the tick positions
                xticks = axs[row].get_xticks()
                # yticks = axs[row].get_yticks()
                # Set labels only every 20
                axs[row].set_xticklabels([str(int(tick)) if tick % 90 == 0 else '' for tick in xticks])
                # axs[row].set_yticklabels([str(int(tick)) if tick % 20 == 0 else '' for tick in yticks])

                axs[row].grid(True)
                if row != 0:
                    axs[row].set_ylabel('')
                axs[row].tick_params(axis='x', which='major', labelsize=32)
                axs[row].tick_params(axis='y', which='major', labelsize=32)
    
    print(f"Azimuth: {sorted_azimuth}")
    #  Adjust figure margins to prevent cut-off labels
    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.1)
    # fig.tight_layout(pad=2)
    if filepath:
        fig.savefig(filepath)

# Varying elevation
def plot_loc_acc_vs_el(base_path, model_types, precision=1, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=1, filepath=None):
    
    fig, axs = plt.subplots(1, n_alt_ranges)  # Maximize width)
  
    elevations = [2, 5, 10, 30, 40, 60, 90]


    for row in range(n_alt_ranges):
        for model_type in model_types:
            corr_loc_list = []
            fail_loc_list = []
            for _, el in enumerate(elevations):
                if el == 90:
                    data = load_data(base_path, model_type, el, 0, obs_el, obs_az)
                else:
                    data = load_data(base_path,  model_type, el, map_az, obs_el, obs_az)
                # err_1to0_list = []
                loc_error_list = data['location_err_acc']
                altitude_list = data['altitude_acc']
                nr_pnp_fails = data['nr_pnp_fails']
                        
                # Convert lists to numpy arrays for easier manipulation
                # err_1to0 = np.array(err_1to0_list)
                loc_error = np.array(loc_error_list)
                altitude = np.array(altitude_list)
                # inliers = np.array(inliers_list)
                

                # Sort the arrays by altitude
                sorted_indices = np.argsort(altitude)
                altitude = altitude[sorted_indices]
                loc_error = loc_error[sorted_indices]
                # inliers = inliers[sorted_indices]
                

                # Split data into three equal parts (low, medium, high altitude ranges)
                altitude_splits = np.array_split(altitude, n_alt_ranges)
                loc_error_splits = np.array_split(loc_error, n_alt_ranges)
                # inliers_splits = np.array_split(inliers, 3)

                
                # Define quantities in each altitude range
                # err_1to0_range=err_1to0_splits[row][0].tolist()
                loc_error_range=loc_error_splits[row]               
                altitude_range=altitude_splits[row]
                # inliers_range=inliers_splits[row].tolist()
                # Compute cumulative distribution
                corr_loc = 100*float(np.sum(np.array(loc_error_range )< precision))/len(loc_error_range)
                fail_loc = 100*float(np.sum(loc_error_range > 10) / len(loc_error_range))
                # avg_inliers = int(np.sum(inliers_range)/len(inliers_range))
                
                corr_loc_list.append(corr_loc)
                fail_loc_list.append(fail_loc)

            print(f"\nAltitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m")
            print(f"Model: {model_type}, precision {precision} m: {corr_loc_list}")
            # print(f"Model: {model_type}, fail loc: {fail_loc_list}")
            # print(f"Model: {model_type}, corr_loc_list: {corr_loc_list}, fail_loc_list: {fail_loc_list}")
            # Plot loc err CDF
            label = MODEL_LABELS[model_type]
            color = MODEL_COLORS[model_type]
            linestyle = MODEL_LINESTYLE[model_type]
            if n_alt_ranges == 1:
                axs.plot(elevations, corr_loc_list , label=label, color=color, linewidth=4, linestyle=linestyle, marker='o')
                # for j in range(len(fail_loc_list)):
                #     axs.annotate(f'{fail_loc_list[j]:.1f}%', (elevations[j], corr_loc_list[j]), fontsize=12, color=model_colors[model_type])
                
                axs.axvline(x=obs_el, color='black', linewidth=4, linestyle='--')
                # axs.set_title(f"Altitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=20)    
                # axs.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.5, 0.8))
                axs.legend(fontsize=28)
                # axs.legend(fontsize=28, loc='upper left', bbox_to_anchor=(0.35, 0.4))
                axs.set_ylabel("Localization accuracy [%]", fontsize=36)
                axs.set_xlabel(r'Sun Elevation [deg]', fontsize=36)
                axs.tick_params(axis='both', which='major', labelsize=32)
                axs.set_ylim(0, 101)
                axs.set_xlim(0, 91)
                # axs.set_xticks(np.arange(-180, 181, 45))
                axs.yaxis.set_major_locator(MultipleLocator(10))
                axs.xaxis.set_major_locator(MultipleLocator(10))
                axs.grid(True)
                fig.set_size_inches(10, 10)
                fig.tight_layout(rect=[0, 0, 1, 0.96])

            else:
                axs[row].plot(elevations, corr_loc_list , label=label, color=color, linewidth=4, linestyle='-', marker='o')
                axs[row].axvline(x=obs_el, color='black', linewidth=4, linestyle='--')
                if row == n_alt_ranges -1 :
                    axs[row].set_title(f"{int(np.min(altitude_range))}-{200} m", fontsize=40)
                else: 
                    axs[row].set_title(f"{int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=40)    
                # axs[row].legend(fontsize=28) if row == 2 else None
                axs[row].legend(fontsize=28, loc='upper left', bbox_to_anchor=(0.02, 0.63)) if row == 2 else None
                axs[row].set_ylabel("Localization accuracy [%]", fontsize=36) if row==0 else None        
                axs[row].set_ylim(0, 101)
                axs[row].set_xlim(0, 91)
                axs[row].set_xlabel(r'Sun Elevation [deg]', fontsize=36)
                axs[row].yaxis.set_major_locator(MultipleLocator(10))
                axs[row].xaxis.set_major_locator(MultipleLocator(10))
                # Customize labels to show only every 20
                # Get the tick positions
                xticks = axs[row].get_xticks()
                # yticks = axs[row].get_yticks()

                # Set labels only every 20
                axs[row].set_xticklabels([str(int(tick)) if tick % 20 == 0 else '' for tick in xticks])
                # axs[row].set_yticklabels([str(int(tick)) if tick % 20 == 0 else '' for tick in yticks])

                axs[row].grid(True)
                if row != 0:
                    axs[row].set_ylabel('')
                axs[row].tick_params(axis='x', which='major', labelsize=32)
                axs[row].tick_params(axis='y', which='major', labelsize=32)

            # # axs[row].set_ylim(-10, 110)
            # # axs[row].set_xlim(-0.1, 5.1)
            #     for ax in axs.flat:
            #         # ax.legend(fontsize=10)
            #         ax.tick_params(axis='both', which='major', labelsize=28)
            #         # ax.set_xlabel(fontsize=14)
            #         # ax.set_ylabel(fontsize=14)
            #         ax.grid(True)


    # fig.suptitle(f'Cumulative Match and Pose Accuracy with Fixed Elevation Diff. {fixed_az - obs_az} deg', fontsize=12)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    #  Adjust figure margins to prevent cut-off labels
    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.1)
    # fig.tight_layout(pad=2) 
    if filepath:
        fig.savefig(filepath)


# Martian day       
def plot_loc_acc_vs_lmst(base_path, model_types, precision=1, n_alt_ranges=3, show_title=True, show_suptitle=True, filepath=None):
    

    """
    Plot localization accuracy at a given precision versus Local Mean Solar Time (LMST) on Mars.
    
    This function visualizes the localization accuracy of different models across various altitude ranges 
    and LMSTs on Mars. The data is grouped into altitude ranges, and the accuracy is plotted for each 
    model type.
    
    Args:
        base_path (str): Path to the results folder
        model_types (list of str): A list of strings representing the types of models to be evaluated 
                   (e.g., 'pretrained', 'finetuned', 'geo', 'sift').
        precision (float, optional): The precision threshold (in meters) for localization accuracy. 
                                     Defaults to 1.
        n_alt_ranges (int, optional): Number of altitude ranges to split the data into. Defaults to 3.
        show_title (bool, optional): Whether to display titles for individual subplots. Defaults to True.
        show_suptitle (bool, optional): Whether to display the overall figure title. Defaults to True.
        filepath (str, optional): Filepath to save the generated plot. If None, the plot is displayed 
                                  interactively. Defaults to None.
    Returns:
        None: The function either displays the plot interactively or saves it to the specified filepath.
    """
     
    
    fig, axs = plt.subplots(1, n_alt_ranges)

    ref_lmst = '15:00'
    ref_hours, ref_minutes = map(int, ref_lmst.split(':'))
    ref_lmst_numeric = ref_hours + ref_minutes / 60.0  

    map_el = 40
    map_az = 180

    daytimes = ['morning_5_30', 'morning_6_00', 'morning_8_00', 'zenith', 'hirise', 'set']
    morning_5_30_dic = {'path':os.path.join(base_path, "mars_mbl_V1_training_morning_5_30_vs-HiRISE_time_shadow_thresh_1"),
               'lmst':'5:30',
               'obsv_el':5,
               'obsv_az':15}
    morning_6_00_dic = {'path':os.path.join(base_path, "mars_mbl_V1_training_morning_6_00_vs-HiRISE_time_shadow_thresh_1"),
               'lmst':'6:00',
               'obsv_el':12,
               'obsv_az':13}
    morning_8_00_dic = {'path':os.path.join(base_path, "mars_mbl_V1_training_morning-vs-HiRISE_time_shadow_thresh_1"),
               'lmst':'8:00',
               'obsv_el':40,
               'obsv_az':5}
    zenith_dic = {'path':os.path.join(base_path, "mars_mbl_V1_training_zenith-vs-HiRISE_time_shadow_thresh_1"),
               'lmst':'11:29',
               'obsv_el':87,
               'obsv_az':270}
    hirise_dic = {'path':os.path.join(base_path, "mars_mbl_V1_training_var_EL_64m_200queries_shadow_thresh_1"),
               'lmst':'15:00',
               'obsv_el':40,
               'obsv_az':180}
    set_dic = {'path':os.path.join(base_path, "mars_mbl_V1_training_set-vs-HiRISE_time_shadow_thresh_1"),
               'lmst':'17:00',
               'obsv_el':12,
               'obsv_az':167}
    
    martian_day_dic = {'morning_5_30':morning_5_30_dic, 'morning_6_00':morning_6_00_dic, 'morning_8_00':morning_8_00_dic, 'zenith':zenith_dic, 'hirise':hirise_dic, 'set':set_dic}


    for row in range(n_alt_ranges):
        for model_type in model_types:
            lmst_list = []
            corr_loc_list = []
            lmst_numeric = []

            for daytime in daytimes:
                
                # Convert LMST to float hours
                lmst = martian_day_dic[daytime]['lmst']
                hours, minutes = map(int, lmst.split(':'))
                lmst_numeric.append(hours + minutes / 60.0)  


                data = load_data(martian_day_dic[daytime]['path'], 
                                 model_type, 
                                 map_el, map_az, 
                                 martian_day_dic[daytime]['obsv_el'], 
                                 martian_day_dic[daytime]['obsv_az'])
  
                loc_error_list = data['location_err_acc']
                altitude_list = data['altitude_acc']

                # Convert lists to numpy arrays for easier manipulation
                loc_error = np.array(loc_error_list)
                altitude = np.array(altitude_list)      

                # Sort the arrays by altitude
                sorted_indices = np.argsort(altitude)
                altitude = altitude[sorted_indices]
                loc_error = loc_error[sorted_indices]
                
                # Split data into three equal parts (low, medium, high altitude ranges)
                altitude_splits = np.array_split(altitude, n_alt_ranges)
                loc_error_splits = np.array_split(loc_error, n_alt_ranges)

                
                # Define quantities in each altitude range
                loc_error_range=loc_error_splits[row]               
                altitude_range=altitude_splits[row]
                # Compute cumulative distribution
                corr_loc = 100*float(np.sum(np.array(loc_error_range )< precision))/len(loc_error_range)
                fail_loc = 100*np.sum(loc_error_range > 100) / len(loc_error_range)
                # avg_inliers = int(np.sum(inliers_range)/len(inliers_range))
                
                corr_loc_list.append(corr_loc)
                lmst_list.append(martian_day_dic[daytime]['lmst'])


            # Plot loc err CDF
            label = MODEL_LABELS[model_type]
            color = MODEL_COLORS[model_type]
            linestyle = MODEL_LINESTYLE[model_type]
    
            if n_alt_ranges == 1:
                
                axs.plot(lmst_numeric, corr_loc_list, label=label,  linestyle=linestyle, marker='o', color=color)
                axs.axvline(x=ref_lmst_numeric, color='black', linewidth=1.5, linestyle='--')
                if show_title:
                    axs.set_title(f"Altitude range: {int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=20)    
                axs.legend(fontsize=20)
                axs.set_ylabel("Pose accuracy [%]", fontsize=20)
                axs.set_xlabel('Locar Mean Solar Time (LMST)', fontsize=20)
                axs.tick_params(axis='both', which='major', labelsize=20)
                axs.set_ylim(0, 101)
                # axs.set_xlim(-180, 181)
                axs.set_xticks(lmst_numeric)
                axs.set_xticklabels(lmst_list)  # Set original LMST labels
                axs.yaxis.set_major_locator(MultipleLocator(10))   # Major ticks every 10
                fig.set_size_inches(12, 12)
                axs.grid(True)
                tick_labels = axs.get_xticklabels()
                tick_labels[0].set_y(-0.02)  # Shift the second label down
                
            else:
                axs[row].plot(lmst_numeric, corr_loc_list, label=label, linewidth=4, linestyle='-', marker='o', color=color)
                axs[row].axvline(x=ref_lmst_numeric, color='black', linewidth=4, linestyle='--')
                if show_title:
                    if row == n_alt_ranges -1 :
                        axs[row].set_title(f"{int(np.min(altitude_range))}-{200} m", fontsize=40)
                    else: 
                        axs[row].set_title(f"{int(np.min(altitude_range))}-{int(np.max(altitude_range))} m", fontsize=40)            
                 # axs[row].legend(fontsize=28) if row == 2 else None
                axs[row].legend(fontsize=26, loc='upper left', bbox_to_anchor=(0.09, 0.55)) if row == 0 else None # 1 m acc.  
                # axs[row].legend(fontsize=26, loc='upper left', bbox_to_anchor=(0.09, 0.3)) if row == 0 else None # 2 m acc. 
                axs[row].set_ylabel("Localization accuracy [%]", fontsize=36) if row==0 else None             
                axs[row].set_ylim(0, 101)
                axs[row].set_xticks(lmst_numeric)
                axs[row].set_xticklabels(lmst_list)  # Set original LMST labels
                axs[row].set_xlabel('LMST', fontsize=36)
                axs[row].yaxis.set_major_locator(MultipleLocator(10))   # Major ticks every 1
                axs[row].grid(True)
                # Get the tick labels and adjust the position of a specific label (e.g., '1.05')
                tick_labels = axs[row].get_xticklabels()
                # tick_labels[0].set_y(-0.08)  # Shift the second label down
                tick_labels[1].set_y(-0.05)  # Shift the second label down
                tick_labels[-1].set_y(-0.05)  # Shift the second label down
                axs[row].tick_params(axis='x', which='major', labelsize=28)
                axs[row].tick_params(axis='y', which='major', labelsize=32)

    if show_suptitle:
        fig.suptitle(f'Localization accuracy @ {precision} m', fontsize=12)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    #  Adjust figure margins to prevent cut-off labels
    fig.subplots_adjust(left=0.08, right=0.986, top=0.95, bottom=0.14)
    # fig.tight_layout(pad=2) 

    if filepath:
        fig.savefig(filepath)


# Terrain morphology
def plot_loc_acc_vs_alt_per_morphology(base_path, model_types, precision=1, terrain_type='dunes', show_title=True, filepath=None):
    """
    Plots localization accuracy versus altitude for different model types and lighting conditions for a specified terrain morphology.
    Args:
        base_path (str): The base directory path where the data files are located.
        model_types (list of str): List of model types to evaluate (e.g., ['pretrained', 'finetuned', 'geo', 'sift']).
        precision (float, optional): The precision threshold for localization accuracy in meters. Defaults to 1.
        terrain_type (str, optional): The type of terrain to analyze (e.g., 'dunes' or 'cliff'). Defaults to 'dunes'.
        show_title (bool, optional): Whether to display the plot title. Defaults to True.
        filepath (str, optional): Filepath to save the plot. If None, the plot will be displayed instead of saved. Defaults to None.
    Returns:
        None: The function generates a plot and either displays it or saves it to the specified filepath.
    """

    fig, axs = plt.subplots()

    altitude_list = [64, 100, 200]
    obsv_sun_el_az = (40, 180)
    map_sun_el_az_list = [(5, 0), (40, 180)]


    sun_el_al_linestyle = {(5, 0): '-',
                           (40, 180): '--'}
    
    for model_type in model_types:
        model_color = MODEL_COLORS[model_type]
        for map_sun_el_az in map_sun_el_az_list:
            obs_el, obs_az = obsv_sun_el_az
            map_el, map_az = map_sun_el_az
            line_style = sun_el_al_linestyle[map_sun_el_az]
            corr_loc_list = []
            for altitude in altitude_list:
                filepath = os.path.join(base_path, f"mars_mbl_V1_training_EL_AZ_sample_{terrain_type}_alt{altitude}m_0angles_shadow_thresh_1")
                print(f"filepath: {filepath}")
                print(f"map_el: {map_el}")
                print(f"map_az: {map_az}")
                print(f"obs_el: {obs_el}")
                print(f"obs_az: {obs_az}")
                data = load_data(filepath, model_type, map_el, map_az, obs_el, obs_az)
                # err_1to0_list = []
                loc_error = data['location_err_acc']
                corr_loc = 100*float(np.sum(np.array(loc_error)< precision))/len(loc_error)
                if map_sun_el_az == (5, 0):
                    print(f"{model_type}, {altitude}: {corr_loc}")
                # avg_inliers = int(np.sum(inliers_range)/len(inliers_range))
                corr_loc_list.append(corr_loc)

            # Plot loc err CDF
            label = f"{MODEL_LABELS[model_type]}, (ΔAZ, ΔEL)=({obs_az - map_az}"+r'$^{\circ}$'+f", {obs_el - map_el}" +r'$^{\circ}$'+")"     
            axs.plot(altitude_list, corr_loc_list , label=label, color=model_color, linestyle=line_style, marker='o')
            # axs.axvline(x=ref_el, color='black', linewidth=1.5, linestyle='--')
            if show_title:
                axs.set_title(f"Localization accuracy @ {precision} m", fontsize=20)    
            # axs.legend(fontsize=22, loc='upper left', bbox_to_anchor=(0.5, 0.8))
            # axs.legend(fontsize=18)
            axs.set_ylabel("Pose accuracy [%]", fontsize=22)
            axs.set_xlabel("Altitude [m]", fontsize=22)
            axs.tick_params(axis='both', which='major', labelsize=22)
            axs.set_ylim(0, 101)
            # axs.set_xlim(0, 91)
            # axs.set_xticks(np.arange(-180, 181, 45))
            axs.yaxis.set_major_locator(MultipleLocator(10))
            # axs.xaxis.set_major_locator(MultipleLocator(10))
            axs.grid(True)
            fig.set_size_inches(14, 14)
    
    if filepath:
        fig.savefig(filepath)

# Comparison w GAM
def plot_gam_comp(fig, axs, base_path, model_types, test_queries=50, map_el=40, map_az=180, obs_el=40, obs_az=180, pose_prior=False, uncertainty="low", filepath=None):
    
    for model_type in model_types:
        # data = load_json_data(base_path, tests_name, model_type, fixed_el, fixed_az, obs_el, obs_az, n_queries, top_k=top_k, conf_th=conf_th)
        data = load_data(base_path, model_type, map_el, map_az, obs_el, obs_az, pose_prior=pose_prior, uncertainty=uncertainty)
        
        # err_1to0_list = []
        loc_error = data['location_err_acc']
        altitude = data['altitude_acc']


        loc_error_list = []
        altitude_list = []
        for i in range(test_queries):
            if i < len(altitude):     
                loc_error_list.append(loc_error[i])
                altitude_list.append(altitude[i])
            
        loc_error = np.array(loc_error_list)
        altitude = np.array(altitude_list)
        

        bins = np.arange(0.1, 10, 0.1)
        # Define quantities in each altitude range
        # err_1to0_range=err_1to0_splits[row][0].tolist()
    
        # Compute cumulative distribution
        loc_err_cum = [100*np.sum(loc_error < Th) / len(loc_error) for Th in bins]
        corr_loc = 100*np.sum(loc_error < 1) / len(loc_error)
        corr_loc_5m = 100*np.sum(loc_error < 5) / len(loc_error)
        fail_loc = 100*np.sum(loc_error > 10000) / len(loc_error)
        
        # Create label with additional info     
        # umulative pose accuracy vs bins
        label = f"{MODEL_LABELS[model_type]}\n(@1m: {'{:.1f}'.format(corr_loc)}%, @5m: {'{:.1f}'.format(corr_loc_5m)}%)"
        color = MODEL_COLORS[model_type]
        linestyle = MODEL_LINESTYLE[model_type]
        axs.plot(bins, loc_err_cum, label=label, linewidth=4, linestyle=linestyle, color=color)
        axs.set_ylim(0, 101)
        axs.set_xlim(0, 10)

        # axs[row, col].legend(fontsize=10)

        # Make the legend, labels, and ticks larger     
        axs.set_ylabel('Cumulative Accuracy [%]', fontsize=36)
        axs.set_xlabel('Localization error [m]', fontsize=36)
        # axs.legend(fontsize=28, loc='upper left', bbox_to_anchor=(0.35, 0.4))
        axs.legend(fontsize=26)
        axs.tick_params(axis='both', which='major', labelsize=32)
        axs.grid(True)
        axs.yaxis.set_major_locator(MultipleLocator(10))
        fig.set_size_inches(10, 10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
           
    # fig.suptitle(f'Cumulative Pose Accuracy \nAZ diff: {fixed_az - obs_az} deg, EL diff: {fixed_el - obs_el} deg \nAltitude range: 64-200m', fontsize=22)
    if filepath:
        fig.savefig(filepath)

