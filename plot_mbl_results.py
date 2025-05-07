
import os
import matplotlib.pyplot as plt
import utils.results_utils as plt_utils 


BASE_PATH = "/Users/dariopisanti/Documents/PhD_Space/NASA_JPL/LORNA/MBL_results/jezero_crater"
var_EL_results_path = os.path.join(BASE_PATH, "mars_mbl_V1_training_var_EL_64m_200queries_shadow_thresh_1")
var_AZ_results_path = os.path.join("/Volumes/TOSHIBA EXT/LORNA/Phd_thesis/Results", "mars_mbl_V1_training_var_AZ_64m_200queries_shadow_thresh_1")
martian_day_results_path = "/Volumes/TOSHIBA EXT/LORNA/Phd_thesis/Results"
morphology_results_path = "/Volumes/TOSHIBA EXT/LORNA/Phd_thesis/Results"
off_nadir_results_path = os.path.join(BASE_PATH, "mbl_V1_100m_0yaw_15viewpoint_test")
ctx_results_path = '/Users/dariopisanti/repos/lorna_software_backup/mbl_lab_backup/results/jezero_CTX_500queries_64-100m'
model_types = ['geo', 'finetuned', 'pretrained','sift']


##### FIXED SUN AZ AND EL
results_path = var_AZ_results_path
map_az =0
map_el = 10
obs_az = 0
obs_el = 10
# Localization accuracy
plt_utils.plot_loc_cum_acc(results_path, model_types, map_el, map_az, obs_el, obs_az , n_alt_ranges=1, show_2m_precision=False, show_title=False)
# plt_utils.plot_loc_cum_acc_alt_range(results_path, model_types, min_alt=100, max_alt=200, map_az=180, map_el=40, obs_el=40, obs_az=180)
# plt_utils.plot_loc_err_density_vs_alt(results_path, "geo", map_az=180, map_el=40, obs_el=40, obs_az=180, n_alt_ranges=1)
# Matching accuracy
plt_utils.plot_match_cum_acc(results_path, model_types, map_el, map_az, obs_el, obs_az, n_alt_ranges=1, show_title=True)


######## MAP SUN ELEVATION VARIATION #########
# Localization accuracy
plt_utils.plot_loc_acc_vs_el(var_EL_results_path, model_types, precision=1, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=3)

######## MAP SUN AZIMUTH VARIATION #########
# Localization accuracy
plt_utils.plot_loc_acc_vs_az(var_AZ_results_path, model_types, precision=1, map_el=10, obs_el=10, obs_az=0, n_alt_ranges=3)

######## MARTIAN DAY ##########################
plt_utils.plot_loc_acc_vs_lmst(martian_day_results_path, model_types, precision=1, n_alt_ranges=3, show_title=False, show_suptitle=False)

######## TERRAIN MORPHOLOGY ###################
plt_utils.plot_loc_acc_vs_alt_per_morphology(morphology_results_path, model_types, precision=1, terrain_type='dunes')

######## EFFECT OF SCALE CHANGES
plt_utils.plot_loc_cum_acc(var_EL_results_path, model_types, map_el=40, map_az=180, obs_el=40, obs_az=180, n_alt_ranges=3, show_title=True)

######## EFFECT OF OFF-NADIR ATTITUDE
off_nadir_title = "0° yaw, 15° max pitch, 15° max roll"
nadir_title = "Nadir"
plt_utils.plot_loc_cum_acc_alt_range(off_nadir_results_path, model_types, min_alt=64, max_alt=200, map_el=40, map_az=180, obs_el=40, obs_az=180, title=off_nadir_title)
plt_utils.plot_loc_cum_acc_alt_range(var_EL_results_path,    model_types, min_alt=64, max_alt=200, map_el=40, map_az=180, obs_el=40, obs_az=180, title=nadir_title)

######## EFFECT OF CTX-like vs HiRISE-like
hirise_title = "HiRISE-like"
ctx_title = "CTX-like"
ctx_model_types = ['geo_ctx_v2', 'geo', 'pretrained','sift']
plt_utils.plot_loc_cum_acc_alt_range(ctx_results_path,    ctx_model_types, min_alt=64, max_alt=200, map_el=40, map_az=180, obs_el=40, obs_az=180, title=ctx_title)
plt_utils.plot_loc_cum_acc_alt_range(var_EL_results_path,     model_types, min_alt=64, max_alt=200, map_el=40, map_az=180, obs_el=40, obs_az=180, title=hirise_title)


######## GAM COMPARISON ##############
fig, axs = plt.subplots()
wo_gam_exp_path = var_EL_results_path
wo_gam_model_types =['geo', 'finetuned', 'pretrained', 'sift']
wo_gam_test_nqueries = 500
w_gam_exp_path = '/Users/dariopisanti/repos/lorna_software_backup/mbl_lab_backup/results/jezero_hirise_baselines_500queries'
w_gam_test_nqueries = 500
w_gam_model_types =['finetuned_gam', 'pretrained_gam', 'sift_gam']
plt_utils.plot_gam_comp(fig, axs, wo_gam_exp_path, wo_gam_model_types, wo_gam_test_nqueries, map_el=40, map_az=180, obs_el=40, obs_az=180)
plt_utils.plot_gam_comp(fig, axs, w_gam_exp_path ,  w_gam_model_types,  w_gam_test_nqueries, map_el=40, map_az=180, obs_el=40, obs_az=180)

plt.show()

   

