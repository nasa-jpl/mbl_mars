
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib import cm
from models.LoFTR.src.utils.plotting import make_matching_figure
from utils.match_utils import get_top_k


def prepare_img(img):
    if (len(img.shape)==3) and (img.shape[0]==1):
        img = img.squeeze(0)
    if np.amax(img) <= 1:
        img = img * 255.0
    return img


# def plot_correspondences(im1, im2, points1, points2, title1=None, title2=None, points_thresh=True, text=None, save_filepath=None):
#     im1 = prepare_img(im1)
#     im2 = prepare_img(im2)
    
#     if points_thresh:
#         if len(points1) > 1000:
#             points1 = points1[::100, :]
#             points2 = points2[::100, :]

#     color_str = "orange"
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12), dpi=100)
#     ax1.imshow(im1, aspect='equal', cmap='gray')
#     if title1 is not None:
#         ax1.set_title(title1)
#     ax2.imshow(im2, aspect='equal', cmap='gray')
#     if title2 is not None:
#         ax2.set_title(title2)

#     for i in range(len(points1)):
#         ax1.plot([points1[i,0]], [points1[i,1]], marker='.', markersize=5, color=color_str)
#         ax1.text(points1[i,0]+5, points1[i,1]-10, str(i), fontsize=15, color=color_str)
#         ax2.plot([points2[i,0]], [points2[i,1]], marker='.', markersize=5, color=color_str)
#         ax2.text(points2[i,0]+5, points2[i,1]-10, str(i), fontsize=15, color=color_str)
#     ax1.axis("off")
#     ax2.axis("off")
#     if text:
#         ax1.text(0.95, 0.95, text, ha='right', va='top', transform=ax1.transAxes, fontsize=16, color='white')
#         ax1.set_facecolor('black')
#     plt.tight_layout()
#     plt.draw()
#     if save_filepath:
#         fig.savefig(save_filepath, format='png')
#     else:
#         plt.show()


def plot_imgs(im1, im2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(im1, aspect='equal')
    ax2.imshow(im2, aspect='equal')
    #ax1.axis("off")
    #ax2.axis("off")
    plt.tight_layout()
    plt.draw()
    plt.show()

def plot_map_box(query_img_gray, map_gray, box, query_id, save_dir=None):
    # box is [left, top, right, bottom]
    [r_start, c_start, r_end, c_end] = box
    map_crop_img = map_gray[c_start:c_end, r_start:r_end]

    color_win = "orange"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    ax1.imshow(query_img_gray.cpu(), aspect='equal')
    ax1.set_title('Query')
    ax2.imshow(map_crop_img.cpu(), aspect='equal')
    ax2.set_title("Map search box")

    left, top, right, bottom = r_start, c_start, r_end, c_end
    width, height=right-left, bottom-top

    ax3.imshow(map_gray.cpu(), aspect='equal')
    ax3.set_title("Map with window of width="+str(width)+", height="+str(height))

    rect = Rectangle((left, top), width, height, linewidth=2, edgecolor=color_win, facecolor='none')
    ax3.add_patch(rect)

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.tight_layout()
    plt.draw()
    #plt.show()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "window_query_"+str(query_id)+".png"))
        plt.close(fig)
    else:
        return fig

def plot_correspondences(im1, im2, points1, points2, points_noisy, query_id, save_dir="./correspondeces"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    color_str = "orange"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    ax1.imshow(im1, aspect='equal')
    ax1.set_title('From Query')
    ax2.imshow(im2, aspect='equal')
    ax2.set_title("To Map")
    ax3.imshow(im2, aspect='equal')
    ax3.set_title("To Map Noisy")
    for i in range(len(points1)):
        ax1.plot([points1[i,0]], [points1[i,1]], marker='*', markersize=15, color=color_str)
        ax1.text(points1[i,0]+5, points1[i,1]-10, str(i), fontsize=15, color=color_str)
        ax2.plot([points2[i,0]], [points2[i,1]], marker='*', markersize=15, color=color_str)
        ax2.text(points2[i,0]+5, points2[i,1]-10, str(i), fontsize=15, color=color_str)
        ax3.plot([points_noisy[i,0]], [points_noisy[i,1]], marker='*', markersize=15, color=color_str)
        ax3.text(points_noisy[i,0]+5, points_noisy[i,1]-10, str(i), fontsize=15, color=color_str)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    plt.tight_layout()
    plt.draw()
    #plt.show()
    if save_dir:
        fig.savefig(os.path.join(save_dir,"points_query_"+str(query_id)+".png"))
        plt.close(fig)
    else:
        return fig

def show_matches(batch, top_k=100, path=None):
    img0 = (batch['image0'][0][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (batch['image1'][0][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = batch['mkpts0_f'].cpu().numpy()
    kpts1 = batch['mkpts1_f'].cpu().numpy()
    #print(batch['scale0'].shape)
    #print(kpts0.shape)
    if 'scale0' in batch:
        kpts0 = kpts0 / batch['scale0'][0].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / batch['scale1'][0].cpu().numpy()[[1, 0]]

    mconf = batch['mconf'].cpu().numpy()

    # keep only the top k matches for visual clarity
    kpts0, kpts1, mconf, _ = get_top_k(kpts0, kpts1, mconf, top_k=top_k)

    color = cm.jet(mconf, alpha=0.7)
    text = ['Matches: {}'.format(len(kpts0))]
    figure = make_matching_figure(img0, img1, kpts0, kpts1, color, text=text, path=path)

def show_matches_on_map(query, map, kpts0, kpts1, mconf, top_k=100, map_box=None, path=None):
    
    

    # Create copies of keypoints to avoid modifying the originals
    kpts0_copy = np.copy(kpts0)
    kpts1_copy = np.copy(kpts1)
    mconf_copy = np.copy(mconf)

    # Ensure kpts0 and kpts1 are 2D arrays (even if there's only one keypoint)
    if kpts0_copy.ndim == 1:
        kpts0_copy = np.expand_dims(kpts0_copy, axis=0)
    if kpts1_copy.ndim == 1:
        kpts1_copy = np.expand_dims(kpts1_copy, axis=0)

    if map_box:
        [r_start, c_start, r_end, c_end] = map_box 
        map_crop = map[c_start:c_end, r_start:r_end]
        # Translate map kpts to map search area
        kpts1_copy[:,0] -= r_start
        kpts1_copy[:,1] -= c_start
    else:
        map_crop =map
        
    # print(f"kpts1[0] after translating: {kpts1[0]}")

    # keep only the top k matches for visual clarity
    kpts0_copy, kpts1_copy, mconf_copy, _ = get_top_k(kpts0_copy, kpts1_copy, mconf_copy, top_k=top_k)

    color = cm.jet(mconf_copy, alpha=0.7)
    text = ['Matches: {}'.format(len(kpts0_copy))]
    figure = make_matching_figure(query, map_crop, kpts0_copy, kpts1_copy, color, text=text, path=path)

