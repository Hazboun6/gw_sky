import scipy as sci
import numpy as np
import subprocess
import os

def find_bin(source_freq, source_strain, bin_num = 1):
    """
    Finds the maximum bin number allowed within given source number that will not result in discontinous PSD curve.
    """
    
    binned_h = sci.stats.binned_statistic(source_freq, source_strain, 'sum', bin_num)
    if 0.00000000e+00 in binned_h[0]:
        return bin_num-1

    else:
        bin_num += 1 
        return find_bin(source_freq, source_strain, bin_num)

    
def plot_points(source_strain, source_freq, ax3, p, L):
    """
    Plots the individual sources on the Strain vs Freq subplot of the SMBBHs
    """
    
    
    if p =='all': # If requested source is all requested sources, add them.
    
        ax3.scatter(source_freq[0:L], source_strain[0:L], marker = ".", s = 100, color = 'indigo')
        plot_psd(ax3, source_freq[0:L], source_strain[0:L])
        
    if isinstance(p,list): # If requested source is a list of multiple sources, add them.
        
        for i in range(0, len(p)):
            ax3.scatter(source_freq[i], source_strain[i], marker = ".", s = 100, color = 'indigo')
        len_p = len(p)
        plot_psd(ax3, source_freq[0:len_p], source_strain[0:len_p])

    else: # Otherwise, add single source to SMBBH figure.
        ax3.scatter(source_freq[0], source_strain[0], marker = ".", s = 100, color = 'indigo')

    
def plot_psd(ax3, passed_freq, passed_strain):
    """
    Calculates and plots the PSD on the Strain vs freq subplot
    """

    binNum = find_bin(passed_freq, passed_strain)
    # PSD Calculation
    binned_strain = sci.stats.binned_statistic(passed_freq, passed_strain, 'sum', binNum)
    binned_freq = sci.stats.binned_statistic(passed_freq, passed_freq, 'sum', binNum)
    binned_median = sci.stats.binned_statistic(passed_freq, passed_freq, 'median', binNum)
    
    # Strain Calculation
    S_hi = (binned_strain[0]*binned_strain[0])/binned_freq[0]
    H = np.sqrt(S_hi*binned_freq[0])
    ax3.plot(binned_median[0], H, label = "PSD", linewidth = 1.5)
    
    
def smbhb_freq_range(freq_pref, smbhb_count):
    """
    Returns an array with indeces associated with sources requested.
    """
    
    
    if freq_pref == 1: # Random sample of all sources
    
        sample = np.random.choice(199999, smbhb_count)
        print('Random Sample')
        return sample
    
    elif freq_pref == 2: # Loudest sources only
        print('Loudest Only')
        sample = np.arange(0, smbhb_count)
        return sample
         
         
def path_creation():
    """
    Creates a series of folders and paths for image creation in the directory this script is located.
    """
    newpath = os.getcwd() # Grabs current path
    paths_dict = {}
    i = 1
    while True:
        
        val = 'gw_sky_vis'
        new_folder_name = f'{val}_{i}'
        new_folder_path = os.path.join(newpath, new_folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            break
        i += 1
    
    paths_dict[0] = 'gifs'
    paths_dict[1] = 'images'
    paths_dict[2] = 'pallette'
    paths_dict[3] = 'folder_name'
    
    
    paths_dict['gifs'] = os.path.join(new_folder_path, 'gifs')
    paths_dict['images'] = os.path.join(new_folder_path, 'images')
    paths_dict['pallette'] = os.path.join(new_folder_path, 'pallete')
    paths_dict['folder_name'] = new_folder_name
    

    os.makedirs(paths_dict['gifs'])
    os.makedirs(paths_dict['images'])
    os.makedirs(paths_dict['pallette'])
    print('Folders successfully created')

    return paths_dict
    

def giffify(input_pattern, output_file, pallet_path, gif_fps = 30):
    """
    Given input, output, pallete paths creates a gif from generated Matplotlib images.
    """
    create_pallete = (
        f"ffmpeg -i {input_pattern} "
        f"-vf palettegen {pallet_path}"
    )
    
    create_gif = (
        f"ffmpeg -y -i {input_pattern} "
        f"-i {pallet_path} -filter_complex "
        f'"fps={gif_fps},scale=1032:-1:flags=lanczos[x];[x][1:v]paletteuse" -loop 0 {output_file}'
    )

    try:
        # Execute the FFmpeg command using subprocess
        subprocess.run(create_pallete, shell=True, check=True)
        print("Pallete creation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during pallete creation process: {e}")

    try:
        # Execute the FFmpeg command using subprocess
        subprocess.run(create_gif, shell=True, check=True)
        print("Gif creation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during gif creation process: {e}")
        
        
def source_creator(source_iterations, source_iteration_order, psrs = []):
    """
    Creates a list of lists that contains requested source count.
    """
    
    split_order = source_iteration_order.split(',')
    source_it = []
    psrs = []

    for i in split_order:
        source_it.append(int(i))

    for i in range(int(source_iterations)):
        psrs.append([])
    for i in range(int(source_iterations)):
        for j in range(source_it[i]):
            psrs[i].append(j)
            
    return psrs