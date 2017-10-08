



feature_map_dl = np.save(path + '/K562vsGM12878_chr18_size100_step10_Pearson_important_samples', feature_map)
feature_map_pixel = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples.npy')
feature_map_pearson = np.load(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples.npy')