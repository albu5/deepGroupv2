import numpy as np
from utils import read_anno, pairwise_features_interaction_batch, pairwise_features_at_time_t,\
    pairwise_interactions_at_time_t


anno_str = "../data/csv_anno/anno01.mat"
x = read_anno(anno_path=anno_str)
# print(x.shape)
pf = pairwise_features_at_time_t(x, 0)
pi = pairwise_interactions_at_time_t(x, 0)
print(pi)
px, py = pairwise_features_interaction_batch(pf, pi)





