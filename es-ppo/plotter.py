import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set(color_codes=True)

## change experiment names here to plot

data_folder = 'logs'
if not os.path.isdir(data_folder):
    print('WARNING! LOG FOLDER NOT FOUND, MAKE SURE YOU HAVE CORRECT DATA FOLDER')
simple_colors = ['red','green','blue','purple']
colors = ['red','orange','yellow','green','blue','purple','black','grey','olive','cyan','pink','darkgreen','lightblue','plum']

'''HalfCheetah'''
title = 'Compare ES pop_size'
exp_names = ['HC_es_pop100_1515000','HC_es_pop50_765000']

title = 'Compare PPO n_episode'
exp_names = ['HC_pg_nep21_315000','HC_pg_nep50_750000']
##
title = 'HalfCheetah Compare ES/PPO'
exp_names = ['HC_espg_50_50_761733','HC_espg_50_21_694000','HC_espg_100_21_1288333','HC_es_pop100_1515000','HC_pg_nep50_750000']
label_names = ['HC_espg_50_50_375_761733','HC_espg_50_21_375_694000','HC_espg_100_21_375_1288333','HC_es_pop100_375_1515000','HC_pg_nep50_375_750000']

title = 'HalfCheetah Compare ES/PPO'
exp_names = ['HC_espg_50_21_350_625000','HC_espg_50_50_350_761066','HC_espg_100_21_350_1088333','HC_es_pop100_1515000','HC_pg_nep50_750000']

exp_names = ['HC_es_pop100_1515000','HC_espg_100_21_350_1088333','HC_espg_100_21_1288333']
label_names = ['HC_es_pop100_1515000','HC_espg_100_21_350_1088333','HC_espg_100_21_375_1288333']

title = 'HalfCheetah Compare ES/PPO'
exp_names = ['1_HC_espg_765000','HC_es_pop50_765000','HC_pg_nep50_750000']

#'''Hopper'''
#title = 'Compare ES pop_size'
#exp_names = ['HP_es_pop50_765000','HP_es_pop100_1515000']
##
#title = 'Compare PPO n_episode'
#exp_names = ['HP_pg_nep21_315000','HP_pg_nep50_750000']
####
#title = 'Hopper Compare ES/PPO'
#exp_names = ['HP_es_pop50_255000','HP_pg_nep21_105000','HP_espg_50_50_g50_250966','HP_espg_50_21_129000']
#label_names = ['HP_es_pop50_255000','HP_pg_nep21_105000','HP_espg_50_50_250966','HP_espg_50_21_129000']
#
#'''InvertedPendulum'''
#title = 'Compare ES pop_size'
#exp_names = ['IP_es_pop100_505000','IP_es_pop50_255000']
#
#title = 'Compare PPO n_episode'
#exp_names = ['IP_pg_nep21_105000','IP_pg_nep50_250000']
#
#title = 'InvertedPendulum Compare ES/PPO'
#exp_names = ['IP_espg_50_50_120_252400','IP_espg_50_50_110_252100','IP_es_pop50_255000','IP_pg_nep50_250000']
#
#title = 'InvertedPendulum Compare ES/PPO'
#exp_names = ['1_IP_espg_255000','IP_es_pop50_255000','IP_pg_nep50_250000']

#'''InvertedDoublePendulum'''
#title = 'ES tuning'
#exp_names = ['IDP_es_pop100_a0.1_s0.01_1515000','IDP_es_pop100_a0.1_s0.02_1515000','IDP_es_pop100_a0.1_s0.05_1515000']
#


label_names = exp_names ## use this if names are same

n = len(exp_names)
data_list = []
for i in range(n):
    data = np.loadtxt(os.path.join(data_folder,exp_names[i]))
#    print(len(data))
    data_list.append(data)


for i in range(n):
    ax = sns.tsplot(data=data_list[i], color=colors[i], condition=label_names[i])

plt.xlabel('Gen')
plt.ylabel('Return')
plt.title(title)
plt.tight_layout()
plt.show()
