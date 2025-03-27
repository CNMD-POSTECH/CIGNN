import numpy as np
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

filename = 'pred.npy'
font_path = '/scratch/x3100a06/miniconda3/fonts/Arial_Bold.ttf'
title_fontprop = fm.FontProperties(fname=font_path)

fig_height = 12
fig_width = 12

title_size_ratio = 0.07
axis_title_size_ratio = 0.06
tick_label_size_ratio = 0.05
legend_size_ratio = 0.05
text_size_ratio = 0.04

basis = 6

title_size = title_size_ratio * basis * 72  # 1인치 = 72포인트
axis_title_size = axis_title_size_ratio * basis * 72
tick_label_size = tick_label_size_ratio * basis * 72
legend_size = legend_size_ratio * basis * 72
text_size = text_size_ratio * basis * 72

def create_scatter_plot(ax, x, y, xlabel, ylabel, color, xmin, ymin, xmax, ymax, range_step, r2, default=0.005, label=None):
    ax.scatter(x, y, color=color, alpha=0.1, s=5)
    ax.plot([xmin-0.1, xmax+0.1], [xmin-0.1, xmax+0.1], color=color, linestyle='--', linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=axis_title_size)
    ax.set_ylabel(ylabel, fontsize=axis_title_size)
    
    ax.set_xticks(np.arange(xmin, xmax, range_step))
    ax.set_yticks(np.arange(ymin, ymax, range_step))
    
    # 축 스타일 설정
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(2)
        
    # 축 눈금 설정
    ax.tick_params(axis='y', length=5, direction='in', width=2, labelsize=tick_label_size)
    ax.tick_params(axis='x', length=5, direction='in', width=2, labelsize=tick_label_size)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin-default, xmax+default)
    ax.set_ylim(ymin-default, ymax+default)
    
    # R^2 값 표시
    ax.text(0.05, 0.95, f'R²: {r2:.2f}', transform=ax.transAxes, fontsize=axis_title_size, verticalalignment='top', color=color, weight='bold', fontproperties=title_fontprop)
    
    # Label 표시
    if label is not None:
        if label == 'a' or label == 'c':
            ax.text(-0.34, 1.0, label, transform=ax.transAxes, fontsize=axis_title_size+10, fontweight='bold', va='top', ha='right', fontproperties=title_fontprop)
        elif label == 'b' or label == 'd':
            ax.text(-0.24, 1.0, label, transform=ax.transAxes, fontsize=axis_title_size+10, fontweight='bold', va='top', ha='right', fontproperties=title_fontprop)
        
# 파일로부터 데이터를 불러오기
data = np.load(filename, allow_pickle=True)
TE, PE, TF, PF, TQ, PQ = [], [], [], [], [], []
for d in data:
    TE.append(d['energy']['ground_truth'])
    PE.append(d['energy']['prediction'])
    TF.append(d['forces']['ground_truth'])
    PF.append(d['forces']['prediction'])
    TQ.append(d['charge']['ground_truth'])
    PQ.append(d['charge']['prediction'])
TE = np.array(TE)
PE = np.array(PE)
TF = np.concatenate(TF, axis=0)
PF = np.concatenate(PF, axis=0)
TQ = np.concatenate(TQ, axis=0)
PQ = np.concatenate(PQ, axis=0)

print(TE.shape, PE.shape)
print(TF.shape, PF.shape)
print(TQ.shape, PQ.shape)

# 에너지에 대한 mae, rmse, r2 계산
energy_mae = mean_absolute_error(TE, PE)
energy_rmse = np.sqrt(mean_squared_error(TE, PE))
energy_r2 = r2_score(TE, PE)
print('\n')
print('Energy:')
print('MAE : {:.3f}'.format(energy_mae * 1000))
print('RMSE : {:.3f}'.format(energy_rmse * 1000))
print('R2 : {:.3f}'.format(energy_r2))

# Force에 대한 mae, rmse, r2 계산
force_mae = mean_absolute_error(TF, PF)
force_rmse = np.sqrt(mean_squared_error(TF, PF))
force_r2 = r2_score(TF, PF)
print('\n')
print('Force:')
print('MAE : {:.3f}'.format(force_mae * 1000))
print('RMSE : {:.3f}'.format(force_rmse * 1000))
print('R2 : {:.3f}'.format(force_r2))

TQ_POS = TQ[TQ > 0]
PQ_POS = PQ[TQ > 0]
charge_mae_pos = mean_absolute_error(TQ_POS, PQ_POS)
charge_rmse_pos = np.sqrt(mean_squared_error(TQ_POS, PQ_POS))
charge_r2_pos = r2_score(TQ_POS, PQ_POS)
print('\n')
print('Charge (positive):')
print('MAE : {:.3f}'.format(charge_mae_pos))
print('RMSE : {:.3f}'.format(charge_rmse_pos))
print('R2 : {:.3f}'.format(charge_r2_pos))

TQ_NEG = TQ[TQ < 0]
PQ_NEG = PQ[TQ < 0]
charge_mae_neg = mean_absolute_error(TQ_NEG, PQ_NEG)
charge_rmse_neg = np.sqrt(mean_squared_error(TQ_NEG, PQ_NEG))
charge_r2_neg = r2_score(TQ_NEG, PQ_NEG)
print('\n')
print('Charge (negative):')
print('MAE : {:.3f}'.format(charge_mae_neg))
print('RMSE : {:.3f}'.format(charge_rmse_neg))
print('R2 : {:.3f}'.format(charge_r2_neg))

# Calculate the global min and max
energy_min = min(TE.min(), PE.min())
energy_max = max(TE.max(), PE.max())
force_min = min(TF.reshape(-1,1).min(), PF.reshape(-1,1).min())
force_max = max(TF.reshape(-1,1).max(), PF.reshape(-1,1).max())
charge_min = min(TQ.min(), PQ.min())
charge_max = max(TQ.max(), PQ.max())

print('Energy:', energy_min, energy_max)
print('Force:', force_min, force_max)
print('Charge:', charge_min, charge_max)

energy_min = -10.82
energy_max = -10.698
force_min = -9
force_max = 9.01
positive_charge_min = 1.5
positive_charge_max = 3.01
negative_charge_min = -1.5
negative_charge_max = -1.09

r2_force = r2_score(TF, PF)
r2_energy = r2_score(TE, PE)
r2_positive_charge = r2_score(TQ_POS, PQ_POS)
r2_negative_charge = r2_score(TQ_NEG, PQ_NEG)

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.225, wspace=0.35)

# 에너지 scatter plot
ax1 = fig.add_subplot(gs[0, 0])
create_scatter_plot(ax1, TE, PE, r'E$_{\mathrm{AIMD}}$ (eV/Atom)', r'E$_{\mathrm{CNMP}}$ (eV/Atom)', color='#6E8FCA', xmin=energy_min, ymin=energy_min, xmax=energy_max, ymax=energy_max, range_step=0.04, r2=r2_energy, label='a')

# Force scatter plot
ax2 = fig.add_subplot(gs[0, 1])
create_scatter_plot(ax2, TF, PF, r'E$_{\mathrm{AIMD}}$ (eV/Å)', r'F$_{\mathrm{CNMP}}$ (eV/Å)', color='#CA726E', xmin=force_min, ymin=force_min, xmax=force_max, ymax=force_max, range_step=3, r2=r2_force, label='b')

# Positive charge scatter plot
ax3 = fig.add_subplot(gs[1, 0])
create_scatter_plot(ax3, TQ_POS, PQ_POS, r'Hf Q$_{\mathrm{AIMD}}$ (e)', r'Hf Q$_{\mathrm{CNMP}}$ (e)', color='#D0B17C', 
                    xmin=positive_charge_min, ymin=positive_charge_min, xmax=positive_charge_max, ymax=positive_charge_max, range_step=0.5, r2=r2_positive_charge, default=0.05, label='c')

# Negative charge scatter plot
ax4 = fig.add_subplot(gs[1, 1])
create_scatter_plot(ax4, TQ_NEG, PQ_NEG, r'O Q$_{\mathrm{AIMD}}$ (e)', r'O Q$_{\mathrm{CNMP}}$ (e)', color='#B4CA6E', 
                    xmin=negative_charge_min, ymin=negative_charge_min, xmax=negative_charge_max, ymax=negative_charge_max, range_step=0.1, r2=r2_negative_charge, default=0.01, label='d')

plt.subplots_adjust(hspace=0.225)
plt.savefig('/scratch/x3100a06/MLP_Paper/CIGNN/dataset/Figure_2/figure2.png', bbox_inches='tight', dpi=300)