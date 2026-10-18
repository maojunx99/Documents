"""Reproduce all blog figures using only data/blog_data.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

HERE=Path(__file__).resolve().parent
FIG=HERE/'figures'
FIG.mkdir(exist_ok=True)
D=json.loads((HERE/'data/blog_data.json').read_text(encoding='utf-8'))
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.labelcolor':'#334155','text.color':'#172033','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#cbd5e1','xtick.color':'#475569','ytick.color':'#475569',
    'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none',
})
BLUE='#2563eb'; ORANGE='#db6b21'; GREEN='#0f8a7b'; GRAY='#9ca3af'
BENCH=D['benchmarks']
LABELS=['GSM8K','MATH500','HumanEval','MBPP','MT-Bench']

def finish(fig,name,note=None):
    if note:
        fig.set_layout_engine('constrained',rect=(0,.08,1,1))
        fig.text(.015,.015,note,ha='left',va='bottom',fontsize=9,color='#64748b')
    fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def grid(ax):
    ax.grid(axis='y',alpha=.16,zorder=0)
    ax.set_axisbelow(True)

# 1. Token mixture and actual answer length, not nominal directory sizes.
mix=D['targeted_mix']; buckets=['short_function','code_reasoning_and_repair','general_instruction','math','replay']
names=['短函数代码','代码推理/修复','通用指令','数学','已有训练数据']
colors=[BLUE,'#739df1',GREEN,ORANGE,GRAY]
fig,axes=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
v=[mix['pool_buckets'][b]/mix['pool_supervised_tokens']*100 for b in buckets]
bars=axes[0].barh(names[::-1],v[::-1],color=colors[::-1],height=.62)
axes[0].bar_label(bars,fmt='%.1f%%',padding=6,fontsize=11)
axes[0].set_xlim(0,41);axes[0].set_xlabel('参与监督的回答 token 占比（%）');axes[0].set_title('补充数据：按回答 token 配比')
x=np.arange(5);stats=[mix['length_statistics']['train/'+b] for b in buckets]
thinking=np.array([s['thinking_tokens']['mean'] for s in stats]);final=np.array([s['final_tokens_including_end']['mean'] for s in stats])
axes[1].bar(x,thinking,width=.58,color=BLUE,label='思考过程')
axes[1].bar(x,final,bottom=thinking,width=.58,color=GREEN,label='最终答案（含结束标记）')
axes[1].scatter(x,[s['input_tokens']['mean'] for s in stats],marker='_',s=300,color=ORANGE,linewidths=2.2,label='问题与回答总长')
axes[1].set_xticks(x,['短函数','代码推理\n/修复','通用指令','数学','已有训练\n数据'])
axes[1].set_ylabel('平均 token 数');axes[1].set_ylim(0,2050);axes[1].set_title('补充数据的平均长度（上限 4096 token）');grid(axes[1])
axes[1].legend(fontsize=9,frameon=False,ncol=2,loc='upper left')
finish(fig,'01_data_mix_and_lengths')

# 2. All five validation endpoints, including the recovered epoch0.
ep=D['scratch_validation'];x=np.arange(1,6)
fig,axes=plt.subplots(1,3,figsize=(13.4,4.4),layout='constrained')
axes[0].plot(x,[r['loss_epoch'] for r in ep],'-o',color=BLUE,lw=2)
axes[0].set_ylabel('验证总损失');axes[0].set_title('基础训练：轮末验证损失')
axes[1].plot(x,[r['full_acc_epoch']*100 for r in ep],'-o',color=GREEN,lw=2)
axes[1].set_ylabel('token 匹配率（%）');axes[1].set_title('基础训练：离线 token 匹配率')
axes[2].plot(x,[r['unary_loss_epoch'] for r in ep],'-o',color=BLUE,lw=2,label='主干预测损失')
axes[2].plot(x,[r['selector_loss_epoch'] for r in ep],'-s',color=ORANGE,lw=2,label='候选选择损失（未乘权重）')
axes[2].set_ylabel('各损失分量');axes[2].set_title('基础训练：两个损失分量');axes[2].legend(frameon=False,fontsize=9)
for ax in axes:
    ax.set_xticks(x);ax.set_xlabel('完成训练轮数');grid(ax)
finish(fig,'02_training_validation','每个点代表一轮结束后的验证结果。token 匹配率反映预测情况，不是题目答对率。')

# 3. 697 actual optimizer updates, both ranks / all accumulation microbatches.
curve=D['targeted_training']['overlap']['curve']; x=np.array([r['update'] for r in curve])
fig,axes=plt.subplots(2,2,figsize=(12,7.1),layout='constrained')
metrics=[('loss','组合训练损失',BLUE),('selector_loss','候选概率分布匹配损失',ORANGE),('full_acc','训练 token 匹配率',GREEN),('native_tf_unfiltered_overlap','候选概率分布一致度',BLUE)]
for ax,(key,title,color) in zip(axes.flat,metrics):
    y=np.array([r[key] for r in curve]); factor=100 if key=='full_acc' else 1
    ax.plot(x,y*factor,color=color,alpha=.2,lw=.7,label='每步记录')
    smooth=np.array([y[max(0,i-10):min(len(y),i+11)].mean() for i in range(len(y))])
    ax.plot(x,smooth*factor,color=color,lw=2,label='平滑趋势（21 步平均）')
    ax.set_title(title);ax.set_xlabel('参数更新次数');grid(ax)
    if key=='full_acc':ax.set_ylabel('%')
axes[0,0].legend(frameon=False,fontsize=9)
finish(fig,'03_distribution_matching_training','浅色线为每步记录，深色线为 21 步移动平均。分布一致度越高，表示候选概率越接近教师。')

# 4. Frozen checkpoint, all benchmarks, no per-column best-checkpoint selection.
fig,axes=plt.subplots(1,2,figsize=(13,5.1),layout='constrained')
x=np.arange(5);width=.25
for ax,mode in zip(axes,['t0','t1']):
    for i,(model,label,color) in enumerate([('dspark_historical','DSpark（历史参考）',GRAY),('official','官方 DFlash2',BLUE),('ours','本文训练模型',ORANGE)]):
        y=[D['evaluation'][mode][model][b]['request_mean_tau'] for b in BENCH]
        bars=ax.bar(x+(i-1)*width,y,width*.95,color=color,label=label,hatch='//' if i==0 else None,zorder=3)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3,rotation=90)
    ax.set_xticks(x,LABELS,rotation=18);ax.set_ylim(0,7.15);ax.set_ylabel('逐请求平均接受长度');ax.set_title(('贪心解码（T=0）' if mode=='t0' else '采样解码（T=1）')+'：每个基准 32 道题');grid(ax)
axes[0].legend(frameon=False,fontsize=9,loc='upper right')
finish(fig,'04_five_benchmark_tau','官方与本文最终模型采用统一测试设置；DSpark 为此前对相同题目的测试结果。')

# 5. Actual conditional acceptance, aggregated by reached positions, not TF accuracy.
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
for ax,mode in zip(axes,['t0','t1']):
    for model,label,color,marker in [('official','官方 DFlash2',BLUE,'o'),('ours','本文训练模型',ORANGE,'s')]:
        y=D['evaluation'][mode][model]['overall']['conditional_accept']
        ax.plot(np.arange(1,8),np.array(y)*100,'-'+marker,lw=2,color=color,label=label)
    ax.set_xticks(np.arange(1,8));ax.set_xlabel('草稿位置（1–7）');ax.set_ylabel('条件接受率（%）');ax.set_ylim(72,90)
    ax.set_title(('贪心解码' if mode=='t0' else '采样解码')+'：五基准合计条件接受率');grid(ax);ax.legend(frameon=False,fontsize=10)
finish(fig,'05_conditional_acceptance','只有前面的 token 均被接受，才统计当前位置是否通过；越靠后，参与统计的片段越少。')

# 6. Same-wave C1 measurements, no obsolete C4 or historical DSpark timing mixed in.
fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
for ax,mode in zip(axes,['t0','t1']):
    for i,(model,label,color) in enumerate([('official','官方 DFlash2',BLUE),('ours','本文训练模型',ORANGE)]):
        y=[D['evaluation'][mode][model][b]['generation_tokens_per_request_second'] for b in BENCH]
        bars=ax.bar(np.arange(5)+(i-.5)*.36,y,.34,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax.set_xticks(np.arange(5),LABELS,rotation=18);ax.set_ylim(0,290);ax.set_ylabel('生成速度（token/s）');ax.set_title(('贪心解码' if mode=='t0' else '采样解码')+'：实测生成速度');grid(ax)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False,fontsize=10)
finish(fig,'06_c1_throughput','生成速度 = 总输出 token 数 / 总请求耗时。每个实例一次处理一个请求。')

# 7. A full first-epoch scratch comparison, not the final-model benchmark.
A=json.loads((HERE/'data/first_epoch_ablations.json').read_text(encoding='utf-8'))
first=A['random_initialization_first_epoch']
fig,axes=plt.subplots(1,2,figsize=(13.2,5),layout='constrained')
metrics=[('full_acc_epoch','整体 token 匹配率'),('teacher_forced_selector_acc_epoch','候选选择匹配率\n（给定正确前文）'),('unary_candidate_target_mass_at_16_epoch','16 个候选覆盖\n的教师概率')]
for i,(key,label) in enumerate(metrics):
    y=2-i
    vals=[first[arm]['validation'][key]*100 for arm in ['alpha_0_1','alpha_1_0']]
    axes[0].plot(vals,[y,y],color='#cbd5e1',lw=2,zorder=1)
    for value,color,offset,marker in zip(vals,[BLUE,ORANGE],[11,-17],['o','s']):
        axes[0].scatter(value,y,s=65,color=color,marker=marker,zorder=3)
        axes[0].annotate(f'{value:.2f}%',(value,y),xytext=(0,offset),textcoords='offset points',ha='center',color=color,fontsize=10)
axes[0].set_yticks([2,1,0],[label for key,label in metrics]);axes[0].set_ylim(-.55,2.55)
axes[0].set_xlim(40,95);axes[0].set_xlabel('验证指标（%）');axes[0].set_title('从头训练第一轮：离线指标')
axes[0].scatter([],[],color=BLUE,s=45,marker='o',label='选择器损失权重 0.1')
axes[0].scatter([],[],color=ORANGE,s=45,marker='s',label='选择器损失权重 1.0')
axes[0].legend(frameon=False,fontsize=9,loc='lower right')
axes[0].grid(axis='x',alpha=.16);axes[0].set_axisbelow(True)
comp=[first['comparison'][b] for b in BENCH+['overall']]
delta=np.array([r['delta'] for r in comp]);ci=np.array([r['paired_prompt_ci95'] for r in comp])
axes[1].bar(np.arange(6),delta,width=.58,color=[GREEN if v>=0 else ORANGE for v in delta],zorder=3)
axes[1].errorbar(np.arange(6),delta,yerr=np.vstack([delta-ci[:,0],ci[:,1]-delta]),fmt='none',ecolor='#475569',capsize=4,lw=1.3,zorder=4)
axes[1].axhline(0,color='#475569',lw=1)
axes[1].set_xticks(np.arange(6),LABELS+['五项均值'],rotation=18)
axes[1].set_ylabel('接受长度变化（权重 1.0 − 0.1）')
axes[1].set_ylim(float(ci[:,0].min())-.04,float(ci[:,1].max())+.04)
axes[1].set_title('实际贪心生成：变化接近零');grid(axes[1])
finish(fig,'07_first_epoch_selector_ablation','两组均取第一轮结束的模型。每个基准 32 道题；误差线按题目配对重采样，不包含训练随机种子变化。')
print('Rendered',len(list(FIG.glob('*.png'))),'PNG figures and matching SVG files.')
