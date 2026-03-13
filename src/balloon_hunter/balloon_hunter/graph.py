#!/usr/bin/env python3
"""
DKF vs EKF Comparison Analysis (Fixed Version)
==============================================
Loads two CSV logs (one DKF, one EKF) and generates comparison graphs.
Fixed: AttributeError for rolling on numpy and ValueError for indexing.
"""

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 그래프 폰트 및 해상도 설정
matplotlib.rcParams['figure.dpi'] = 130
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def load(path):
    df = pd.read_csv(path)
    # 모든 데이터를 숫자로 변환 (에러 발생 시 NaN 처리)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def find_logs():
    """최신 DKF 및 EKF 로그 파일을 자동으로 찾습니다."""
    d = os.path.expanduser('~/dkf_logs')
    dkf = sorted(glob.glob(os.path.join(d, 'log_*_DKF.csv')))
    ekf = sorted(glob.glob(os.path.join(d, 'log_*_EKF.csv')))
    if not dkf: print('No DKF log found!'); sys.exit(1)
    if not ekf: print('No EKF log found!'); sys.exit(1)
    return dkf[-1], ekf[-1]

def active_region(df):
    """타겟이 화면에 보이는 구간만 추출합니다."""
    mask = df['u_gt'].notna() & df['v_gt'].notna()
    if mask.sum() == 0: return df
    t0 = df.loc[mask, 'timestamp_s'].iloc[0] - 0.5
    t1 = df.loc[mask, 'timestamp_s'].iloc[-1] + 0.5
    return df[(df['timestamp_s']>=t0)&(df['timestamp_s']<=t1)].copy()

def graph1_coordinates(dkf, ekf, out):
    """이미지 좌표(u) 시계열 비교 그래프"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # DKF subplot
    ax = axes[0]
    ax.set_title('DKF Experiment: Image Coordinate (u-axis)', fontsize=13)
    ax.scatter(dkf['timestamp_s'].values, dkf['u_yolo'].values, s=6, alpha=0.3, color='gray', label='YOLO raw')
    ax.plot(dkf['timestamp_s'].values, dkf['u_gt'].values, linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
    ax.plot(dkf['timestamp_s'].values, dkf['u_est'].values, linewidth=1.2, color='#2980b9', label='DKF estimate')
    ax.axhline(y=424, color='orange', linestyle=':', alpha=0.5, label='Image center')
    ax.set_ylabel('u (pixels)'); ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.3)

    # EKF subplot
    ax = axes[1]
    ax.set_title('EKF Experiment: Image Coordinate (u-axis)', fontsize=13)
    ax.scatter(ekf['timestamp_s'].values, ekf['u_yolo'].values, s=6, alpha=0.3, color='gray', label='YOLO raw')
    ax.plot(ekf['timestamp_s'].values, ekf['u_gt'].values, linewidth=1.5, color='#2ecc71', linestyle='--', label='Ground Truth')
    ax.plot(ekf['timestamp_s'].values, ekf['u_est'].values, linewidth=1.2, color='#e74c3c', label='EKF estimate')
    ax.axhline(y=424, color='orange', linestyle=':', alpha=0.5, label='Image center')
    ax.set_ylabel('u (pixels)'); ax.set_xlabel('Time (s)'); ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  -> {out}')

def graph2_error(dkf, ekf, out):
    """추정 오차(픽셀 거리) 비교 그래프"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    ax = axes[0]
    ax.set_title('Estimation Error: Distance to Ground Truth (Smoothed)', fontsize=13)
    w = 25 # 0.5초 이동 평균 (50Hz 기준)
    
    # 주의: .rolling()은 Pandas 객체에서 호출하고 마지막에 .values를 붙여야 함
    dkf_smooth = dkf['err_est_px'].rolling(w, min_periods=1).mean().values
    ekf_smooth = ekf['err_est_px'].rolling(w, min_periods=1).mean().values
    
    ax.plot(dkf['timestamp_s'].values, dkf_smooth, linewidth=2, color='#2980b9', label='DKF (0.5s avg)')
    ax.plot(ekf['timestamp_s'].values, ekf_smooth, linewidth=2, color='#e74c3c', label='EKF (0.5s avg)')
    ax.set_ylabel('Error (pixels)'); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.set_title('Raw Estimation Error', fontsize=13)
    ax.plot(dkf['timestamp_s'].values, dkf['err_est_px'].values, linewidth=0.8, alpha=0.5, color='#2980b9', label='DKF')
    ax.plot(ekf['timestamp_s'].values, ekf['err_est_px'].values, linewidth=0.8, alpha=0.5, color='#e74c3c', label='EKF')
    ax.set_ylabel('Error (pixels)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  -> {out}')

def graph3_control(dkf, ekf, out):
    """제어 성능(중앙 오차 수렴 및 요 각도) 그래프"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # 이미지 중앙 오차 (ex)
    ax = axes[0]
    ax.set_title('Control Performance: ex (horizontal distance from image center)', fontsize=13)
    ax.plot(dkf['timestamp_s'].values, dkf['ex_from_center'].values, linewidth=1.2, color='#2980b9', label='DKF', alpha=0.8)
    ax.plot(ekf['timestamp_s'].values, ekf['ex_from_center'].values, linewidth=1.2, color='#e74c3c', label='EKF', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax.axhspan(-20, 20, alpha=0.1, color='green', label='±20px Tolerance')
    ax.set_ylabel('ex (pixels)'); ax.legend(loc='upper right', fontsize=10); ax.grid(True, alpha=0.3)

    # 드론 Yaw
    ax = axes[1]
    ax.set_title('Drone Yaw Angle', fontsize=13)
    ax.plot(dkf['timestamp_s'].values, np.degrees(dkf['drone_yaw'].values), linewidth=1.5, color='#2980b9', label='DKF exp')
    ax.plot(ekf['timestamp_s'].values, np.degrees(ekf['drone_yaw'].values), linewidth=1.5, color='#e74c3c', label='EKF exp', linestyle='--')
    ax.set_ylabel('Yaw (deg)'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # 드론 속도
    ax = axes[2]
    ax.set_title('Drone Speed', fontsize=13)
    dkf_speed = np.sqrt(dkf['drone_vx']**2+dkf['drone_vy']**2+dkf['drone_vz']**2).values
    ekf_speed = np.sqrt(ekf['drone_vx']**2+ekf['drone_vy']**2+ekf['drone_vz']**2).values
    ax.plot(dkf['timestamp_s'].values, dkf_speed, linewidth=1.5, color='#2980b9', label='DKF exp')
    ax.plot(ekf['timestamp_s'].values, ekf_speed, linewidth=1.5, color='#e74c3c', label='EKF exp', linestyle='--')
    ax.set_ylabel('Speed (m/s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  -> {out}')

def graph4_summary(dkf, ekf, out):
    """요약 통계 바 차트"""
    def get_stats(df, name):
        v = df['err_est_px'].dropna()
        ex = df['ex_from_center'].dropna().abs()
        # 수렴 시간 계산: |ex| < 20px가 0.5초 동안 지속되는 첫 시점
        conv_mask = (ex < 20).rolling(25, min_periods=25).sum() == 25
        ci = conv_mask[conv_mask].index
        ct = df.loc[ci[0], 'timestamp_s'] - df['timestamp_s'].iloc[0] if len(ci)>0 else float('nan')
        return {'name': name, 'mean': v.mean(), 'median': v.median(), 
                'std': v.std(), 'max': v.max(), 'p90': v.quantile(0.9), 'conv_t': ct}

    sd = get_stats(dkf, 'DKF')
    se = get_stats(ekf, 'EKF')

    labels = ['Mean\nError', 'Median\nError', 'Std Dev', 'Max\nError', '90th\n%ile', 'Converge\nTime (s)']
    dv = [sd['mean'], sd['median'], sd['std'], sd['max'], sd['p90'], sd['conv_t']]
    ev = [se['mean'], se['median'], se['std'], se['max'], se['p90'], se['conv_t']]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels)); w = 0.35
    b1 = ax.bar(x-w/2, ev, w, label='EKF', color='#e74c3c', alpha=0.8)
    b2 = ax.bar(x+w/2, dv, w, label='DKF', color='#2980b9', alpha=0.8)

    # 막대 위에 수치 표시
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            if not np.isnan(h): ax.text(b.get_x()+b.get_width()/2, h+0.2, f'{h:.1f}', ha='center', fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=12)
    ax.set_title('DKF vs EKF: Performance Summary Statistics', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # 개선율 표시
    if se['mean'] > 0:
        imp = (se['mean']-sd['mean'])/se['mean']*100
        color = 'green' if imp > 0 else 'red'
        ax.text(0.02, 0.95, f'DKF Improvement: {imp:+.1f}% (Mean Error)',
                transform=ax.transAxes, fontsize=13, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  -> {out}')

def text_summary(dkf, ekf):
    """콘솔에 텍스트 요약 출력"""
    print('\n' + '='*60)
    print('        DKF vs EKF Performance Comparison Results')
    print('='*60)
    for name, df in [('DKF', dkf), ('EKF', ekf)]:
        v = df['err_est_px'].dropna()
        hz = df['yolo_hz'].iloc[-1] if 'yolo_hz' in df.columns else 0
        print(f'\n  [{name}]  ({len(df)} samples, YOLO {hz:.1f}Hz)')
        print(f'    Mean Error:     {v.mean():.2f} px')
        print(f'    90th Percentile: {v.quantile(0.9):.2f} px')
        print(f'    Max Error:      {v.max():.2f} px')
    
    md = dkf['err_est_px'].mean()
    me = ekf['err_est_px'].mean()
    if me > 0:
        print(f'\n  >>> Final Improvement: {(me-md)/me*100:+.1f}% <<<')
    print('='*60)

def main():
    # 인자 확인 또는 자동 찾기
    if len(sys.argv) >= 3:
        dkf_path, ekf_path = sys.argv[1], sys.argv[2]
    else:
        dkf_path, ekf_path = find_logs()
        print(f'Using DKF: {dkf_path}')
        print(f'Using EKF: {ekf_path}')

    # 데이터 로드 및 전처리
    dkf = active_region(load(dkf_path))
    ekf = active_region(load(ekf_path))
    
    out_dir = os.path.dirname(dkf_path)
    base = 'comparison'

    print('\nGenerating graphs...')
    graph1_coordinates(dkf, ekf, os.path.join(out_dir, f'{base}_1_coordinates.png'))
    graph2_error(dkf, ekf, os.path.join(out_dir, f'{base}_2_error.png'))
    graph3_control(dkf, ekf, os.path.join(out_dir, f'{base}_3_control.png'))
    graph4_summary(dkf, ekf, os.path.join(out_dir, f'{base}_4_summary.png'))
    
    text_summary(dkf, ekf)
    print(f'\nAll result images saved in: {out_dir}/')

if __name__ == '__main__':
    main()