
from __future__ import print_function
import numpy as np
import scipy.io as sio
import GPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pykrige.ok import OrdinaryKriging

#############################################################################################
# 엑셀 파일 로드
file_path = 
df = pd.read_excel(file_path)

# '측정시간' 열을 datetime 형식으로 변환
df['측정시간'] = pd.to_datetime(df['측정시간'])

# 센서 이름 매핑
sensor_names = {
    1: 'S-V1', 3: 'S-V2', 9: 'S-V3', 11: 'S-V4', 12: 'S-VC1', 13: 'S-VC2', 14: 'S-VC3', 15: 'S-VC4', 
    16: 'S-V5', 17: 'S-V6', 19: 'S-H2', 20: 'S-V7', 21: 'S-H3', 24: 'S-H1', 25: 'S-V8'
}


# Specify the times for GPR analysis
specified_times = ['2023-10-05 10:09:58', '2023-10-05 10:19:58', '2023-10-05 15:35:10']
specified_times = pd.to_datetime(specified_times)

# Print specified times
print("Specified times for analysis:")
print(specified_times)

# Extract temperature data for specified times
temperature_data = df[df['측정시간'].isin(specified_times)].drop(columns=['측정시간'])

# Print extracted temperature data for specified times
print("\nExtracted Temperature Data for Specified Times:")
print(temperature_data)

#############################################################################################
# Prepare data for GPR
#############################################################################################

# 기존 데이터 포인트
x1 = np.array([-291, -281, -269, -269, -54, -55, -55, -55, -55, -56, -56, 62, 107, 158, 158])
y1 = np.array([200, 204, 206, 223, 171, 160, 154, 148, 141, 88, 87, 86, 85, 84, 92])
xy = np.vstack([x1, y1]).T

# 파이프라인을 따라 길이(s) 계산
s = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2)
s = np.insert(s, 0, 0)  # 시작점에서 길이 0 추가
s = np.cumsum(s)  # 누적합으로 전체 길이 계산

#############################################################################################

# 예측에 사용할 그리드 포인트 생성
s_new = np.linspace(s.min(), s.max(), 1000)
y_new = np.linspace(y1.min(), y1.max(), 1000)
S, Y = np.meshgrid(s_new, y_new)
new_data = np.vstack([S.ravel(), Y.ravel()]).T


# GP 모델 학습 및 예측
# 커널의 lengthscale을 조정하여 모델의 유연성 변경
kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=5.0) # lengthscale 값을 조정

folder1 = 

# 예측에 사용할 그리드 포인트 생성 (2차원 공간)
X, Y = np.meshgrid(np.linspace(min(x1), max(x1), 100), 
                   np.linspace(min(y1), max(y1), 100))
new_data = np.vstack([X.ravel(), Y.ravel()]).T


# GPR 예측 그래프 그리기
for i in range(len(temperature_data)):
    data = temperature_data.iloc[i].values
    m = GPy.models.GPRegression(xy, data[:, None], kernel)
    m.optimize()
    mean, variance = m.predict(new_data)

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, mean.reshape(X.shape), levels=np.linspace(data.min(), data.max(), 100), cmap=plt.cm.jet)
    plt.scatter(xy[:, 0], xy[:, 1], color='gray', s=30, zorder=10)
    
    plt.colorbar(contour, label='Temperature (°C)').ax.tick_params(labelsize=16)
    plt.xlabel('X Coordinate', fontsize=18, fontweight='bold')
    plt.ylabel('Y Coordinate', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder1}/gpr_plot_{i}.png")
    plt.show()
