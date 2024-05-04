import csv
import numpy as np 
import matplotlib.pyplot as plt
import serial
import time
import os


def ListRawData(port='COM5'): #아두이노에서 전압 데이터 받는 함수 
    ser = serial.Serial(port, 9600, timeout=1)
    time.sleep(2) # 아두이노 리셋 후 안정화를 위한 대기 시간
    heart_data_list = []
    start_time =0
    end_time = 0
    zero_count=0
    on_flag=0
    try:
        if ser.in_waiting > 0:
            if on_flag==0:
                if input("시작하려면 'q'를 입력하세요: ") == 'q':  # Start
                    start_time=time.time()
                    while True:
                        try:
                            line = ser.readline().decode('utf-8').rstrip()
                            print(line) # 읽은 데이터 출력
                            heart_data_list.append(line)
                            if line== '0':
                                zero_count += 1
                            else:
                                zero_count = 0
                            if zero_count == 250:
                                print('측정이 종료되었습니다')
                                end_time=time.time()
                                heart_data_list.append(end_time-start_time)
                                on_flag =1
                                break
                        except :
                            return(heart_data_list)
    finally:
        ser.close() # 시리얼 포트 닫기
    return(heart_data_list)

def SaveList(lst,name='name'): #리스트 csv로 저장하는 함수 (.csv붙여야함)
    """
    리스트를 csv 형태로 저장
    Args:
        lst (list): 리스트파일
        name ('string'): csv파일 제목
    """
    if os.path.exists(name):
        print(f"Error: 파일 '{name}'이(가) 이미 존재합니다. 저장을 중단합니다.")
        return  # 함수를 빠져나와 더 이상 진행하지 않음
    new_lst = [] 
    for i in lst:
        new_lst.append([i])
    with open(name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(new_lst)

def FilterLargeNum(raw_data): # 이상치 적당한 값으로 대체(인덱스수는 유지)
    """
    raw data를 입력받아 큰 숫자를 원래 값 범위로 치환해서 채워준 후 해당 갯수 표시 하고 수저오딘 리스트 반환 
    """
    count_error = 0
    raw_data_new = []
    for i in range(len(raw_data)):
        if raw_data[i] >1000:
            tmp = raw_data[i]
            print(tmp)
            count_error+=1
            if i == 0:
                tmp = raw_data[i+1]
                count_error -=1
            while tmp > 1000:
                tmp = tmp//10
            raw_data_new.append(tmp)  
        else : 
            raw_data_new.append(raw_data[i])
    print("sdljflskdjflsdkjflsjdfj count_error = "+str(count_error))
    return raw_data_new 

def LoadCsvToList(name,path ="data//"):#데이터 폴더에 저장되어있는 csv파일 리스트로 로드 
    """
    CSV 파일을 불러와서 리스트로 저장
    Args:
        name (str): csv파일 제목
    Returns:
        list: CSV 파일의 내용을 담은 리스트
    """
    data = []
    with open(path+name, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                data.append(int(row[0]))
            except:
                continue
    return data

def GenerateLabelList(raw_data,start_idx,end_idx=0,threshold = -120): # 미분값 하강 기울기 기준으로 펄스에서 내려가는 전압 포착
    """
    raw_data 특정 섹션 정답 라벨 리스트 반환하고 plot
    threshold : 인덱스간 차이 
    """
    label_lst = []
    t_idx = []  
    if end_idx == 0 :
        end_idx = start_idx + 1000
    for i in range(len(raw_data[start_idx:end_idx])):
        t_idx.append(start_idx+i)
    for i in range(len(raw_data[start_idx:end_idx])):
        if raw_data[start_idx+i]-raw_data[start_idx+i-1]<threshold and raw_data[start_idx+i+1]-raw_data[start_idx+i]>threshold: 
            label_lst.append(start_idx+i+1)
    return(label_lst)

def DeleteErrorLabelLst(label_lst,threshold = 1.3): # GenerateLabelList 함수에서 잘못 들어간 라벨들 제거 
    """
    라벨리스트 중 헐거운 임계값 설정으로 잘못 포함된 수치 제거 
    threshold : 정상적 차이보다 threshold 정도 만큼 더 작은 인덱스 차이 
    """
    filtered_label_lst= [label_lst[0]]
    eliminated_label_lst = []
    for i in range(2,len(label_lst)):
        if label_lst[i]-label_lst[i-1]<(label_lst[i-1]-label_lst[i-2])/threshold:
            eliminated_label_lst.append(label_lst[i-1])
            print(label_lst[i-1])
            continue
        else:
            filtered_label_lst.append(label_lst[i-1])
    filtered_label_lst.append(label_lst[-1])
    print(eliminated_label_lst)
    return filtered_label_lst

def ShowLabelList(raw_data, label_lst, start_idx, end_idx=0, scale=1):#라벨 인덱스와 함께 입력시 라벨 표시 (raw 데이터만 표기할 경우 label_lst = [])
    """
    라벨링된 리스트를 raw 그래프 위에 plot
    """
    if end_idx == 0:
        end_idx = start_idx + 1000

    # 인덱스 범위를 벗어나지 않는 레이블만 필터링
    filtered_labels = [label for label in label_lst if start_idx <= label < end_idx]

    t_idx = list(range(start_idx, end_idx))
    
    plt.figure(figsize=(15*scale, 7*scale))
    plt.grid()
    plt.plot(t_idx, raw_data[start_idx:end_idx], label="Raw Data")
    
    # 필터링된 레이블에 대해 점 찍기
    for label in filtered_labels:
        plt.scatter(label, raw_data[label], color='r', label="Label" if filtered_labels.index(label) == 0 else "")

    plt.legend()
    plt.show()

def LoadTextToList(file_path='heartDATA.txt'):
    heart_data = []
    with open(file_path, 'r') as file:
        content = file.read().replace('\n', '').split(',')
        heart_data = [int(num.strip()) for num in content if num.strip().isdigit()]           
    return heart_data

def PulseChunking(pulse_raw_data,num_chunk):
    """
    1차원 배열 펄스 raw 데이터를 num_chunk 갯수 만큼 묶기 
    pulse_raw_data : raw data       // dtype = list 
    num_chunk      : 몇 개씩 묶을지  // dtype = integer  
    normalize_coeff: raw_data를 0 ~ 1 사이 정규화 // dtype = float 
    """
    pulse_chunked_data =[]
    normalize_coeff =0.0014
    len_chunk = (len(pulse_raw_data)-num_chunk)//num_chunk
    for j in range(num_chunk):
        pulse_data_tmp = []
        for i in range(num_chunk):
            pulse_data_tmp.append(pulse_raw_data[j*len_chunk+i]*normalize_coeff) 
        pulse_chunked_data.append(pulse_data_tmp)
    return pulse_chunked_data

def ListToMovingWindow(inp_lst,len_chunk,step=10,start_idx = 0):
    """
    raw data를 입력 받아 각 스텝 별 결과를 리스트로 반환 
    inp_lst           : 1차원 배열   
    step              : 스텝 
    normalize_coeff   : raw_data를 0 ~ 1 사이 정규화 
    return            : chunk가 저장된 2차원 배열 
    """
    large_lst = []
    normalize_coeff =0.0014
    for i in range(len(inp_lst)):
        inp_lst[i]=inp_lst[i]*normalize_coeff
    for i in range(start_idx,(len(inp_lst)-len_chunk)//step):
        large_lst.append(inp_lst[i*step:i*step+len_chunk])
    return large_lst    

def PeakFilter(derivative_raw_data,section_range=10):
    """
    구간 최대소 차이를 해당 인덱스의 출력 값으로 가짐 
    derivative_raw_data : raw 데이터
    section_range       : 피크 감도 범위 
    """
    der2_raw_data = []
    for i in range(len(derivative_raw_data)-section_range):
        section_max = 0
        for j in range(section_range):
            if derivative_raw_data[i]-derivative_raw_data[i-j]>section_max:
                section_max = derivative_raw_data[i]-derivative_raw_data[i-j]    
        der2_raw_data.append(section_max)
    return(der2_raw_data)
    
def PeakNear(hr_data, label):
    """
    피크에 라벨링이 안되어 있는 부분을 피크로 맞춰주어 반환 
    """
    new_label = []
    for i in label:
        while hr_data[i]<hr_data[i-1]:
            i-=1
        while hr_data[i]<hr_data[i+1]:
            i+=1
        new_label.append(i)
    return new_label

def ShowLabelToBpm(label,start_idx=0, end_idx = 0, sampling_rate = 0.005):
    """
    라벨에 대한 bpm 그래프를 plot하고 bpm 리스트 반환
    sampling_rate  : 데이터 샘플링 레이트 
    """
    if end_idx ==0:
        end_idx = start_idx + 200
    bpm_lst = []
    for i in range(1,len(label)):
        time_interval = label[i]-label[i-1]
        bpm_lst.append(60//(sampling_rate*time_interval))
    plt.figure(figsize = (15,7))
    plt.plot(bpm_lst[start_idx:end_idx])
    return bpm_lst