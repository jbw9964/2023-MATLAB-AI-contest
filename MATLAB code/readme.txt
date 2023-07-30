1. 모든 .m 파일은 같은 directory에 있어야 합니다.
2. 현재 작업중인 directory도 동일하게 설정합니다.
3. Command Window 창에서 predict_model함수를 실행합니다.
4. predict_model('model이 들어있는 경로','test data가 들어있는 경로', n_fft, win_length)를 입력합니다. 파일의 h5 모델을 이용하려면 n_fft = 960, win_length = 960으로 설정합니다.
5. Check_Audio_data_functions_not_use_in_model 파일 안 함수들은 불러온 Audio data를 재생할 수 있는 함수 파일입니다.
