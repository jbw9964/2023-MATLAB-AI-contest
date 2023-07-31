function audioPlayerGUI(audio_data, sample_rate, name)
    % GUI Figure 생성
    fig = uifigure('Name', 'Audio Player GUI', 'Position', [100, 100, 400, 200]);

    % AudioPlayer 객체 생성
    player = audioplayer(audio_data, sample_rate);
    
    % 버튼 위에 텍스트 추가
    uitextarea(fig, 'Value', name , 'Position', [150, 40, 100, 40]);

    % Play 버튼 생성
    uibutton(fig, 'Text', 'Play', 'Position', [50, 100, 100, 50], 'ButtonPushedFcn', @(src, event) playAudio(player));

    % Stop 버튼 생성
    uibutton(fig, 'Text', 'Stop', 'Position', [250, 100, 100, 50], 'ButtonPushedFcn', @(src, event) stopAudio(player));
end



