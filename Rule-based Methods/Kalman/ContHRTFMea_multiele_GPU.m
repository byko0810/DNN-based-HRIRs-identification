clear
close all
clc

addpath('utils')

% True HRIR
load('data/FABIAN_multiele_180_12ch_gpu.mat');
% observation signal
load('data/obssig_multiele_180_12ch_gpu.mat');
% Excitation signal
load('data/PESQ_multiele_180_12ch_gpu.mat');

% gpu selection
gpuDevice(3)

ps = gpuArray(PS);

% Kalman filter parameters
est_length = 192;
ns = size(hrir,2);

% lim_length = 8000;
lim_length = 400;
% hop_size = lim_length-2*ns*est_length;
hop_size = lim_length-300;
Total_sample = size(hrir,3);
Kalman_hrir_p = zeros(size(hrir), 'gpuArray');
% KalmanSL_hrir_p = zeros(size(hrir), 'gpuArray');

sigma_sqr = 0.01;
max_iter = 1;
A = eye(est_length*ns);
gamma = 10e-7*eye(est_length*ns, 'gpuArray');
initx = zeros(est_length*ns,1, 'gpuArray');
initV = eye(est_length*ns, 'gpuArray');

i = 0;
while 1
    if lim_length+i*hop_size < Total_sample
        hrir_temp = hrir(:,:,1+i*hop_size:lim_length+i*hop_size);
        y_temp = y(1+i*hop_size:lim_length+i*hop_size);
        ps_temp = ps(:,1+i*hop_size:lim_length+est_length-1+i*hop_size);
        
        x = hrir_temp;
        y_temp = gpuArray(y_temp');
        
        % observation matrix
        C = zeros(size(y_temp,2),est_length*ns, 'gpuArray');
        for j = 1 : size(y_temp,2)
            C(j,:) = reshape(flip(ps_temp(:, j : est_length + j-1),2)',1,[]);
        end
        
        [xfilt, ~, ~, ~] = kalman_filter_GPU(y_temp, A, C, gamma, sigma_sqr, initx, initV);
        xfilt = reshape(xfilt, [], ns, size(xfilt,2));
        xfilt = cat(1,xfilt,zeros(size(x,1)-size(xfilt,1), size(xfilt,2), size(xfilt,3)));

        % [F, H, Q, R, initx2, initV2, ~] = learn_kalman_GPU(y_temp, A, C, gamma, sigma_sqr, initx, initV, max_iter);
        % [xsmoothL, ~] = kalman_smoother_GPU(y_temp, F, H, Q, R, initx2, initV2);
        % [xsmoothL, ~] = kalman_smoother_GPU(y_temp, A, C, gamma, sigma_sqr, initx, initV);
        % xsmoothL = reshape(xsmoothL, [], ns, size(xsmoothL,2));
        % xsmoothL = cat(1,xsmoothL,zeros(size(x,1)-size(xsmoothL,1), size(xsmoothL,2), size(xsmoothL,3)));

        if i == 0
            Kalman_hrir_p(:,:,1:lim_length) = xfilt;
            % KalmanSL_hrir_p(:,:,1:lim_length) = xsmoothL;
        else
            % Kalman_hrir_p(:,:,1+i*hop_size+2*ns*est_length:lim_length+i*hop_size) = xfilt(:,:,1+2*ns*est_length:lim_length);
            Kalman_hrir_p(:,:,1+i*hop_size+300:lim_length+i*hop_size) = xfilt(:,:,1+300:lim_length);
            % KalmanSL_hrir_p(:,:,1+i*hop_size+2*ns*est_length:lim_length+i*hop_size) = xsmoothL(:,:,1+2*ns*est_length:lim_length);
        end

    elseif lim_length+i*hop_size >= Total_sample
        hrir_temp = hrir(:,:,1+i*hop_size:end);
        y_temp = y(1+i*hop_size:end);
        ps_temp = ps(:,1+i*hop_size:end);
        
        x = hrir_temp;
        y_temp = gpuArray(y_temp');
        
        % observation matrix
        C = zeros(size(y_temp,2),est_length*ns, 'gpuArray');
        for j = 1 : size(y_temp,2)
            C(j,:) = reshape(flip(ps_temp(:, j : est_length + j-1),2)',1,[]);
        end
        
        [xfilt, ~, ~, ~] = kalman_filter_GPU(y_temp, A, C, gamma, sigma_sqr, initx, initV);
        xfilt = reshape(xfilt, [], ns, size(xfilt,2));
        xfilt = cat(1,xfilt,zeros(size(x,1)-size(xfilt,1), size(xfilt,2), size(xfilt,3)));

        % [F, H, Q, R, initx2, initV2, ~] = learn_kalman_GPU(y_temp, A, C, gamma, sigma_sqr, initx, initV, max_iter);
        % [xsmoothL, ~] = kalman_smoother_GPU(y_temp, F, H, Q, R, initx2, initV2);
        % [xsmoothL, ~] = kalman_smoother_GPU(y_temp, A, C, gamma, sigma_sqr, initx, initV);
        % xsmoothL = reshape(xsmoothL, [], ns, size(xsmoothL,2));
        % xsmoothL = cat(1,xsmoothL,zeros(size(x,1)-size(xsmoothL,1), size(xsmoothL,2), size(xsmoothL,3)));

        % Kalman_hrir_p(:,:,1+i*hop_size+2*ns*est_length:end) = xfilt(:,:,1+2*ns*est_length:end);
        Kalman_hrir_p(:,:,1+i*hop_size+300:end) = xfilt(:,:,1+300:end);
        % KalmanSL_hrir_p(:,:,1+i*hop_size+2*ns*est_length:end) = xsmoothL(:,:,1+2*ns*est_length:end);
        
        Kalman_hrir_p = gather(Kalman_hrir_p);
        save('save/Kalman_hrir_p_180_12ch.mat', 'Kalman_hrir_p');
        % KalmanSL_hrir_p = gather(KalmanSL_hrir_p);
        % save('save/KalmanSL_hrir_p_180_12ch.mat', 'KalmanSL_hrir_p');
        break
    end
    i = i + 1;
end
fprintf('Finish!!')