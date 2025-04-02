clear
close all
clc

% True HRIR
load('data/FABIAN_multiele_180_12ch_gpu.mat');
% observation signal
load('data/obssig_multiele_180_12ch_gpu.mat');
% Excitation signal
load('data/PESQ_multiele_180_12ch_gpu.mat');

% Kalman filter parameters
est_length = 192;
ns = size(hrir,2);
% fs = 44.1e3;
% fs = 24e3;
% angle = 180/10; % final angle (degree)
% rv = 45; % rotation speed (degree/sec)
total_time = angle / rv;
% sigma_sqr = 0.01;
stepsize = 0.5;

y = y';

ps = PS;

% white noise generation
% wn = normrnd(0,sqrt(sigma_sqr),[1,size(y,2)]);
% y = y+wn;

pred_H = zeros(est_length*ns, size(hrir,3));
loss_M = [];
total_loss = 0;
pred_h = zeros(est_length*ns, 1);
for i = 1 : size(pred_H, 2)
    pred_H(:,i) = pred_h; 
    x = reshape(flip(ps(:, i : est_length + i-1),2)',1,[]);
    e = (y(i) - pred_h'*x');
    pred_h = pred_h + stepsize * e / norm(x)^2 * x';        
    loss = e^2;
    total_loss = total_loss + loss;
    loss_M = horzcat(loss_M, loss);
end

pred_H = reshape(pred_H, [], ns, size(pred_H,2));
pred_H = cat(1,pred_H,zeros(size(hrir,1)-size(pred_H,1), size(pred_H,2), size(pred_H,3)));
d = squeeze(10*log10(sum((hrir - pred_H).^2)./sum(hrir.^2)));
D = sum(d, 'all')/(size(y,2)*ns);

save('save/NLMS_hrir_p_180_12ch.mat', 'pred_H','-v7.3')

time_axis = 0 : 1/fs : (length(hrir(:,1,1))-1)/fs;
ang_axis = 0 : (1/fs)*rv : (total_time-1/fs)*rv;


cmax = 1.5;
cmin = -1.755;
dmax = 10e-1;
dmin = 10e-4;

figure(1)
im = imagesc(time_axis*1000,ang_axis,squeeze(hrir(:,4,:)).'); axis xy
im.AlphaData = .8;
colormap jet
colorbar
clim([cmin cmax]);
grid on
set(gcf,'position',[50 100 650 400]);
set(gca,'FontName','times','FontSize',18);
xlabel('Time, ms','FontSize',18)
ylabel('Azimuth angle, \circ','FontSize',18)
saveas(gcf,'plot/True_HRIR0ele.emf')
saveas(gcf,'plot/True_HRIR0ele.fig')

figure(2)
im = imagesc(time_axis*1000,ang_axis,squeeze(pred_H(:,4,:)).'); axis xy
im.AlphaData = .8;
colormap jet
colorbar
clim([cmin cmax]);
grid on
set(gcf,'position',[50 100 650 400]);
set(gca,'FontName','times','FontSize',18);
xlabel('Time, ms','FontSize',18)
ylabel('Azimuth angle, \circ','FontSize',18)
saveas(gcf,'plot/NLMS_HRIR0ele.emf')
saveas(gcf,'plot/NLMS_HRIR0ele.fig')

figure(3)
im = imagesc(time_axis*1000,ang_axis,squeeze(abs(hrir(:,4,:)-pred_H(:,4,:))).'); axis xy
im.AlphaData = .8;
colormap jet
colorbar
grid on
set(gcf,'position',[50 100 650 400]);
set(gca,'ColorScale','log')
clim([dmin dmax]);
set(gca,'FontName','times','FontSize',18);
xlabel('Time, ms','FontSize',18)
ylabel('Azimuth angle, \circ','FontSize',18)
saveas(gcf,'plot/NLMS-ture_HRIR0ele.emf')
saveas(gcf,'plot/NLMS-ture_HRIR0ele.fig')



N = size(hrir,1);
freq_axis = (0 : N/2)*fs/N;

HRTF_t = abs(fft(hrir, [], 1));
HRTF_t = HRTF_t(1:N/2+1,:,:);
HRTF_t(2:end-1,:,:) = 2*HRTF_t(2:end-1,:,:);
HRTF_t = 20*log10(HRTF_t);


NLMS_HRTF = abs(fft(pred_H, [], 1));
NLMS_HRTF = NLMS_HRTF(1:N/2+1,:,:);
NLMS_HRTF(2:end-1,:,:) = 2*NLMS_HRTF(2:end-1,:,:);
NLMS_HRTF = 20*log10(NLMS_HRTF);


% analysis of objective performance
freq_ind = find(freq_axis<=20e3 & freq_axis>=50);
ang_ind = 300;
lsd = sum((HRTF_t(freq_ind,:,ang_ind:end) - NLMS_HRTF(freq_ind,:,ang_ind:end)).^2)/size(HRTF_t,1);
LSD = sum(lsd, 'all')/(size(HRTF_t,3)*ns);


cmax = 30;
cmin = -10;
dmax = 10;
dmin = 10e-2;
flim = [0 20];

figure(4)
im = imagesc(freq_axis/1000,ang_axis,squeeze(HRTF_t(:,4,:)).'); axis xy
im.AlphaData = .8;
colormap jet
colorbar
hc = colorbar;
title(hc, 'dB')
clim([cmin cmax]);
xlim(flim)
grid on
set(gcf,'position',[50 100 650 400]);
set(gca,'FontName','times','FontSize',18);
xlabel('Frequency, kHz','FontSize',18)
ylabel('Azimuth angle, \circ','FontSize',18)
saveas(gcf,'plot/True_HRTF0ele.emf')
saveas(gcf,'plot/True_HRTF0ele.fig')


figure(5)
im = imagesc(freq_axis/1000,ang_axis,squeeze(NLMS_HRTF(:,4,:)).'); axis xy
im.AlphaData = .8;
colormap jet
colorbar
hc = colorbar;
title(hc, 'dB')
clim([cmin cmax]);
xlim(flim)
grid on
set(gcf,'position',[50 100 650 400]);
set(gca,'FontName','times','FontSize',18);
xlabel('Frequency, kHz','FontSize',18)
ylabel('Azimuth angle, \circ','FontSize',18)
saveas(gcf,'plot/NLMS_HRTF0ele.emf')
saveas(gcf,'plot/NLMS_HRTF0ele.fig')


figure(6)
im = imagesc(freq_axis/1000,ang_axis,squeeze(abs(NLMS_HRTF(:,4,:)-HRTF_t(:,4,:))).'); axis xy
im.AlphaData = .8;
colormap jet
colorbar
hc = colorbar;
title(hc, 'dB')
grid on
set(gcf,'position',[50 100 650 400]);
set(gca,'ColorScale','log')
clim([dmin dmax]);
xlim(flim)
set(gca,'FontName','times','FontSize',18);
xlabel('Frequency, kHz','FontSize',18)
ylabel('Azimuth angle, \circ','FontSize',18)
saveas(gcf,'plot/NLMS-ture_HRTF0ele.emf')
saveas(gcf,'plot/NLMS-ture_HRTF0ele.fig')
