clear
close all
clc


%% parameters of continuous measurement %%
var_wn = 0.01; % variance of white noise
est_length = 192; % number of samples for estimated HRIR filter


%% gpu setting
gpuDevice(1)

%% load sphere HRTF %%
load('data/FABIAN_multiele_180_12ch_gpu.mat')
total_sample = size(hrir,3); 
total_time = total_sample/fs;
ns = size(hrir, 2); % number of sound sources


%% removing NaN by peripheral value
% for i = 1 : size(hrir, 2)
%     temp = squeeze(hrir(:, i, :));
%     [row, col] = find(isnan(temp));
%     col = unique(col);
%     for j = 1 : length(col)
%         if j ~=length(col)
%             if (col(j+1)-col(j))~=1 
%                 hrir(:,i,col(j)) = 0.5*(hrir(:,i,col(j)-1) + hrir(:,i,col(j)+1));
%             else 
%                 if (col(j+2)-col(j))==2
%                     error("It's NaN value in a row")
%                 end
%                 hrir(:,i,col(j)) = 0.5*(hrir(:,i,col(j)-1) + hrir(:,i,col(j)+2));
%             end
%         else
%             hrir(:,i,col(j)) = 0.5*(hrir(:,i,col(j)-1) + hrir(:,i,col(j)+1));
%         end
%     end
% end
% save('FABIAN_multiele_180_2ch_gpu.mat','hrir', 'fs', 'rv', 'angle','-v7.3')

%% initial parameters
% time_axis = 0 : 1/fs : (length(hrir(:,1))-1)/fs;
% ang_axis = 0 : (1/fs)*rv : (total_time-1/fs)*rv;
% figure
% plot(time_axis, hrir(:,1,60))
% figure
% im = imagesc(time_axis*1000,ang_axis,squeeze(hrir(:,1,:)).'); axis xy
% im.AlphaData = .8;
% colormap jet
% colorbar
% hc=colorbar;
% grid on
% set(gcf,'position',[50 100 650 400]);
% set(gca,'FontName','times','FontSize',18);
% xlabel('Time, ms','FontSize',18)
% ylabel('Incidence angle, \circ','FontSize',18)


%% Generation of PESQ (Perfect Sweep) and white gaussian noise signal %%
% Perfect sweep generation
ps = perfectsweep(fs, est_length*ns);
ps = ps * 100;

ps_c = zeros(ns, size(ps,2), 'gpuArray');
for i = 1 : ns
    ps_c(i,:) = circshift(ps, (i-1)*est_length);
end

% ps_f = fft(ps/length(ps));
% ps_f = ps_f(1:length(ps)/2+1);
% f = 0 : fs/2/(length(ps_f)-1) : fs/2;
% logps_f = 20*log10(abs(ps_f));
% figure
% plot(f/1000,logps_f,'LineWidth',1.4)
% ylim([0 80])
% xlim([0 12])
% set(gca,'FontName','times','FontSize',18);
% xlabel('Frequency, kHz','FontSize',18)
% ylabel('Magnitude, dB', 'FontSize',18)
% grid on


% % perfectsweep check
% figure
% t = 0 : 1/fs : (length(ps)-1)/fs;
% plot(t, ps,'LineWidth',1.4)
% set(gca,'FontName','times','FontSize',18);
% xlabel('Time, s','FontSize',18)
% ylabel('Amplitude', 'FontSize',18)
% xlim([0 0.05])
% audiowrite('perfectsweepsig.wav',ps,fs)

% [acf,lags] = autocorr(ps,est_length-1);
% figure
% plot(lags,acf,'LineWidth',1.4)
% set(gca,'FontName','times','FontSize',18);
% xlabel('Lag','FontSize',18)
% ylabel('Autocorrelation', 'FontSize',18)
% grid on

PS = zeros(ns, total_sample, 'gpuArray');
Quo = fix(total_sample/length(ps));
Rem = rem(total_sample,length(ps));
for i = 1 : ns
    if rem(total_sample,length(ps)) == 0
        PS(i,:) = repmat(ps_c(i,:),1,total_sample/length(ps));
    else        
        PS(i,1:Quo*length(ps)) = repmat(ps_c(i,:),1,Quo);
        PS(i,Quo*length(ps)+1:end) = ps_c(i,1:Rem);
    end
end

%% simulation of continuous HRTF measurement %%
% generation of observation signal
y = zeros(total_sample,1, 'gpuArray');
for i = 1 : total_sample
    temp = circshift([zeros(total_sample,ns);squeeze(hrir(:,:,i))].',-i,2);
    y(i) = sum(dot(flip(PS,2),temp(:,1:total_sample)));
end

% measurement noise
sigma_sqr = 0.01;
wn = gpuArray(normrnd(0,sqrt(sigma_sqr),[1,size(y,2)]));
y = gather(y+wn);

saveobssig = strcat('data/obssig_multiele_', num2str(angle), '_', num2str(ns), 'ch_gpu.mat');
save(saveobssig,'y')
% audiowrite('data/obssig_multiele_FA_180_2ch_gpu.wav',y./max(abs(y)),fs)

saveps = strcat('data/PESQ_multiele_', num2str(angle), '_', num2str(ns), 'ch_gpu.mat');
PS = gather([zeros(ns,est_length-1, 'gpuArray') PS]); % zero padding on left side of input signal
save(saveps,'PS')
