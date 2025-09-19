% Microphone calibration

addpath('D:\Qian\202508Experiment_data_logging\Data_logging\')

% Calculate PSD
N = 2^14;
[PSD1,w] = PSD_Cfunc_amp( P1-mean(P1), P1-mean(P1) , N , 0.5*N , 4*N, 51200 );
[PSD2,w] = PSD_Cfunc_amp( PMT_OH_1-mean(PMT_OH_1), PMT_OH_1-mean(PMT_OH_1), N , 0.5*N , 4*N, 51200 );

%%

figure(1)
hold off
plot(w, abs(PSD1), w, abs(PSD2), 'r', 'LineWidth', 1)
hold on
plot([0 2000], [1 1], 'b--')
xlim([0 2000])

% subplot(2,1,2)
% hold off
% plot(w, angle(PSD2), 'k', w, angle(PSD1),'r', 'LineWidth', 1)
% hold on
% plot([0 2000], [1 1], 'b--')
% xlim([0 2000])
% ylim([-5 5])