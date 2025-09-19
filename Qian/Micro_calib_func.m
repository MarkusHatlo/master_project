% close all
N = 2^12;

[PSD1,w] = PSD_Cfunc_amp( data.P1 ,data.Pref, N , 0.5*N, 4*N, 51200 );
[PSD2,w] = PSD_Cfunc_amp( data.P2 ,data.Pref, N , 0.5*N, 4*N, 51200 );
[PSD3,w] = PSD_Cfunc_amp( data.P3 ,data.Pref, N , 0.5*N, 4*N, 51200 );
% [PSD4,w] = PSD_Cfunc_amp( data.P4 ,data.Pref, N , 0.5*N, 4*N, 51200 );
% [PSD5,w] = PSD_Cfunc_amp( data.P5 ,data.Pref, N , 0.5*N, 4*N, 51200 );
% [PSD6,w] = PSD_Cfunc_amp( data.P6 ,data.Pref, N , 0.5*N, 4*N, 51200 );


figure(123490)
subplot(2,1,1)

% plot(w, abs(PSD1)./abs(PSD1), 'k--', w, abs(PSD2)./abs(PSD1), 'b', w, abs(PSD3)./abs(PSD1), 'r')
% xlim([0 2050])
plot(w, abs(PSD3)./abs(PSD1), 'r', w, abs(PSD2)./abs(PSD1), 'b')
xlim([0 2050])

ylim([0 1.3])

subplot(2,1,2)
% plot(w, angle(PSD1./PSD1), 'k--', w, angle(PSD2./PSD1), 'b', w, angle(PSD3./PSD1), 'r')
plot(w, angle(PSD3./PSD1), 'r', w, angle(PSD2./PSD1), 'b')
xlim([0 2050])
ylim([-1 1]*pi)