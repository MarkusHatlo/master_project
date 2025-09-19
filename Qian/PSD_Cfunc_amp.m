function [PSD,w] = PSD_Cfunc_amp( sig, sig0 , N , No, Nfft, FS )
% power spectral density PSD = P.*conj(P) ，注意P为频率阈，非时域
% cross power spectra between the reference signal p′ref from the signal generator and the measured pressure fluctuations p′i.
% Each spectrum was obtained using the Welch method by averaging 50% overlapping segments of the signal multiplied by a Hanning window

% sig:  (data(:,4)-mean(data(:,4)))  
% sig0: (data(:,4)-mean(data(:,4)))
% N: 2^12, number of samples
% No: floor(0.5.*N),determines overlap
% Nfft: 2.*N  % 返回NFFT点DFT，提高计算速度，节省计算资源
% FS: frame sample rate

% 此处重叠率50% welch method(Welch方法是一种常用于估计信号功率谱密度的频谱分析方法，它是一种改进的周期图法。Welch方法通过将信号分成多个重叠的子段，计算每个子段的傅里叶变换，最后取平均来得到频谱估计。这种方法相对于传统的周期图方法更具有稳定性和抗噪声的能力。)
y = buffer(sig, N , No); %  buffer(x,n,p) 将输入序列划分为重叠的段，傅里叶变换前需要分段。n每段长度，p相邻段间重叠长度
y0 = buffer(sig0, N , No);

[n1,n2] = size(y);

w = ([0:Nfft-1]./Nfft - 0.5).*FS; % 
df = FS./N;
PSD = 0.*w';
PSDr = 0.*w';

for j = 2:n2-1
    % 加窗是因为想避免ifft之后的结果的 end effects
    PSD1 =  2.*(fftshift(fft(  hanning(n1).*( y(:,j) )  , Nfft )./n1 )); % Hanning窗就是β=0.5的余弦窗
    PSD2 =  2.*(fftshift(fft(  hanning(n1).*( y0(:,j) )  , Nfft )./n1 ));   
    PSD = PSD + PSD1.*conj(PSD2); % power spectral density 
    PSDr = PSDr + PSD2.*conj(PSD2);    
end

% estimated pressure amplitude (Erik 论文公式3.3)
PSD = 2.*PSD((w>2.*FS./N))./(n2-2)./  sqrt( (PSDr((w>2.*FS./N))+max(PSDr((w>2.*FS./N))).*1e-4 )./ (n2-2)  );
w = w((w>2.*FS./N));

PSD = PSD;%.*exp(-1j);

end