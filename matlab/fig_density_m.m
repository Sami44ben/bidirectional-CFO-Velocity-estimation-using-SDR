%FIG_DENSITY_M  Paper Fig. 6: precision of the half sum against its closed form.
%
%   The recorded pilot grid of the static run is decimated in FREQUENCY (keep
%   every F-th pilot), in TIME (keep every T-th symbol) and in APERTURE (keep the
%   first M' symbols).  The per-burst pilot-slope estimator is re-run on each
%   subset and the standard deviation of the raw half sum is scored on the
%   held-out window (no tare, no drift correction).
%
%   Closed form (white noise), per direction, pilot SNR rho, Np pilots, symbol
%   instants t_m:   var(nu) = 1 / (2 rho Np (2 pi)^2 S),  S = sum (t_m - tbar)^2,
%   and var(half sum) = (var_D + var_U)/4.  Model: observed^2 = white^2 + excess^2,
%   with the excess (oscillator phase wander) common to all subcarriers and
%   scaling with the aperture as M^-gamma, gamma fitted here.
%
%   Left : std against pilot subcarriers, one curve per time decimation;
%          markers measured, dotted = white-noise bound, dashed = model.
%   Right: std against the aperture M'*Ts, all 30 pilots, same line styles.
%
%   Comb-12 of the recorded comb (every 24th subcarrier, 3 pilots) with a pilot on
%   every symbol is the NR PT-RS density (K=2, L=1) on this hardware.
%
%   Needs a capture recorded with ICC_LOG_PILOTS=1 (fields Rp_DL / Rp_UL).
%   The capture is ~350 MB; matfile() reads it in blocks.  Expect ~10 minutes.
%
%   Reads : ../data/captures/capture_icc_WARM_0922_0807.mat
%   Writes: ../paper/figures/fig_density.pdf and fig_density.fig

clear; clc;
HERE = fileparts(mfilename('fullpath'));
CAP  = fullfile(HERE,'..','data','captures','capture_icc_WARM_0922_0807.mat');
OUT  = fullfile(HERE,'..','paper','figures','fig_density.pdf');
F_LIST = [1 2 4 8 12 24];       % frequency decimation
T_LIST = [1 2 4];               % time decimation
M_LIST = [16 24 32 48 64 96 128 140];   % aperture (first M' symbols)
GAMMA_FROM = 32;                % fit the excess where the unwrap is reliable

mf = matfile(CAP);
info = whos(mf,'Rp_DL');
assert(~isempty(info),'this capture has no recorded pilots (record with ICC_LOG_PILOTS=1)');
sz = info.size; R = sz(1); NPIL = sz(2); NSYM = sz(3);
fprintf('pilots %d x %d x %d (exchanges x pilots x symbols)\n', R, NPIL, NSYM);

d    = icc_load(CAP);
tsym = double(d.TSYM);
co_d = double(d.cfo_dl_coarse(:));
co_u = double(d.cfo_ul_coarse(:));
ho   = d.m_score;
rho_d = 10.^(double(d.snr_dl(:))/10);          % pilot-EVM SNR, one per burst
rho_u = 10.^(double(d.snr_ul(:))/10);

% combos: {F, T, first M symbols}
C = zeros(0,3);
for T = T_LIST, for F = F_LIST, C(end+1,:) = [F T NSYM]; end, end %#ok<SAGROW>
for M = M_LIST(M_LIST < NSYM), C(end+1,:) = [1 1 M]; end %#ok<SAGROW>
nC = size(C,1);
FD = nan(R,nC);

BLK = 250;
fprintf('  reading %d exchanges in blocks of %d ...\n', R, BLK);
for i0 = 1:BLK:R
    i1 = min(i0+BLK-1, R);
    A = mf.Rp_DL(i0:i1,:,:);
    B = mf.Rp_UL(i0:i1,:,:);
    for c = 1:nC
        F = C(c,1); T = C(c,2); M = C(c,3);
        fs = 1:F:NPIL; ts = 1:T:M;
        for i = 1:(i1-i0+1)
            g = i0+i-1;
            if ~ho(g), continue; end
            a = reshape(A(i,fs,ts), numel(fs), numel(ts));
            b = reshape(B(i,fs,ts), numel(fs), numel(ts));
            if ~all(isfinite(a(:))) || ~all(isfinite(b(:))), continue; end
            FD(g,c) = (co_d(g) + slope_hz(a, tsym*T) + co_u(g) + slope_hz(b, tsym*T))/2;
        end
    end
end

% ---- measured std and the white-noise closed form ----------------------------
res = zeros(nC,6);               % [Np Nt aperture_ms std_hz white_hz mean_hz]
for c = 1:nC
    F = C(c,1); T = C(c,2); M = C(c,3);
    np = numel(1:F:NPIL); tt = (0:T:M-1)'*tsym; S = sum((tt-mean(tt)).^2);
    x  = FD(ho,c); x = x(isfinite(x));
    kk = 1/(2*np*(2*pi)^2*S);
    wb = sqrt(mean((kk./rho_d(ho) + kk./rho_u(ho))/4));
    res(c,:) = [np numel(tt) M*tsym*1e3 std(x) wb mean(x)];
end
full = find(C(:,1)==1 & C(:,2)==1 & C(:,3)==NSYM);
exc  = @(c) sqrt(max(res(c,4).^2 - res(c,5).^2, 0));
ex0  = exc(full);
ap   = find(C(:,1)==1 & C(:,2)==1);
apf  = ap(C(ap,3) >= GAMMA_FROM);
pfit = polyfit(log(C(apf,3)), log(arrayfun(exc, apf)), 1);
gam  = -pfit(1);
model = sqrt(res(:,5).^2 + (ex0*(C(:,3)/NSYM).^(-gam)).^2);    % excess flat in Np, M^-gamma in aperture

hz = d.hz;
fprintf('\n  %5s %5s %6s %9s %9s %9s %8s\n','Np','Nt','ap[ms]','std[m/s]','white','model','mean[Hz]');
for c = 1:nC
    fprintf('  %5d %5d %6.2f %9.4f %9.4f %9.4f %+8.2f\n', res(c,1),res(c,2),res(c,3), ...
            res(c,4)/hz, res(c,5)/hz, model(c)/hz, res(c,6));
end
fprintf('  white %.3f Hz, observed %.3f Hz (x%.2f), excess %.3f Hz, gamma %.2f, Np* %.1f\n', ...
        res(full,5), res(full,4), res(full,4)/res(full,5), ex0, gam, NPIL*(res(full,5)/ex0)^2);

% ---- figure --------------------------------------------------------------
f = figure('Units','inches','Position',[1 1 3.15 1.45],'Color','w');   % drawn at print size (placed at 0.9 column width = 3.15 in)
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); %#ok<NASGU>
cols = [0 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];
mk = {'o','s','^'};

nexttile; hold on; grid on; box on
for j = 1:numel(T_LIST)
    r = find(C(:,2)==T_LIST(j) & C(:,3)==NSYM);
    [~,o] = sort(res(r,1)); r = r(o);
    plot(res(r,1), res(r,4)/hz, mk{j}, 'Color',cols(j,:), 'MarkerFaceColor',cols(j,:), 'MarkerSize',3, ...
         'DisplayName',sprintf('1 in %d symbols',T_LIST(j)));
    plot(res(r,1), model(r)/hz, '--', 'Color',cols(j,:), 'LineWidth',.8, 'HandleVisibility','off');
    plot(res(r,1), res(r,5)/hz, ':', 'Color',cols(j,:), 'LineWidth',1, 'HandleVisibility','off');
end
set(gca,'XScale','log','YScale','log','FontSize',7); xlim([1.6 40]); ylim([0.012 0.14]);
set(gca,'XTick',[2 3 4 8 15 30],'YTick',[0.02 0.05 0.1],'YTickLabel',{'0.02','0.05','0.1'});
xlabel('pilots per symbol N_p'); ylabel('std of the half sum [m/s]');
lg_ = legend('Location','southwest','FontSize',6); try, lg_.ItemTokenSize = [10 6]; catch, end;

nexttile; hold on; grid on; box on
[~,o] = sort(C(ap,3)); r = ap(o);
plot(res(r,3), res(r,4)/hz, 'o', 'Color',cols(1,:), 'MarkerFaceColor',cols(1,:), 'MarkerSize',3, 'DisplayName','measured');
plot(res(r,3), model(r)/hz, '--', 'Color',cols(1,:), 'LineWidth',.8, 'DisplayName','model (8)');
plot(res(r,3), res(r,5)/hz, ':', 'Color',cols(1,:), 'LineWidth',1, 'DisplayName','white bound (7)');
plot(res(r,3), ex0*(C(r,3)/NSYM).^(-gam)/hz, '-.', 'Color',[.4 .4 .4], 'LineWidth',.8, ...
     'DisplayName',['excess M^{-' sprintf('%.2f',gam) '}']);
set(gca,'XScale','log','YScale','log','FontSize',7);
set(gca,'XTick',[0.3 0.5 1 2],'XTickLabel',{'0.3','0.5','1','2'}, ...
        'YTick',[0.02 0.05 0.1 0.2 0.5],'YTickLabel',{'0.02','0.05','0.1','0.2','0.5'});
xlabel('aperture M T_s [ms]');
lg_ = legend('Location','northeast','FontSize',6); try, lg_.ItemTokenSize = [10 6]; catch, end;

exportgraphics(f, OUT, 'ContentType','vector');
savefig(f, strrep(OUT,'.pdf','.fig'));      % editable MATLAB figure next to the PDF
fprintf('  wrote %s (+ .fig)\n', OUT);

% ==========================================================================
function r = slope_hz(Rp, tsym)
%SLOPE_HZ  The two-pass weighted pilot phase-slope estimator of acquisition/dsp.py.
[~, nsym] = size(Rp);
T = (0:nsym-1)'*tsym;
phi0 = unwrap(angle(sum(Rp .* conj(Rp(:,1)), 1))).';
s0 = wls(phi0, T, ones(nsym,1));
H  = mean(Rp .* exp(-1j*s0*T).', 2);      % .' NOT ' : ' conjugates
z  = sum(Rp .* conj(H), 1).';             % .' NOT ' : ' conjugates
phi = unwrap(angle(z)); w = abs(z).^2;
[s1,p1,t1] = wls(phi, T, w);
res = phi - (p1 + s1*(T - t1));
thr = max(0.5*pi, 3.0*1.4826*median(abs(res - median(res))));
keep = abs(res) <= thr;
if nnz(keep) < min(8,nsym), keep = true(nsym,1); end
s = wls(phi(keep), T(keep), w(keep));
r = s/(2*pi);
end

function [s,pb,tb] = wls(phi, t, w)
W = sum(w); tb = sum(w.*t)/W; pb = sum(w.*phi)/W;
den = sum(w.*(t-tb).^2);
if den > 0, s = sum(w.*(t-tb).*(phi-pb))/den; else, s = NaN; end
end
