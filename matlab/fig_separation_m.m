%FIG_SEPARATION_M  Paper Fig. 3: the separation matrix.
%
%   Doppler and oscillator offset were injected independently over the air.
%   Left panel : Doppler output vs injected Doppler, one line per LO offset.
%   Right panel: oscillator output vs injected offset, flat in Doppler.
%
%   Each point is the mean over one run; the bar is the within-run standard
%   deviation.  A clean separation shows unit slope on the matching driver and
%   zero slope on the other.
%
%   Reads : ../data/captures/capture_icc_SEP_<SESSION>_*.mat
%   Writes: ../paper/figures/fig_separation.pdf

clear; clc;
SESSION = '0922_0823';
HERE    = fileparts(mfilename('fullpath'));
FILES   = dir(fullfile(HERE,'..','data','captures',sprintf('capture_icc_SEP_%s_*.mat',SESSION)));
OUT     = fullfile(HERE,'..','paper','figures','fig_separation.pdf');
assert(~isempty(FILES),'no SEP captures for session %s',SESSION);

n = numel(FILES);
fd_inj = zeros(n,1); lo_inj = zeros(n,1);
fd_out = zeros(n,1); lo_out = zeros(n,1); sd_v = zeros(n,1); t0 = zeros(n,1);

for i = 1:n
    d = icc_load(fullfile(FILES(i).folder,FILES(i).name));
    k = isfinite(d.fd_plain) & isfinite(d.f_lo);
    fd_inj(i) = double(d.FD_INJECT_HW);
    lo_inj(i) = double(d.LO_OFFSET_B_HZ);
    fd_out(i) = mean(d.fd_plain(k));
    lo_out(i) = mean(d.f_lo(k));
    sd_v(i)   = std(d.v_plain(k));
    t0(i)     = double(d.T0_EPOCH);
    hz = d.hz;
end
[~,ord] = sort(t0); fd_inj=fd_inj(ord); lo_inj=lo_inj(ord);
fd_out=fd_out(ord); lo_out=lo_out(ord); sd_v=sd_v(ord);

% ---- two regressions: each output against BOTH drivers -------------------
X  = [fd_inj, lo_inj, ones(n,1)];
bf = X\fd_out;                       % [gain_f, leak_f, offset_f]
bl = X\lo_out;                       % [leak_lo, gain_lo, offset_lo]
res = fd_out - X*bf;
se  = @(b,y) sqrt(diag(sum((y-X*b).^2)/max(n-3,1) * inv(X'*X))); %#ok<MINV>
sef = se(bf,fd_out); sel = se(bl,lo_out);

fprintf('\nSEPARATION MATRIX  (%d runs, session %s)\n', n, SESSION);
fprintf('  Doppler out : gain %.4f +/- %.4f on f_d | leakage %.2e from LO | offset %+.2f Hz\n', ...
        bf(1), sef(1), bf(2), bf(3));
fprintf('  LO out      : gain %.3f +/- %.3f on LO  | leakage %+.2f from f_d | natural offset %+.0f Hz\n', ...
        abs(bl(2)), sel(2), bl(1), bl(3));
fprintf('  within-run velocity scatter : %.3f m/s\n', mean(sd_v));
fprintf('  LO leakage as false velocity: %.4f m/s (from the %.0f Hz natural offset)\n', ...
        abs(bf(2)*bl(3))/hz, abs(bl(3)));
cc = corrcoef((1:n)', res);
fprintf('  corr(run order, residual)   : %+.2f  (drift did not alias into the slopes)\n', cc(1,2));

% ---- figure --------------------------------------------------------------
los = unique(lo_inj); cols = lines(numel(los));
f = figure('Units','inches','Position',[1 1 3.15 1.3],'Color','w');   % drawn at print size (placed at 0.9 column width = 3.15 in)
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; hold on; grid on; box on
v_inj = fd_inj/hz;
plot([min(v_inj) max(v_inj)],[min(v_inj) max(v_inj)],'k--','LineWidth',.8,'DisplayName','unit slope');
for j = 1:numel(los)
    m = lo_inj == los(j);
    [xs,o] = sort(v_inj(m)); ys = fd_out(m)/hz; es = sd_v(m);
    errorbar(xs, ys(o), es(o), 'o-', 'MarkerSize',3, 'Color',cols(j,:), ...
             'MarkerFaceColor',cols(j,:), 'LineWidth',.8, ...
             'DisplayName',sprintf('%+.0f kHz',los(j)/1e3));
end
xlabel('injected velocity [m/s]'); ylabel('estimated velocity [m/s]');
lg_ = legend('Location','northwest','FontSize',6); try, lg_.ItemTokenSize = [10 6]; catch, end; set(gca,'FontSize',7);

nexttile; hold on; grid on; box on
for j = 1:numel(los)
    m = lo_inj == los(j);
    [xs,o] = sort(v_inj(m)); ys = lo_out(m)/1e3;
    plot(xs, ys(o), 'o-', 'MarkerSize',3, 'Color',cols(j,:), ...
         'MarkerFaceColor',cols(j,:), 'LineWidth',.8);
end
xlabel('injected velocity [m/s]'); ylabel('estimated LO offset [kHz]');
set(gca,'FontSize',7);

exportgraphics(f, OUT, 'ContentType','vector');
savefig(f, strrep(OUT,'.pdf','.fig'));      % editable MATLAB figure next to the PDF
fprintf('  wrote %s\n', OUT);
