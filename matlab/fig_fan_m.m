%FIG_FAN_M  Environmental motion: false peer velocity at zero true velocity.
%
%   Both radios are clamped in all four captures, so the true peer velocity is
%   exactly zero and every non-zero output is a measured false velocity.  No
%   tare is applied (beta_+ is 0.01 Hz on this pair, far below the floor).
%
%   Also reported: the node A / node B disagreement, which needs no ground
%   truth at all and rises with the fan, so the loss of reciprocity is visible
%   from the data alone.
%
%   This figure is generated but NOT placed in the paper (page budget); its
%   numbers appear in Section V-D and in the effects table.
%
%   Reads : ../data/captures/capture_icc_FAN_OFF1.mat, FAN_ON, FAN_ON_FAST, FAN_OFF2
%   Writes: ../paper/figures/fig_fan_matlab.pdf

clear; clc;
HERE = fileparts(mfilename('fullpath'));
P    = @(n) fullfile(HERE,'..','data','captures',n);
CAPS = {P('capture_icc_FAN_OFF1.mat'), P('capture_icc_FAN_ON.mat'), ...
        P('capture_icc_FAN_ON_FAST.mat'), P('capture_icc_FAN_OFF2.mat')};
LAB  = {'fan off','fan on, 100 rpm','fan on, 800 rpm','fan off (after)'};
RPM  = [0 100 800 0];
RADIUS = 0.15;                     % blade radius [m]
OUT  = fullfile(HERE,'..','paper','figures','fig_fan_matlab.pdf');

n = numel(CAPS);
D = cell(n,1); V = cell(n,1); TT = cell(n,1); M = cell(n,1);

fprintf('\nENVIRONMENTAL MOTION (radios clamped, true velocity 0)\n');
fprintf('  %-18s %6s %8s %8s %8s %10s %10s\n','','n','RMS','p95','p99','A-B rms','tip m/s');
for i = 1:n
    d = icc_load(CAPS{i});
    v = d.v_dcomp;                          % no tare: beta_+ is 0.01 Hz on this pair
    % clamped for the whole capture: score everything after the calibration window
    m = ~d.m_cal & isfinite(v);
    dis = d.v_align - d.v_plain;            % node disagreement proxy, no truth needed
    tip = 2*pi*RADIUS*RPM(i)/60;
    fprintf('  %-18s %6d %8.3f %8.3f %8.3f %10.3f %10.1f\n', LAB{i}, nnz(m), ...
        sqrt(mean(v(m).^2)), prctile(abs(v(m)),95), prctile(abs(v(m)),99), ...
        sqrt(mean(dis(m & isfinite(dis)).^2)), tip);
    D{i}=d; V{i}=v; TT{i}=d.t; M{i}=m;
end

f = figure('Units','inches','Position',[1 1 7 2.6],'Color','w');
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); %#ok<NASGU>
nexttile; hold on; grid on; box on
off = 0;
for i = 1:n
    plot(TT{i}+off, V{i}, 'LineWidth',.5, 'DisplayName',LAB{i});
    off = off + TT{i}(end) + 5;
end
xlabel('time, captures concatenated [s]'); ylabel('false peer velocity [m/s]');
legend('Location','northwest','FontSize',6.5); set(gca,'FontSize',8);

nexttile; hold on; grid on; box on
for i = 1:n
    e = sort(abs(V{i}(M{i}))); e = e(isfinite(e));
    plot(e, linspace(0,1,numel(e)), 'LineWidth',.9, 'DisplayName',LAB{i});
end
set(gca,'XScale','log','FontSize',8); xlim([1e-3 inf]);
xlabel('|false velocity| [m/s]'); ylabel('CDF');
legend('Location','northwest','FontSize',6.5);

exportgraphics(f, OUT, 'ContentType','vector');
savefig(f, strrep(OUT,'.pdf','.fig'));      % editable MATLAB figure next to the PDF
fprintf('  wrote %s\n', OUT);
