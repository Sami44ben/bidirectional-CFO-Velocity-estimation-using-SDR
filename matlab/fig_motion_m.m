%FIG_MOTION_M  Paper Fig. 5: peer motion.
%
%   Top   : the drift-compensated velocity estimate over the run, with the exchanges below the
%           SNR gate marked.  The dashed line ends the static phase.
%   Bottom: its integral, i.e. the displacement the estimate implies.
%
%   The travelled distance was not independently instrumented, so this figure
%   establishes sign resolution, speed ordering and the absence of drift under
%   motion.  The accuracy figures in the paper come from the injected sweep.
%
%   Reads : ../data/captures/capture_icc_MOVE_REF_test2.mat
%   Writes: ../paper/figures/fig_motion.pdf

clear; clc;
HERE = fileparts(mfilename('fullpath'));
CAP  = fullfile(HERE,'..','data','captures','capture_icc_MOVE_REF_test2.mat');
OUT  = fullfile(HERE,'..','paper','figures','fig_motion.pdf');

d = icc_load(CAP);
v  = d.v_dcomp;                         % drift compensated; no tare (beta_+ is 0.01 Hz on this pair)
t  = d.t; ok = d.ok & isfinite(v);

% interpolate across gated exchanges so the integral has no holes
vi = interp1(t(ok), v(ok), t, 'linear', 'extrap');
s  = cumtrapz(t, vi);                               % implied displacement [m]

% ---- motion bursts: |v| above a threshold, gaps of one exchange bridged ---
vs  = movmean(vi, 5, 'omitnan');
act = abs(vs) > 0.25 & d.m_dyn;
runs = []; k = 1;
while k <= numel(t)
    if act(k)
        j = k;
        while j+1 <= numel(t) && (act(j+1) || (j+2 <= numel(t) && act(j+2))), j = j+1; end
        if t(j)-t(k) > 0.8, runs(end+1,:) = [k j]; end %#ok<SAGROW>
        k = j+1;
    else
        k = k+1;
    end
end

stat = d.m_static & ok;
fprintf('\nPEER MOTION  %s\n', d.tag);
fprintf('  %d exchanges, %.0f s, gate keeps %d (%.0f%%)\n', d.R, t(end), nnz(ok), 100*nnz(ok)/d.R);
fprintf('  static phase: mean %+.4f m/s, sd %.4f\n', mean(v(stat)), std(v(stat)));
fprintf('  %d motion bursts:\n', size(runs,1));
fprintf('    %8s %7s %12s %9s %9s\n','start[s]','dur[s]','displ.[m]','mean|v|','peak|v|');
for i = 1:size(runs,1)
    a = runs(i,1); b = runs(i,2);
    ds = abs(s(b)-s(a));
    fprintf('    %8.1f %7.1f %12.2f %9.2f %9.2f\n', t(a), t(b)-t(a), ds, ds/(t(b)-t(a)), max(abs(vi(a:b))));
end
fprintf('  net displacement over the run: %+.2f m\n', s(end));

% ---- figure --------------------------------------------------------------
tsw = double(d.STATIC_S);
f = figure('Units','inches','Position',[1 1 3.15 1.3],'Color','w');   % drawn at print size (placed at 0.9 column width = 3.15 in)
tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact'); %#ok<NASGU>

nexttile; hold on; grid on; box on
plot(t(ok), v(ok), 'LineWidth',.6);
plot(t(~ok), v(~ok), '.', 'MarkerSize',4, 'Color',[0.85 0.33 0.10], 'DisplayName','below SNR gate');
xline(tsw,'k--','LineWidth',.7,'HandleVisibility','off');
ylim([-2.6 2.6]); ylabel('v [m/s]'); set(gca,'FontSize',7,'XTickLabel',[]);
lg_ = legend({'estimate','below SNR gate'},'Location','southwest','FontSize',6); try, lg_.ItemTokenSize = [10 6]; catch, end;

nexttile; hold on; grid on; box on
plot(t, s, 'LineWidth',.9);
xline(tsw,'k--','LineWidth',.7);
ylabel('displacement [m]'); xlabel('host time [s]'); set(gca,'FontSize',7);

exportgraphics(f, OUT, 'ContentType','vector');
savefig(f, strrep(OUT,'.pdf','.fig'));      % editable MATLAB figure next to the PDF
fprintf('  wrote %s\n', OUT);
