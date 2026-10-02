%FIG_GAP_M  Paper Fig. 4: the gap sweep against its closed form.
%
%   Left : measured half-sum shift against the bias predicted from the SAME
%          run's drift rate and gap, -r*Delta/2.  Points on the diagonal mean
%          the first-order model holds.
%   Right: RMSE of the raw, drift-compensated and epoch-aligned estimators
%          against the inter-direction gap (markers), with the closed form of
%          the raw half sum, RMSE^2 = sigma^2 + (r*Delta/2)^2, drawn for the
%          warm drift rate and for the cold-start drift rate (lines).  The
%          dotted verticals mark Delta* = 2*sigma/|r|, where the drift bias
%          equals the floor.
%
%   No tare anywhere.  The radios are clamped throughout, so the true velocity
%   is zero and every output is error.
%
%   Reads : ../data/captures/capture_icc_GAP_<WARM>_*.mat and <COLD>,
%           ../data/captures/capture_icc_WARM_0922_0807.mat (the static floor sigma)
%   Writes: ../paper/figures/fig_gap.pdf and fig_gap.fig

clear; clc;
S_WARM = '0922_0840';
S_COLD = '0922_0912';
HERE = fileparts(mfilename('fullpath'));
OUT  = fullfile(HERE,'..','paper','figures','fig_gap.pdf');

% ---- the static floor sigma (held-out half-sum scatter of the static run) ----
w = icc_load(fullfile(HERE,'..','data','captures','capture_icc_WARM_0922_0807.mat'));
k = w.m_score & isfinite(w.fd_plain);
sigma = std(w.fd_plain(k));                       % [Hz]
hz = w.hz;
clear w

sets = struct('name',{'warm','cold'},'session',{S_WARM,S_COLD}, ...
              'col',{[0 0.45 0.74],[0.85 0.33 0.10]});

f = figure('Units','inches','Position',[1 1 3.15 1.4],'Color','w');   % drawn at print size (placed at 0.9 column width = 3.15 in)
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); %#ok<NASGU>
axL = nexttile; hold(axL,'on'); grid(axL,'on'); box(axL,'on');
axR = nexttile; hold(axR,'on'); grid(axR,'on'); box(axR,'on');
lims = 0; rs = zeros(1,numel(sets));

for s = 1:numel(sets)
    F = dir(fullfile(HERE,'..','data','captures',sprintf('capture_icc_GAP_%s_*.mat',sets(s).session)));
    if isempty(F), warning('no GAP captures for %s',sets(s).session); continue; end
    n = numel(F);
    gap=zeros(n,1); pred=zeros(n,1); meas=zeros(n,1); r=zeros(n,1); t0=zeros(n,1);
    rp=zeros(n,1); rd=zeros(n,1); ra=zeros(n,1); pp=zeros(n,1);
    for i = 1:n
        d = icc_load(fullfile(F(i).folder,F(i).name));
        ok = isfinite(d.fd_plain) & isfinite(d.f_lo);
        m  = ok & isfinite(d.fd_dcomp) & isfinite(d.fd_align);    % common support
        p1 = polyfit(d.t(ok), d.f_lo(ok), 1);                    % whole-run slope, the r of (7)
        p2 = polyfit(d.t(ok), d.f_lo(ok), 2);                    % r(t) for the closed form
        rt = polyval(polyder(p2), d.t);
        s0 = std(diff(d.fd_plain(ok)))/sqrt(2);                  % white part of the scatter
        r(i)    = p1(1);
        gap(i)  = median(d.gap,'omitnan');
        pred(i) = -r(i)*gap(i)/2;
        meas(i) = mean(d.fd_plain(ok));
        rp(i)   = rms_(d.fd_plain(m)/hz);
        rd(i)   = rms_(d.fd_dcomp(m)/hz);
        ra(i)   = rms_(d.fd_align(m)/hz);
        pp(i)   = sqrt(s0^2 + mean((rt(m).*d.gap(m)/2).^2))/hz;  % closed form, raw
        t0(i)   = double(d.T0_EPOCH);
    end
    [~,first] = min(t0);
    if s == 1, rs(s) = median(abs(r)); else, rs(s) = abs(r(first)); end

    fprintf('\nGAP SWEEP %s (%s), no tare, sigma %.3f Hz\n', sets(s).name, sets(s).session, sigma);
    fprintf('  %7s %8s %9s %9s | %7s %7s | %9s %9s\n','gap[s]','r[Hz/s]','pred[Hz]','meas[Hz]', ...
            'raw','model','driftcomp','aligned');
    [~,o] = sort(gap);
    for i = o'
        fprintf('  %7.2f %8.2f %9.2f %9.2f | %7.3f %7.3f | %9.3f %9.3f\n', ...
                gap(i),r(i),pred(i),meas(i),rp(i),pp(i),rd(i),ra(i));
    end

    plot(axL, pred, meas, 'o', 'MarkerSize',3, 'Color',sets(s).col, ...
         'MarkerFaceColor',sets(s).col, 'DisplayName',sets(s).name);
    lims = max([lims; abs(pred); abs(meas)]);

    % colour = session (blue warm, orange cold); shape = estimator
    plot(axR, gap(o), rp(o), 'o', 'Color',sets(s).col, 'MarkerFaceColor',sets(s).col, 'MarkerSize',3, 'HandleVisibility','off');
    plot(axR, gap(o), rd(o), 's', 'Color',sets(s).col, 'MarkerSize',4, 'LineWidth',.9, 'HandleVisibility','off');
    plot(axR, gap(o), ra(o), '^', 'Color',sets(s).col, 'MarkerSize',3, 'LineWidth',.9, 'HandleVisibility','off');
    plot(axR, gap(o), pp(o), 'x', 'Color',sets(s).col*0.6, 'MarkerSize',5, 'LineWidth',1, 'HandleVisibility','off');
end

% ---- closed form of the raw half sum, and the tolerance Delta* ------------
G = logspace(log10(0.02), log10(6), 200);
for s = 1:numel(sets)
    curve = sqrt(sigma^2 + (rs(s)*G/2).^2)/hz;
    plot(axR, G, curve, '-', 'Color',sets(s).col, 'LineWidth',.8, 'HandleVisibility','off');
    dstar = 2*sigma/rs(s);
    xline(axR, dstar, ':', 'Color',sets(s).col, 'LineWidth',1, 'HandleVisibility','off');
    ha = 'right'; xx = dstar/1.08;
    fprintf('  %s: |r| = %.2f Hz/s -> Delta* = %.3f s (bias = floor), %.3f s (RMSE +10%%)\n', ...
            sets(s).name, rs(s), dstar, dstar*sqrt(0.21));
end

lims = 1.15*lims;
plot(axL,[-lims lims],[-lims lims],'k--','LineWidth',.8,'DisplayName','y = x');
xlabel(axL,'predicted bias  -r\Delta/2  [Hz]'); ylabel(axL,'measured half sum shift [Hz]');
lg_ = legend(axL,'Location','northwest','FontSize',6); try, lg_.ItemTokenSize = [10 6]; catch, end; set(axL,'FontSize',7);

% legend by shape only (colour = session, stated in the caption)
k0 = [.2 .2 .2];
plot(axR,nan,nan,'o','Color',k0,'MarkerFaceColor',k0,'MarkerSize',3,'DisplayName','half sum');
plot(axR,nan,nan,'s','Color',k0,'MarkerSize',4,'LineWidth',.9,'DisplayName','drift comp.');
plot(axR,nan,nan,'^','Color',k0,'MarkerSize',3,'LineWidth',.9,'DisplayName','epoch aligned');
set(axR,'XScale','log','YScale','log','FontSize',7);
xlim(axR,[0.02 6]); ylim(axR,[0.018 2]);
set(axR,'XTick',[0.05 0.2 1 4],'XTickLabel',{'0.05','0.2','1','4'},'XTickLabelRotation',0, ...
        'YTick',[0.02 0.1 0.5],'YTickLabel',{'0.02','0.1','0.5'});
xlabel(axR,'DL-UL gap \Delta [s]'); ylabel(axR,'RMSE [m/s]');
legend(axR,'off');   % marker shapes are named in the paper caption (no room in a 1.4 in panel)   % crosses and lines are named in the caption

exportgraphics(f, OUT, 'ContentType','vector');
savefig(f, strrep(OUT,'.pdf','.fig'));      % editable MATLAB figure next to the PDF
fprintf('\n  wrote %s (+ .fig)\n', OUT);

function s = fmt_s(x)
if x < 1, s = sprintf('%.0f ms', 1e3*x); else, s = sprintf('%.1f s', x); end
end

function r = rms_(x)
x = x(isfinite(x)); r = sqrt(mean(x.^2));
end
