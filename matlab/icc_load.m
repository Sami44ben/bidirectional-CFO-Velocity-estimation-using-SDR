function d = icc_load(path)
%ICC_LOAD  Load an ICC capture and add every derived estimator.
%
%   d = icc_load('../data/captures/capture_icc_WARM_0922_0807.mat')
%
% This is the MATLAB twin of bidir_estimators.py.  It works from the RAW
% per-direction quantities that every capture carries, so nothing depends on
% what the acquisition script happened to pre-compute.
%
% Raw fields used:  cfo_dl, cfo_ul  [Hz]   (DL = A->B measured at B, UL = B->A at A)
%                   t_dl_rel, t_ul_rel [s], gap_s [s], phase (0 static, 1 dynamic)
%                   snr_dl, snr_ul [dB], se_dl, se_ul [Hz]
%                   HZ_PER_MPS = FC/c, CAL_S, STATIC_S
%
% Added fields:
%   d.f_lo        half difference  (cfo_dl - cfo_ul)/2   -> oscillator offset [Hz]
%   d.fd_plain    half sum         (cfo_dl + cfo_ul)/2   -> Doppler [Hz]
%   d.drift       local slope of f_lo, Hz/s (TRAILING LS over the last DRIFT_WINDOW exchanges)
%   d.fd_dcomp    fd_plain + drift.*gap/2                -> drift compensated [Hz]
%   d.fd_align    DL interpolated to the UL epoch, then half sum [Hz]
%   d.v_*         the same four, divided by HZ_PER_MPS   -> [m/s]
%   d.v_1way      cfo_dl read directly as Doppler        -> [m/s]
%   d.ok          per-exchange quality gate (both SNRs above GATE_DB)
%   d.m_static, d.m_dyn, d.m_cal, d.m_score   exchange masks
%
% Constants below match Work/SDR/bidir_estimators.py exactly.

DRIFT_WINDOW = 3;       % exchanges in the trailing f_lo slope fit (causal)
MAX_SPAN_MULT = 3.0;    % refuse to interpolate across a hole wider than this
GATE_DB = 8;            % per-direction pilot-EVM SNR gate

d = load(path);
g = @(f) double(d.(f)(:));

t   = g('t_dl_rel');
tu  = g('t_ul_rel');
a2b = g('cfo_dl');
b2a = g('cfo_ul');
if isfield(d,'gap_s'), gap = g('gap_s'); else, gap = tu - t; end
hz  = double(d.HZ_PER_MPS);
R   = numel(t);

% ---- the separation ------------------------------------------------------
d.f_lo     = (a2b - b2a)/2;
d.fd_plain = (a2b + b2a)/2;

% ---- local drift rate of the oscillator ---------------------------------
% TRAILING (causal) least squares: exchange i uses exchanges i-W+1..i only, i.e.
% what the link has measured when exchange i completes.  The first W-1 exchanges
% are left NaN rather than fitted on a truncated window.
D = nan(R,1);
for i = DRIFT_WINDOW : R
    idx = i-DRIFT_WINDOW+1 : i;
    tt = t(idx); ff = d.f_lo(idx);
    k = isfinite(tt) & isfinite(ff);
    if nnz(k) > 2
        tb = mean(tt(k));
        den = sum((tt(k)-tb).^2);
        if den > 0
            D(i) = sum((tt(k)-tb).*(ff(k)-mean(ff(k))))/den;
        end
    end
end
d.drift    = D;
d.fd_dcomp = d.fd_plain + D.*gap/2;

% ---- epoch alignment: move the DL observation to the UL epoch ------------
% Linear interpolation between the two DL bursts bracketing t_ul, refused
% across any bracket wider than MAX_SPAN_MULT times the median spacing.
dt_med = median(diff(t),'omitnan');
a2b_at_ul = nan(R,1);
for i = 1:R-1
    span = t(i+1) - t(i);
    if ~isfinite(span) || span > MAX_SPAN_MULT*dt_med, continue; end
    w = (tu(i) - t(i)) / span;
    if w < -0.5 || w > 1.5, continue; end
    a2b_at_ul(i) = (1-w)*a2b(i) + w*a2b(i+1);
end
d.fd_align = (a2b_at_ul + b2a)/2;

% ---- velocities ----------------------------------------------------------
d.v_plain = d.fd_plain / hz;
d.v_dcomp = d.fd_dcomp / hz;
d.v_align = d.fd_align / hz;
d.v_1way  = a2b        / hz;

% ---- masks ---------------------------------------------------------------
d.ok = isfinite(d.fd_plain) & g('snr_dl') > GATE_DB & g('snr_ul') > GATE_DB;
if isfield(d,'phase') && isfield(d,'STATIC_S') && double(d.STATIC_S) > 0
    ph = g('phase');
    d.m_static = ph == 0;
    d.m_dyn    = ph == 1;
else
    d.m_static = true(R,1);
    d.m_dyn    = false(R,1);
end
if isfield(d,'CAL_S') && double(d.CAL_S) > 0
    d.m_cal = t < double(d.CAL_S);
else
    d.m_cal = (1:R)' <= min(100,R);
end
d.m_score = d.m_static & ~d.m_cal;      % held out: static, not used for the tare

d.t = t; d.gap = gap; d.hz = hz; d.R = R;
d.tag = char(string(d.RUN_TAG));
end
