import os
import sys

# ==============================================================
# SCHEME=nn3 : per-flavour FR three ways (binned map / neural sv3 / truth), 2024 v15 QCD.
# Two passes (torch lives only in the BDT venv, ROOT only in the container):
#   NNPASS=dump  cd BDT && .venv-fakeflavor/bin/python <this file>      -> nn/_cells.json
#   NNPASS=draw  singularity ... python3 <this file>                    -> nn/*.png
# ==============================================================
NN_PT_EDGES = [10., 12., 14., 17., 20., 30., 50., 100.]
NN_ETA_EDGES = [0., 0.9, 1.6, 2.4]
NN_FLAV = {"b": 0, "c": 1, "uds": 2, "g": 3, "jetless": -1}     # flavor_dr codes (4 = pileup is dropped)
NN_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "wp_sip_dr_v15", "QCD", "2024", "nn")


MAP_COLS = ['mu_conept', 'mu_abseta', 'isTight', 'isPrompt', 'ev_met', 'ev_mt', 'ev_hasAway40',
            'pass_mu8', 'pass_mu17', 'w_mu8', 'w_mu17']


def binned_map(cols):
    """The measurement region (non-prompt, MET < 25, MT < 25, away jet > 40, Mu8 below conePt 30 and Mu17 above)
    and the flavour-inclusive binned map N_tight/N_loose per (conePt, |eta|) cell over all its rows.
    Returns w, tight, ok, ip, ie, |eta|, fmap."""
    import numpy as np
    pt = np.minimum(np.asarray(cols['mu_conept'], float), NN_PT_EDGES[-1] - 0.1)
    eta = np.asarray(cols['mu_abseta'], float)
    lowpath = pt <= 30.
    w = np.where(lowpath, np.asarray(cols['w_mu8'], float) * np.asarray(cols['pass_mu8']).astype(bool),
                 np.asarray(cols['w_mu17'], float) * np.asarray(cols['pass_mu17']).astype(bool))
    tight = np.asarray(cols['isTight']).astype(bool)
    meas = (~np.asarray(cols['isPrompt']).astype(bool)) & (np.asarray(cols['ev_met']) < 25) \
        & (np.asarray(cols['ev_mt']) < 25) & np.asarray(cols['ev_hasAway40']).astype(bool) & (w != 0)
    ip = np.digitize(pt, NN_PT_EDGES) - 1
    ie = np.digitize(eta, NN_ETA_EDGES) - 1
    ok = meas & (ip >= 0) & (ip < len(NN_PT_EDGES) - 1) & (ie >= 0) & (ie < len(NN_ETA_EDGES) - 1)
    npt, neta = len(NN_PT_EDGES) - 1, len(NN_ETA_EDGES) - 1
    fmap = np.zeros((npt, neta))
    for a in range(npt):
        for b in range(neta):
            m = ok & (ip == a) & (ie == b)
            fmap[a, b] = w[m & tight].sum() / w[m].sum()
    return w, tight, ok, ip, ie, eta, fmap


def map_dump():
    """SCHEME=map1d|map2d NNPASS=dump (venv, no model): the binned map from the QCD parquet -> ../_map.json.
    FR_PARQUET overrides the parquet (default: the 2024 case-2 QCD skim under NN_BDT)."""
    import json
    import pyarrow.parquet as pq
    bdt = os.environ.get('NN_BDT', '/data6/Users/snuintern2/BDT')
    src = os.environ.get('FR_PARQUET', os.path.join(bdt, 'fakeflavor/data/ntuples_nff24wpks/QCD_MuEnriched.parquet'))
    t = pq.read_table(src, columns=MAP_COLS)
    fmap = binned_map({c: t[c].to_numpy() for c in MAP_COLS})[-1]
    out = os.path.join(os.path.dirname(NN_OUT), '_map.json')
    json.dump(dict(pt_edges=NN_PT_EDGES, eta_edges=NN_ETA_EDGES, fmap=fmap.tolist(), parquet=src, rows=t.num_rows),
              open(out, 'w'))
    print("wrote", out, "from", src, t.num_rows, "rows")


def nn3_dump():
    import json
    import numpy as np
    from pathlib import Path
    bdt = os.environ.get('NN_BDT', '/data6/Users/snuintern2/BDT')
    sys.path.insert(0, bdt)
    os.chdir(bdt)
    from src.utils import load_yaml
    from neuralff import data as D
    from neuralff.fakefactor import FakeFactor

    cfg = load_yaml('neuralff/config/nff3_wp.yaml')
    ff = FakeFactor('neuralff/data/models/nff_qcd24wp_sv3.pt')
    need = ['flavor', 'flavor_dr', 'ev_met', 'ev_mt', 'ev_hasAway40', 'isPrompt', 'isData',
            'pass_mu8', 'pass_mu17', 'w_mu8', 'w_mu17']
    need += list(ff.enc.numerical) + [c for c in ff.enc.categorical if '__isnan' not in c]
    cols = D.load(Path('fakeflavor/data/ntuples_nff24wpks/QCD_MuEnriched.parquet'), cfg, 'sv3',
                  extra=sorted(set(need)))
    n = len(cols['isTight'])
    sp = {k: np.asarray(v) for k, v in ff.splits.items()}
    if sum(len(v) for v in sp.values()) != n:
        raise SystemExit(f"split covers {sum(len(v) for v in sp.values())} rows, parquet gives {n}")
    test = np.zeros(n, bool); test[sp['test']] = True

    f_nn = ff.fake_rate(cols)
    w, tight, ok, ip, ie, eta, fmap = binned_map(cols)
    npt, neta = len(NN_PT_EDGES) - 1, len(NN_ETA_EDGES) - 1
    f_bin = np.where(ok, fmap[np.clip(ip, 0, npt - 1), np.clip(ie, 0, neta - 1)], 0.)

    flavcol = os.environ.get('FLAVCOL', 'flavor_dr')
    fl = np.asarray(cols[flavcol]).astype(int)
    cells = {}
    for name, code in NN_FLAV.items():
        cells[name] = {}
        for sl in range(neta + 1):                      # 0 = |eta| integrated
            rows = []
            for a in range(npt):
                m = ok & test & (fl == code) & (ip == a) & ((ie == sl - 1) if sl else True)
                ww = w[m]
                rows.append(dict(n=int(m.sum()), sw=float(ww.sum()), sw2=float((ww ** 2).sum()),
                                 swt=float(ww[tight[m]].sum()), swb=float((ww * f_bin[m]).sum()),
                                 swn=float((ww * f_nn[m]).sum()), swn2=float((ww * f_nn[m] ** 2).sum())))
            cells[name][str(sl)] = rows
    json.dump(dict(pt_edges=NN_PT_EDGES, eta_edges=NN_ETA_EDGES, fmap=fmap.tolist(), flavcol=flavcol,
                   n_test=int(test.sum()), n_ok_test=int((ok & test).sum()), cells=cells),
              open(os.path.join(NN_OUT, '_cells.json'), 'w'))
    print("wrote", os.path.join(NN_OUT, '_cells.json'), "test rows", int(test.sum()))

    # gate: matrix-form QCD self-closure per flavour on the test split, NO measurement cuts
    # (the population of F_flavour_split_run2024_case2_full.png); pred = sum_fail w f/(1-f), truth = sum_tight w
    wa = np.asarray(cols['weight'], float)
    pt0 = np.asarray(cols['mu_conept'], float)
    ip0 = np.clip(np.digitize(np.minimum(pt0, 99.9), NN_PT_EDGES) - 1, 0, npt - 1)
    ie0 = np.clip(np.digitize(eta, NN_ETA_EDGES) - 1, 0, neta - 1)
    fm0 = np.zeros((npt, neta))
    for a in range(npt):
        for b in range(neta):
            m = (ip0 == a) & (ie0 == b)
            fm0[a, b] = wa[m & tight].sum() / wa[m].sum()
    fb0 = fm0[ip0, ie0]
    clip = lambda f: np.clip(f, 1e-3, 0.99)
    for col in ('flavor', 'flavor_dr'):
        fc = np.asarray(cols[col]).astype(int)
        out = []
        for name, code in NN_FLAV.items():
            m = test & (fc == code)
            t = wa[m & tight].sum()
            pb = (wa[m & ~tight] * clip(fb0[m & ~tight]) / (1 - clip(fb0[m & ~tight]))).sum()
            pn = (wa[m & ~tight] * clip(f_nn[m & ~tight]) / (1 - clip(f_nn[m & ~tight]))).sum()
            out.append(f"{name} bin {pb / t:.2f} nn {pn / t:.2f}")
        print(f"closure[{col}]:", " | ".join(out))


if os.environ.get('SCHEME') == 'nn3' and os.environ.get('NNPASS', 'draw') == 'dump':
    os.makedirs(NN_OUT, exist_ok=True)
    nn3_dump()
    sys.exit(0)
if os.environ.get('SCHEME') in ('map1d', 'map2d') and os.environ.get('NNPASS', 'draw') == 'dump':
    os.makedirs(os.path.dirname(NN_OUT), exist_ok=True)
    map_dump()
    sys.exit(0)

import ROOT
import glob
from array import array
from ROOT import TCanvas, TLatex, TLegend, TGraphAsymmErrors
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# ==============================================================
# Per-parton-flavor muon fake rate (QCD MuEnriched, MC truth).
# The fake muon's source jet is matched (gen / reco) and its parton
# flavor split into b / c / s / d / u / g.  FR = N(tight)/N(loose)
# per flavor, stitched Mu8 (ptcorr<=30) + Mu17 (ptcorr>30), overlaid.
# Histograms produced by fake.cc at
#   {path}/MeasReg/Central/{wp}/{scheme}/{flavor}/fake_ptcorr_abseta
# ==============================================================

era = os.environ.get('ERA', '2016preVFP_flavor')
sample = os.environ.get('SAMPLE', 'QCD')
scheme = os.environ.get('SCHEME', 'genflav')   # genflav (primary) or recoflav
paths = ["Mu8", "Mu17"]
switchPt = 30.

# XVAR=ptcorr (default): muon cone-corrected pT (plots/cone/...).
# XVAR=genjetpt: the matched source GenJet's pT instead (fake_genjetpt_abseta,
# fake.cc) -- only filled under genflav, so SCHEME must be genflav.
# XVAR=recojetpt: the matched source reco jet's pT (fake_recojetpt_abseta,
# fake.cc) -- only filled under recoflav, so SCHEME must be recoflav.
# genjetpt/recojetpt land under plots/jet/... instead of plots/cone/....
xvar = os.environ.get('XVAR', 'ptcorr')
if xvar not in ('ptcorr', 'genjetpt', 'recojetpt'):
    raise ValueError(f"XVAR must be 'ptcorr', 'genjetpt' or 'recojetpt', got {xvar!r}")
if xvar == 'genjetpt' and scheme != 'genflav':
    raise ValueError("XVAR=genjetpt only exists under SCHEME=genflav (fake.cc only fills it there)")
if xvar == 'recojetpt' and scheme != 'recoflav':
    raise ValueError("XVAR=recojetpt only exists under SCHEME=recoflav (fake.cc only fills it there)")
histSuffix = {"ptcorr": "fake_ptcorr_abseta",
              "genjetpt": "fake_genjetpt_abseta",
              "recojetpt": "fake_recojetpt_abseta"}[xvar]

# ERAS: comma-separated era dirs summed into one curve (full-Run2 / full-Run3
# stacking).  Light-flavor fakes are far too rare in a single era -- summing
# 2016preVFP+2016postVFP+2017+2018 lifts uds/gluon out of the 1-event regime.
# Stack only WITHIN a run period: Run2 and Run3 differ in detector and, for
# electrons, in the tight MVA WP itself (elecfake.cc:43-44).
eras = [e.strip() for e in os.environ.get('ERAS', era).split(',') if e.strip()]
# Smallest effective-event count a bin must have before it is drawn at all.
# 1 = draw everything that has any entry.  Be aware that at N_eff<=2 a bin lands
# on exactly 0.000 or 1.000 with a meaningless error bar; raise to 10 to cut
# those (that also drops gen gluon at ptcorr 50-100, which has N_eff 4.9).
minNeff = float(os.environ.get('MIN_NEFF', '1'))

# u/d/s are drawn as one light-quark curve: individually they are statistically
# thin and partonFlavour cannot separate them reliably anyway.
# "pileup" (parton==0 && hadron==0) only exists in recoflav -- the GenJet
# collection has no pileup jets -- so that curve is simply absent from gen plots.
# "undef" (unmatched) is the jet-less category; it stays out of the default plot
# (its statistics collapse with pt) and comes back only under JETLESS=1 below.
# JETLESS=1 : swap the pileup curve for the jet-less one and write into
# plots/Jetless/ instead of plots/.  "jet-less" is the recoflav "unmatched"
# category = the lepton carries no jet pointer (Muon/Electron JetIdx() < 0), i.e.
# it was never clustered into a PF jet -- the population the FR model struggles
# with.  Default off, so the plots under plots/{QCD,TTLJ} are untouched.
jetless = os.environ.get('JETLESS', '0') not in ('', '0', 'false', 'False')
flavors = ["uds", "c", "b", "g", "pileup"]
if jetless:
    flavors = ["uds", "c", "b", "g", "undef"]
# FLAVORS=uds,c,b,g : explicit curve list (overrides the pileup / jet-less choice above)
if os.environ.get('FLAVORS'):
    flavors = [f.strip() for f in os.environ['FLAVORS'].split(',') if f.strip()]
flavorMembers = {
    "uds": ["u", "d", "s"],
    "c": ["c"],
    "b": ["b"],
    "g": ["g"],
    "pileup": ["pileup"],
    "undef": ["unmatched"],
}
# shared flavor palette -- matches the original gen-flavor reference plots
# (plots/QCD/muon/gen/*.pdf, 2026-07-28): b=dark red, c=orange, uds=azure, g=black
flavorColors = {
    "uds": ROOT.kAzure + 1,     # 861
    "c": ROOT.kOrange + 1,      # 801
    "b": ROOT.kRed + 2,         # 634
    "g": ROOT.kBlack,           # 1
    "pileup": ROOT.kOrange + 7, # 807
    "undef": ROOT.kGray + 2,    # 922
}
flavorLabels = {"b": "b jet", "c": "c jet", "uds": "u/d/s jet", "g": "gluon jet",
                "pileup": "pileup jet", "undef": "jet-less"}

etaBins = [0.0, 0.9, 1.6, 2.4]

# ---- SCHEME=nn3 draw pass: per flavour, binned map vs neural sv3 vs truth, from nn/_cells.json ----
NN_LABEL = {"b": "b jet", "c": "c jet", "uds": "u/d/s jet", "g": "gluon jet", "jetless": "jet-less"}
NN_COL = {"bin": ROOT.kBlack, "nn": ROOT.kAzure + 1}
NN_MIN_NEFF = float(os.environ.get('NN_MIN_NEFF', '3'))


def nn3_curves(J, fl, sl):
    """Graphs {bin, nn, band} for one flavour / |eta| slice (cells with N_eff < NN_MIN_NEFF are skipped)."""
    gb, gn, gband = (ROOT.TGraphAsymmErrors() for _ in range(3))
    edges = J["pt_edges"]
    for i, c in enumerate(J["cells"][fl][str(sl)]):
        if c["sw"] <= 0 or c["sw2"] <= 0 or c["sw"] ** 2 / c["sw2"] < NN_MIN_NEFF:
            continue
        x = 0.5 * (edges[i] + edges[i + 1]); hw = 0.5 * (edges[i + 1] - edges[i])
        yb = c["swb"] / c["sw"]; yn = c["swn"] / c["sw"]
        sd = max(c["swn2"] / c["sw"] - yn ** 2, 0.) ** 0.5
        for g, y, ey in ((gb, yb, 0.), (gn, yn, 0.), (gband, yn, sd)):
            n = g.GetN(); g.SetPoint(n, x, y); g.SetPointError(n, hw, hw, ey, ey)
    gband.SetFillColorAlpha(NN_COL["nn"], 0.30); gband.SetLineColorAlpha(NN_COL["nn"], 0.0)
    gn.SetLineColor(NN_COL["nn"]); gn.SetMarkerColor(NN_COL["nn"]); gn.SetLineWidth(2)
    gn.SetMarkerStyle(20); gn.SetMarkerSize(0.8)
    gb.SetLineColor(NN_COL["bin"]); gb.SetMarkerColor(NN_COL["bin"]); gb.SetLineWidth(2)
    gb.SetLineStyle(2); gb.SetMarkerStyle(25); gb.SetMarkerSize(0.8)
    return dict(bin=gb, nn=gn, band=gband)


def nn3_draw_panel(J, fl, sl):
    """Frame + curves in the current pad (house style of plots/wp_sip)."""
    G = nn3_curves(J, fl, sl)
    ROOT.gPad.SetGrid()
    fr = ROOT.gPad.DrawFrame(10, 0, 100, 1.0)
    fr.GetYaxis().SetTitle("fake rate (#mu)"); fr.GetXaxis().SetTitle("p_{T}^{corr} [GeV]")
    fr.GetXaxis().SetTitleSize(0.04); fr.GetYaxis().SetTitleSize(0.04); fr.GetYaxis().SetTitleOffset(0.9)
    G["band"].Draw("2 same"); G["bin"].Draw("P same");
    return G, fr


def nn3_legend(G, x1=0.60, y1=0.70, x2=0.90, y2=0.88, size=0.032):
    lg = ROOT.TLegend(x1, y1, x2, y2); lg.SetFillStyle(0); lg.SetBorderSize(0); lg.SetTextSize(size)
    lg.AddEntry(G["bin"], "Binned map", "lp")
    lg.AddEntry(G["band"], "Neural Network", "fl")
    lg.Draw()
    return lg


def nn3_draw():
    import json
    J = json.load(open(os.path.join(NN_OUT, "_cells.json")))
    slices = [(0, "etaInt", ""), (1, "eta1", "0.0 < |#eta| < 0.9"), (2, "eta2", "0.9 < |#eta| < 1.6"),
              (3, "eta3", "1.6 < |#eta| < 2.4")]
    keep = []
    cms_ = TLatex(); cms_.SetTextSize(0.045); cms_.SetTextFont(61)
    lab_ = TLatex(); lab_.SetTextSize(0.033); lab_.SetTextFont(42); lab_.SetTextAlign(31)
    for fl in NN_FLAV:
        for sl, tag, etatxt in slices:
            c = TCanvas(f"nn_{fl}_{sl}", "", 800, 600)
            c.SetGrid(); c.SetLeftMargin(0.12); c.SetTopMargin(0.08)
            G, fr = nn3_draw_panel(J, fl, sl)
            lg = nn3_legend(G)
            cms_.DrawLatexNDC(0.12, 0.93, "CMS")
            lab_.DrawLatexNDC(0.92, 0.93, f"2024 (v15), QCD MuEnriched, {NN_LABEL[fl]}")
            if etatxt:
                TLatex().DrawLatexNDC(0.15, 0.84, etatxt)
            c.RedrawAxis()
            c.SaveAs(os.path.join(NN_OUT, f"{fl}_{tag}.png"))
            keep.append((G, fr, lg))
    # one overview per |eta| slice: the five flavours side by side
    for sl, tag, etatxt in slices:
        c = TCanvas(f"nngrid_{sl}", "", 1500, 900); c.Divide(3, 2, 0.001, 0.001)
        for i, fl in enumerate(NN_FLAV):
            c.cd(i + 1); ROOT.gPad.SetLeftMargin(0.13); ROOT.gPad.SetBottomMargin(0.13); ROOT.gPad.SetTopMargin(0.09)
            G, fr = nn3_draw_panel(J, fl, sl)
            tt = TLatex(); tt.SetTextFont(62); tt.SetTextSize(0.06); tt.DrawLatexNDC(0.15, 0.93, NN_LABEL[fl])
            keep.append((G, fr, tt))
        c.cd(6)
        G, fr = nn3_draw_panel(J, "b", sl); ROOT.gPad.Clear()
        lg = nn3_legend(G, 0.05, 0.62, 0.95, 0.85, 0.06)
        tx = TLatex(); tx.SetTextSize(0.05); tx.SetTextFont(42)
        tx.DrawLatexNDC(0.08, 0.45, "2024 (v15), QCD MuEnriched"); tx.DrawLatexNDC(0.08, 0.38, etatxt or "|#eta| integrated")
        keep.append((G, fr, lg, tx))
        c.SaveAs(os.path.join(NN_OUT, f"grid_{tag}.png"))


if scheme == 'nn3':
    nn3_draw()
    sys.exit(0)


def map2d_draw():
    """SCHEME=map2d: the flavour-inclusive binned FR map (conePt x |eta|), one printed value per cell, from _map.json."""
    import json
    J = json.load(open(os.path.join(os.path.dirname(NN_OUT), "_map.json")))
    pe, ee, fm = J["pt_edges"], J["eta_edges"], J["fmap"]
    h = ROOT.TH2D("frmap", "", len(pe) - 1, 0, len(pe) - 1, len(ee) - 1, 0, len(ee) - 1)
    for i in range(len(pe) - 1):
        h.GetXaxis().SetBinLabel(i + 1, f"{int(pe[i])}-{int(pe[i + 1])}")
        for j in range(len(ee) - 1):
            h.SetBinContent(i + 1, j + 1, fm[i][j])
    for j in range(len(ee) - 1):
        h.GetYaxis().SetBinLabel(j + 1, f"{ee[j]} - {ee[j + 1]}")
    h.GetXaxis().SetTitle("p_{T}^{corr} [GeV]"); h.GetYaxis().SetTitle("|#eta|")
    h.GetXaxis().SetTitleSize(0.045); h.GetYaxis().SetTitleSize(0.045)
    h.GetXaxis().SetLabelSize(0.045); h.GetYaxis().SetLabelSize(0.05)
    h.GetYaxis().SetTitleOffset(1.25); h.GetXaxis().SetTitleOffset(1.0)
    h.SetMinimum(0); h.SetMaximum(0.6); h.SetMarkerSize(1.6)
    ROOT.gStyle.SetPaintTextFormat(".3f")
    c = TCanvas("frmap_c", "", 900, 600)
    c.SetLeftMargin(0.12); c.SetRightMargin(0.14); c.SetTopMargin(0.08); c.SetBottomMargin(0.13)
    h.Draw("COLZ TEXT")
    cms_ = TLatex(); cms_.SetTextSize(0.045); cms_.SetTextFont(61); cms_.DrawLatexNDC(0.12, 0.93, "CMS")
    lab_ = TLatex(); lab_.SetTextSize(0.033); lab_.SetTextFont(42); lab_.SetTextAlign(31)
    lab_.DrawLatexNDC(0.86, 0.93, "2024 (v15), QCD MuEnriched, binned fake rate")
    out = os.path.join(os.path.dirname(NN_OUT), "FR_binned.png")
    c.SaveAs(out)




def map1d_draw():
    """SCHEME=map1d: the binned FR map as FR vs conePt, one line per |eta| slice (azure / orange / dark red = uds / c / b colours),
    from _map.json (SCHEME=map1d NNPASS=dump writes it)."""
    import json
    J = json.load(open(os.path.join(os.path.dirname(NN_OUT), "_map.json")))
    pe, ee, fm = J["pt_edges"], J["eta_edges"], J["fmap"]
    cols = [flavorColors["uds"], flavorColors["c"], flavorColors["b"]]   # flavour-plot colours: azure / orange / dark red
    c = TCanvas("frmap1d", "", 800, 600)
    c.SetGrid(); c.SetLeftMargin(0.12); c.SetTopMargin(0.08)
    fr = c.DrawFrame(10, 0, 100, 0.9)
    fr.GetYaxis().SetTitle("fake rate (#mu)"); fr.GetXaxis().SetTitle("p_{T}^{corr} [GeV]")
    fr.GetXaxis().SetTitleSize(0.04); fr.GetYaxis().SetTitleSize(0.04); fr.GetYaxis().SetTitleOffset(1.0)
    lg = TLegend(0.60, 0.68, 0.90, 0.88); lg.SetFillStyle(0); lg.SetBorderSize(0); lg.SetTextSize(0.034)
    keep = []
    for j in range(len(ee) - 1):
        g = TGraphAsymmErrors()
        for i in range(len(pe) - 1):
            hw = 0.5 * (pe[i + 1] - pe[i])
            g.SetPoint(i, pe[i] + hw, fm[i][j]); g.SetPointError(i, hw, hw, 0, 0)
        g.SetLineColor(cols[j]); g.SetMarkerColor(cols[j]); g.SetLineWidth(2)
        g.SetMarkerStyle(20); g.SetMarkerSize(0.8)
        g.Draw("P same")
        lg.AddEntry(g, f"{ee[j]:.1f} < |#eta| < {ee[j + 1]:.1f}", "lp")
        keep.append(g)
    lg.Draw()
    cms_ = TLatex(); cms_.SetTextSize(0.045); cms_.SetTextFont(61); cms_.DrawLatexNDC(0.12, 0.93, "CMS")
    lab_ = TLatex(); lab_.SetTextSize(0.033); lab_.SetTextFont(42); lab_.SetTextAlign(31)
    lab_.DrawLatexNDC(0.92, 0.93, "2024 (v15), QCD MuEnriched")
    c.RedrawAxis()
    c.SaveAs(os.path.join(os.path.dirname(NN_OUT), "FR_binned_curves.png"))


if scheme == 'map2d':
    map2d_draw()
    sys.exit(0)
if scheme == 'map1d':
    map1d_draw()
    sys.exit(0)

# era dir is like 2016preVFP_flavor / 2022EE_flavor_ttlj -> era name is the head.
# For a stack, ERA_LABEL names the combination (e.g. Run2) for filename + legend.
eraLabel = os.environ.get('ERA_LABEL') or "_".join(e.split("_")[0] for e in eras)
# TAG: optional filename suffix ({eraLabel}_{TAG}_etaInt.pdf) -- used to keep the
# jetIdx-matched inputs apart from the older dR<0.4 ones without touching the legend.
fileTag = os.environ.get('TAG', '')
fileTag = f"_{fileTag}" if fileTag else ""
# How the source jet was matched, read off the input dir names: the *_jetidx dirs
# come from the current analyzer (NanoAOD JetIdx pointer), the others from the
# older dR<0.4 nearest-jet matching -- which changes what "jet-less" means.
_isJetIdx = [e.endswith("_jetidx") for e in eras]
matchNote = "jetIdx" if all(_isJetIdx) else ("dR<0.4" if not any(_isJetIdx) else "mixed")

# sample: TTLJ for the ttbar flavor run, DY for the DY+jets run, else QCD MuEnriched.
# DY comes out of the same {era}_{flavor,eflavor}[_jetidx] dirs as QCD -- the
# analyzer writes DYJets*.root beside the QCD files and hadd.sh merges them
# into DY.root -- so only the sample name distinguishes them.
if "ttlj" in era.lower() or "TTLJ" in sample:
    sampleTag = "TTLJ"
elif sample.startswith("DY"):
    sampleTag = "DY"
else:
    sampleTag = "QCD"                                         # top-level plot folder
sampleLabel = (os.environ.get('SAMPLE_LABEL')
               or {"TTLJ": "TTLJ", "DY": "DY+jets"}.get(sampleTag, "QCD MuEnriched"))

# SOURCE=tree: build the same (path, wp, flavor) 2D histograms offline from the MakeTree
# ntuples (tree "mu") instead of reading the analyzer's histogram files.  This is how an
# ID that the analyzer histograms were never filled for gets drawn: the "new ID" case
# (loose |SIP3D|<5, tight |SIP3D|<3, no miniIso) lives in the MatchWP,LooseNoIso trees
# (tags nff2ni = Run 2, nff3ni = Run 3), with the tight cut re-applied here (TIGHT).
#   TREEBASE  dir holding {era}_{tag} subdirs (default MakeTree_MatchWP_LooseNoIso)
#   ERAS      bare era names (2017, 2022EE ...) pick {era}_nff2ni / {era}_nff3ni;
#             a name with an underscore (2017_nff) is used as given under TREEBASE
#   TIGHT     RDataFrame expression for tight (default fabs(mu_sip3d) < 3;
#             TIGHT=isTight reads the stored old-WP flag -- the reproduction check)
#   OUTROOT   plots/<OUTROOT> (default newid); may be an absolute path
# conePt stays the tree's mu_conept = pt*(1+max(0, miniIso-0.1)) (neuralff convention).
source = os.environ.get('SOURCE', 'hist')
if source not in ('hist', 'tree'):
    raise ValueError(f"SOURCE must be 'hist' or 'tree', got {source!r}")
if source == 'tree' and xvar != 'ptcorr':
    raise ValueError("SOURCE=tree only supports XVAR=ptcorr")
treeBase = os.environ.get('TREEBASE', '/gv0/Users/snuintern2/SKNanoOutput/fake/MakeTree_MatchWP_LooseNoIso')
treeTight = os.environ.get('TIGHT', 'fabs(mu_sip3d) < 3')
outRoot = os.environ.get('OUTROOT', 'newid')

# plots/Jetless/{QCD,TTLJ}/muon/{gen,reco}/{era}_{etaInt,eta1..3}.png (unchanged, JETLESS.md references this tree)
# plots/cone/{QCD,TTLJ}/muon/{gen,reco}/...  (XVAR=ptcorr, muon's own cone-corrected pT)
# plots/jet/{QCD,TTLJ}/muon/{gen,reco}/...   (XVAR=genjetpt|recojetpt, matched jet's pT)
# plots/<OUTROOT>/{QCD,TTLJ}/muon/{gen,reco}/...  (SOURCE=tree)
schemeDir = "gen" if scheme == "genflav" else "reco"
scriptDir = os.path.dirname(os.path.abspath(__file__))
if source == 'tree':
    plotRoot = os.path.join(scriptDir, "plots", outRoot)
elif jetless:
    plotRoot = os.path.join(scriptDir, "plots", "Jetless")
else:
    binRoot = "jet" if xvar in ("genjetpt", "recojetpt") else "cone"
    plotRoot = os.path.join(scriptDir, "plots", binRoot)
outdir = os.path.join(plotRoot, sampleTag, "muon", schemeDir)
os.makedirs(outdir, exist_ok=True)

_files = []
if source == 'hist':
    for e in eras:
        fkey = f"{os.environ.get('FAKE_ROOT', '/data6/Users/snuintern2/fake')}/{e}/{sample}.root"
        if not os.path.exists(fkey):
            raise FileNotFoundError(f"File does not exist: {fkey}")
        fh = ROOT.TFile.Open(fkey)
        if not fh or fh.IsZombie():
            raise IOError(f"Cannot open ROOT file: {fkey}")
        _files.append(fh)

ptCorrEdges = [10., 12., 14., 17., 20., 30., 50., 100., 200.]   # = fake.cc ptCorrBins
absEtaEdges = [0., 0.9, 1.6, 2.4]                               # = fake.cc absEtaBins
treeFlavorCode = {"b": 0, "c": 1, "uds": 2, "g": 3, "pileup": 4,   # fake.cc FlavorCode
                  "undef": -1}                                     # = unmatched / jet-less
treePathCols = {"Mu8": ("pass_mu8", "w_mu8"), "Mu17": ("pass_mu17", "w_mu17")}


treeLayout = os.environ.get('TREELAYOUT', 'v13')   # v15: TREEBASE/<era>/<sample>.root (SKNano 2.0 output, RNTuple "mu")


def _tree_dir(e):
    if treeLayout == 'v15':
        return os.path.join(treeBase, e)
    if "_" in e:
        return os.path.join(treeBase, e)
    run3 = e[:4] in ("2022", "2023")
    return os.path.join(treeBase, f"{e}_{'nff3ni' if run3 else 'nff2ni'}")


def build_tree_hists():
    # MeasReg/Central selection of the analyzer, offline: non-prompt MC, MET<25, MT<25,
    # away jet 40.  Trigger paths are picked with the pass_* flags (never w>0: that drops
    # the negative-weight events), each with its own trigger-lumi weight.
    pattern = {"QCD": "QCD_Pt-*_MuEnriched.root", "TTLJ": "TTLJ_powheg.root"}.get(sampleTag)
    if treeLayout == 'v15' and sampleTag == "QCD":
        pattern = "QCD_*MuEnriched*.root"
    if pattern is None:
        raise ValueError(f"SOURCE=tree has no file pattern for sample {sampleTag}")
    files = []
    for e in eras:
        found = sorted(glob.glob(os.path.join(_tree_dir(e), pattern)))
        if not found:
            raise FileNotFoundError(f"No {pattern} under {_tree_dir(e)}")
        files += found
    ROOT.EnableImplicitMT(4)
    flavCol = os.environ.get('FLAVCOL') or ("flavor" if scheme == "recoflav" else "genflavor")   # FLAVCOL=flavor_dr: dR<0.4 jet instead of jetIdx
    df = (ROOT.RDataFrame("mu", files)
          .Filter("!isData && !isPrompt && ev_met < 25 && ev_mt < 25 && ev_hasAway40", "meas")
          .Define("xc", f"std::min((double)mu_conept, {ptCorrEdges[-1] - 0.1})")
          .Define("isT", treeTight))
    res = {}
    for p, (passCol, wCol) in treePathCols.items():
        dp = df.Filter(passCol, p)
        for fl, code in treeFlavorCode.items():
            if fl not in flavors:
                continue
            dfl = dp.Filter(f"{flavCol} == {code}")
            for wp, d2 in (("loose", dfl), ("tight", dfl.Filter("isT"))):
                h = ROOT.TH2D(f"tree_{p}_{wp}_{fl}", "", len(ptCorrEdges) - 1, array('d', ptCorrEdges),
                              len(absEtaEdges) - 1, array('d', absEtaEdges))
                h.Sumw2()
                res[(p, wp, fl)] = d2.Histo2D(ROOT.RDF.TH2DModel(h), "xc", "mu_abseta", wCol)
    return res


_treeHists = build_tree_hists() if source == 'tree' else None


def get_hist(path, wp, flavor):
    if _treeHists is not None:
        h = _treeHists[(path, wp, flavor)].GetValue()
        if h.GetEntries() == 0:
            return None
        h = h.Clone(f"{path}_{wp}_{flavor}_{scheme}")
        h.SetDirectory(0)
        return h
    # sum the member flavors (uds = u+d+s) over every era in the stack;
    # returns None if no era has any of them
    h = None
    for fh in _files:
        for m in flavorMembers[flavor]:
            hm = fh.Get(f"{path}/MeasReg/Central/{wp}/{scheme}/{m}/{histSuffix}")
            if not hm:
                continue
            if h is None:
                h = hm.Clone(f"{path}_{wp}_{flavor}_{scheme}")
                h.SetDirectory(0)
            else:
                h.Add(hm)
    return h


def clopper_pearson(k, n, cl=0.68):
    # exact binomial interval; k=0 gives [0, hi] and k=n gives [lo, 1] instead
    # of the zero-length bar that Divide(...,"B") returns in those cases
    if n <= 0:
        return (0.0, 1.0)
    k = min(max(k, 0), n)
    lo = 0.0 if k == 0 else ROOT.Math.beta_quantile((1 - cl) / 2, k, n - k + 1)
    hi = 1.0 if k >= n else ROOT.Math.beta_quantile(1 - (1 - cl) / 2, k + 1, n - k)
    return (lo, hi)


def make_ratio1D(loose2D, tight2D, iy):
    # iy in [1..nEta] projects one |eta| slice; iy=0 integrates over all eta.
    # Returns the raw projections; the ratio and its errors are built per bin in
    # make_combined_graph so that starved bins can be suppressed rather than
    # drawn as a fake 0.000 +- 0.000 point.
    if iy == 0:
        loose1D = loose2D.ProjectionX(f"l_{loose2D.GetName()}_all", 1, loose2D.GetNbinsY())
        tight1D = tight2D.ProjectionX(f"t_{tight2D.GetName()}_all", 1, tight2D.GetNbinsY())
    else:
        loose1D = loose2D.ProjectionX(f"l_{loose2D.GetName()}_{iy}", iy, iy)
        tight1D = tight2D.ProjectionX(f"t_{tight2D.GetName()}_{iy}", iy, iy)
    return loose1D, tight1D


def make_combined_graph(hists, iy):
    # stitch Mu8 (low ptcorr) + Mu17 (high ptcorr) into one FR(ptcorr) graph.
    # A flavor may lack a hist for one path (no MC entries, e.g. sparse light
    # flavor at high pt in TTLJ) -> that path's ratio is None; skip those bins.
    # If only one path was actually measured at all (e.g. the Mu17-only,
    # conePt-cut-free jet-pT study), use that one path across the full range
    # instead of skipping every bin the (absent) other path would have owned.
    ratios = {}
    for p in paths:
        if hists[p]["loose"] is None or hists[p]["tight"] is None:
            ratios[p] = None
        else:
            ratios[p] = make_ratio1D(hists[p]["loose"], hists[p]["tight"], iy)
    ref = next((r for r in ratios.values() if r is not None), None)
    if ref is None:
        return None
    available = [p for p in paths if ratios[p] is not None]

    g = TGraphAsymmErrors()
    for ix in range(1, ref[0].GetNbinsX() + 1):
        upedge = ref[0].GetXaxis().GetBinUpEdge(ix)
        if "Mu8" in available and "Mu17" in available:
            pair = ratios["Mu8"] if upedge <= switchPt else ratios["Mu17"]
        elif available:
            pair = ratios[available[0]]
        else:
            pair = None
        if pair is None:
            continue
        loose1D, tight1D = pair
        L, Lerr = loose1D.GetBinContent(ix), loose1D.GetBinError(ix)
        # N_eff = (sum w)^2 / (sum w^2): the events actually behind this bin.
        # A weighted yield can look healthy while resting on one MC event.
        nEff = (L / Lerr) ** 2 if Lerr > 0 else 0.
        if L <= 0 or nEff < minNeff:
            continue                      # no measurement here -- draw nothing
        y = tight1D.GetBinContent(ix) / L
        n_i = int(round(nEff))
        lo, hi = clopper_pearson(int(round(min(max(y, 0.), 1.) * n_i)), n_i)
        x = loose1D.GetXaxis().GetBinCenter(ix)
        xlo = x - loose1D.GetXaxis().GetBinLowEdge(ix)
        xhi = loose1D.GetXaxis().GetBinUpEdge(ix) - x
        n = g.GetN()
        g.SetPoint(n, x, y)
        g.SetPointError(n, xlo, xhi, max(0., y - lo), max(0., hi - y))
    return g


cms = TLatex(); cms.SetTextSize(0.045); cms.SetTextFont(61)
label = TLatex(); label.SetTextSize(0.033); label.SetTextFont(42); label.SetTextAlign(31)


def draw_one(iy, tag_suffix, eta_text):
    # build one overlay canvas: all flavors for the given eta selection
    c = TCanvas(f"c_{scheme}_{iy}", "", 800, 600)
    c.SetGrid(); c.SetLeftMargin(0.12); c.SetTopMargin(0.08)
    legend = TLegend(0.60, 0.60, 0.90, 0.88)
    legend.SetFillStyle(0); legend.SetBorderSize(0); legend.SetTextSize(0.032)

    graphs = []
    for flavor in flavors:
        hists = {p: {wp: get_hist(p, wp, flavor) for wp in ("loose", "tight")} for p in paths}
        g = make_combined_graph(hists, iy)
        if g is None or g.GetN() == 0:
            continue
        g.SetLineColor(flavorColors[flavor]); g.SetMarkerColor(flavorColors[flavor])
        g.SetMarkerStyle(20); g.SetMarkerSize(0.8); g.SetLineWidth(2)
        if flavor == "undef":
            g.SetLineStyle(2)      # undefined is the only non-solid category
        graphs.append((flavor, g))
        legend.AddEntry(g, flavorLabels[flavor], "lep")

    frame = c.DrawFrame(10, 0, 100, 1.0)
    xTitle = {"ptcorr": "p_{T}^{corr} [GeV]",
              "genjetpt": "p_{T}^{gen jet} [GeV]",
              "recojetpt": "p_{T}^{reco jet} [GeV]"}[xvar]
    frame.GetXaxis().SetTitle(xTitle)
    frame.GetYaxis().SetTitle("fake rate (#mu)")
    frame.GetXaxis().SetTitleSize(0.04); frame.GetYaxis().SetTitleSize(0.04)
    frame.GetYaxis().SetTitleOffset(0.9)
    for _, g in graphs:
        g.Draw("P same")
    legend.Draw()
    cms.DrawLatexNDC(0.12, 0.93, "CMS")
    schemeName = "gen-jet" if scheme == "genflav" else "reco-jet"
    if os.environ.get('FLAVCOL') == "flavor_dr":
        schemeName = "reco-jet (dR<0.4)"
    note = f" ({matchNote})" if (jetless and scheme == "recoflav" and source == "hist") else ""
    label.DrawLatexNDC(0.92, 0.93, f"{eraLabel}, {sampleLabel}, {schemeName} flavor{note}")
    if eta_text:
        TLatex().DrawLatexNDC(0.15, 0.84, eta_text)
    wpNote = os.environ.get('WP_NOTE', "loose |SIP3D|<5, tight |SIP3D|<3, no miniIso")
    if source == 'tree' and wpNote:                 # WP_NOTE='' drops the line; ';' starts a new line
        note = TLatex(); note.SetTextSize(0.028); note.SetTextFont(42)
        for k, line in enumerate(wpNote.split(';')):   # top left, under the |eta| label, clear of the low-pT points
            note.DrawLatexNDC(0.15, 0.79 - 0.04 * k, line.strip())
    c.RedrawAxis()
    outname = os.path.join(outdir, f"{eraLabel}{fileTag}_{tag_suffix}.png")
    c.SaveAs(outname)
    print(f"Saved {outname}")


# eta-integrated headline
draw_one(0, "etaInt", "")
# per |eta| slice
for iy in range(1, len(etaBins)):
    eta_text = f"{etaBins[iy-1]:.1f} < |#eta| < {etaBins[iy]:.1f}"
    draw_one(iy, f"eta{iy}", eta_text)

for fh in _files:
    fh.Close()
