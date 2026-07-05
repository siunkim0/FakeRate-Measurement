import ROOT
import os
from ROOT import TCanvas, TLatex, TLegend, TGraphAsymmErrors
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# set global variables
era = os.environ.get('ERA', '2016preVFP_data')
data_sample = os.environ.get('DATA', 'DoubleMuon')
prompt_samples = ["DY", "WJets"]   # prompt subtraction에 쓸 MC
paths = ["Mu8", "Mu17"]

etaBinColors = [ROOT.kAzure + 1, ROOT.kOrange + 1, ROOT.kRed]

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yes/data")
os.makedirs(outdir, exist_ok=True)


def get_hist(sample, histkey, allow_missing=False):
    fkey = f"/data6/Users/snuintern2/fake/{era}/{sample}.root"
    if not os.path.exists(fkey):
        raise FileNotFoundError(f"File does not exist: {fkey}")

    f = ROOT.TFile.Open(fkey)
    if not f or f.IsZombie():
        raise IOError(f"Cannot open ROOT file: {fkey}")

    h = f.Get(histkey)
    if not h:
        f.Close()
        if allow_missing:
            return None
        raise ValueError(f"Histogram '{histkey}' not found in {fkey}")

    h.SetDirectory(0)
    f.Close()
    return h


# -------------------------------------------------------------------
# Step 1: ZEnriched 영역에서 trigger path 별 MC normalization 추출
#   MC 는 analyzer 에서 full lumi 로 normalize 되어 있으므로 (prescale 미반영),
#   scale = N_data(Z peak) / N_MC(Z peak) 가 그 path 의 유효 prescaled lumi 비율
# -------------------------------------------------------------------
scales = {}
for path in paths:
    n_data = get_hist(data_sample, f"{path}/ZEnriched/mll").Integral()
    n_mc = 0.
    for sample in prompt_samples:
        h = get_hist(sample, f"{path}/ZEnriched/mll", allow_missing=True)
        if h:
            n_mc += h.Integral()
    scales[path] = n_data / n_mc
    print(f"[{era}] {path}: N_data(Z) = {n_data:.1f}, N_MC(Z) = {n_mc:.1f}, scale = {scales[path]:.5g}")


# -------------------------------------------------------------------
# Step 2: MeasReg 에서 data - scale * (prompt MC) 로 prompt 오염 제거
#   MC 는 gen-matching 된 prompt 성분만 뺀다 (MC 의 fake 성분은
#   데이터의 fake 에 이미 포함된 것이므로 빼면 안 됨)
# -------------------------------------------------------------------
def make_subtracted(path, wp):
    h = get_hist(data_sample, f"{path}/MeasReg/Central/{wp}/fake_ptcorr_abseta")
    h.SetName(f"sub_{path}_{wp}")
    for sample in prompt_samples:
        hp = get_hist(sample, f"{path}/MeasReg/Central/{wp}/prompt/fake_ptcorr_abseta",
                      allow_missing=True)
        if hp:
            h.Add(hp, -scales[path])

    # subtraction 후 음수 bin 은 0 으로 (통계 요동 / MC 과대평가 보호)
    for ix in range(1, h.GetNbinsX() + 1):
        for iy in range(1, h.GetNbinsY() + 1):
            if h.GetBinContent(ix, iy) < 0:
                print(f"  WARNING: negative bin after subtraction: {path} {wp} "
                      f"(ix={ix}, iy={iy}, {h.GetBinContent(ix, iy):.1f}) -> set to 0")
                h.SetBinContent(ix, iy, 0.)
    return h


def make_ratio2D(loose2D, tight2D, name):
    # 데이터 히스토그램은 weight=1 이라 Sumw2 가 없을 수 있다;
    # 없으면 Divide 후 에러가 sqrt(ratio) 로 잘못 계산되므로 먼저 켜 준다
    for h in (loose2D, tight2D):
        if h.GetSumw2N() == 0:
            h.Sumw2()
    ratio = tight2D.Clone(name)
    ratio.SetDirectory(0)
    ratio.Divide(loose2D)   # subtraction 후엔 binomial 이 아니므로 일반 error propagation
    return ratio


# Mu8 covers the low-pT region, Mu17 covers the high-pT region;
# switchPt is the ptCorr edge where we hand off from one trigger to the other
switchPt = 30.

ratios = {}      # prompt-subtracted FR
ratios_raw = {}  # subtraction 없이 (비교용)
for path in paths:
    loose_sub = make_subtracted(path, "loose")
    tight_sub = make_subtracted(path, "tight")
    ratios[path] = make_ratio2D(loose_sub, tight_sub, f"fr_{path}")

    loose_raw = get_hist(data_sample, f"{path}/MeasReg/Central/loose/fake_ptcorr_abseta")
    tight_raw = get_hist(data_sample, f"{path}/MeasReg/Central/tight/fake_ptcorr_abseta")
    ratios_raw[path] = make_ratio2D(loose_raw, tight_raw, f"frraw_{path}")

# stitched 2D FR: this is the histogram bkg.cc reads (name must stay "FR")
fr = ratios["Mu8"].Clone("FR")
fr.SetDirectory(0)
fr.SetTitle("")
for ix in range(1, fr.GetNbinsX() + 1):
    upedge = fr.GetXaxis().GetBinUpEdge(ix)
    ratio = ratios["Mu8"] if upedge <= switchPt else ratios["Mu17"]
    for iy in range(1, fr.GetNbinsY() + 1):
        fr.SetBinContent(ix, iy, ratio.GetBinContent(ix, iy))
        fr.SetBinError(ix, iy, ratio.GetBinError(ix, iy))

# save the ROOT file for bkg.cc
outroot = f"/data6/Users/snuintern2/fake/{era}/FR2D_DATA.root"
fout = ROOT.TFile(outroot, "RECREATE")
fr.Write("FR")
fout.Close()
print(f"Saved {outroot}")


# -------------------------------------------------------------------
# Step 3: validation plot (1D combined, prompt-subtracted vs raw)
# -------------------------------------------------------------------
def make_combined_graph(rdict, iy, markerstyle):
    g = TGraphAsymmErrors()
    for ix in range(1, rdict["Mu8"].GetNbinsX() + 1):
        upedge = rdict["Mu8"].GetXaxis().GetBinUpEdge(ix)
        ratio = rdict["Mu8"] if upedge <= switchPt else rdict["Mu17"]

        x = ratio.GetXaxis().GetBinCenter(ix)
        xlo = x - ratio.GetXaxis().GetBinLowEdge(ix)
        xhi = ratio.GetXaxis().GetBinUpEdge(ix) - x
        y = ratio.GetBinContent(ix, iy)
        ey = ratio.GetBinError(ix, iy)
        n = g.GetN()
        g.SetPoint(n, x, y)
        g.SetPointError(n, xlo, xhi, ey, ey)
    g.SetMarkerStyle(markerstyle)
    return g


cms = TLatex()
cms.SetTextSize(0.045)
cms.SetTextFont(61)
label = TLatex()
label.SetTextSize(0.035)
label.SetTextFont(42)
label.SetTextAlign(31)

etaBins = [0.0, 0.9, 1.6, 2.4]

c = TCanvas("c_combined", "", 800, 600)
c.SetGrid()
c.SetLeftMargin(0.12)
c.SetTopMargin(0.08)

legend = TLegend(0.45, 0.62, 0.90, 0.88)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.03)
legend.SetNColumns(2)

graphs = []
for iy in range(1, 4):
    g = make_combined_graph(ratios, iy, 20)        # closed: prompt-subtracted
    graw = make_combined_graph(ratios_raw, iy, 24)  # open: no subtraction
    for gr in (g, graw):
        gr.SetLineColor(etaBinColors[iy - 1])
        gr.SetMarkerColor(etaBinColors[iy - 1])
        gr.SetMarkerSize(0.7)
        gr.SetLineWidth(1)
    graw.SetLineStyle(2)
    graphs += [g, graw]
    legend.AddEntry(g, f"{etaBins[iy-1]:.1f} < |#eta| < {etaBins[iy]:.1f} (sub.)", "lep")
    legend.AddEntry(graw, "no subtraction", "lep")

frame = c.DrawFrame(10, 0, 100, 0.9)
frame.GetXaxis().SetTitle("p_{T}^{corr}")
frame.GetYaxis().SetTitle("fake rate (#mu)")
frame.GetXaxis().SetTitleSize(0.04)
frame.GetYaxis().SetTitleSize(0.04)
frame.GetYaxis().SetTitleOffset(0.9)

for g in graphs:
    g.Draw("P same")

legend.Draw()

cms.DrawLatexNDC(0.12, 0.93, "CMS")
label.DrawLatexNDC(0.92, 0.93, f"{era.replace('_data', '')} data, Mu8+Mu17 Prescaled (13TeV)")

c.RedrawAxis()

outname = os.path.join(outdir, f"FakeRate1D_combined_{data_sample}_{era}.png")
c.SaveAs(outname)
print(f"Saved {outname}")

# 2D map
c2 = TCanvas("c_fr2d", "", 800, 600)
c2.SetLeftMargin(0.12)
c2.SetTopMargin(0.08)
c2.SetRightMargin(0.13)
c2.SetLogx()

ROOT.gStyle.SetPaintTextFormat(".3f")
fr.GetXaxis().SetTitle("p_{T}^{corr}")
fr.GetXaxis().SetMoreLogLabels(True)
fr.GetYaxis().SetTitle("|#eta|")
fr.GetXaxis().SetTitleSize(0.04)
fr.GetYaxis().SetTitleSize(0.04)
fr.GetYaxis().SetTitleOffset(0.9)
fr.GetZaxis().SetRangeUser(0., 0.9)
fr.SetMarkerSize(1.2)
fr.Draw("COLZ TEXT E")

cms.DrawLatexNDC(0.12, 0.93, "CMS")
label.DrawLatexNDC(0.87, 0.93, f"{era.replace('_data', '')} data, Mu8+Mu17 Prescaled (13TeV)")

c2.RedrawAxis()

outname = os.path.join(outdir, f"FakeRate2D_{data_sample}_{era}.png")
c2.SaveAs(outname)
print(f"Saved {outname}")
