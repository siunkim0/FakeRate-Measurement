import ROOT
import os
from ROOT import TCanvas, TLatex
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# set global variables
era = os.environ.get('ERA', '2016preVFP_2')
sample = os.environ.get('SAMPLE', 'QCD')
paths = ["Mu8", "Mu17"]

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yes/2d")
os.makedirs(outdir, exist_ok=True)


def get_hist(path, wp):
    fkey = f"/data6/Users/snuintern2/fake/{era}/{sample}.root"
    if not os.path.exists(fkey):
        raise FileNotFoundError(f"File does not exist: {fkey}")

    f = ROOT.TFile.Open(fkey)
    if not f or f.IsZombie():
        raise IOError(f"Cannot open ROOT file: {fkey}")

    histkey = f"{path}/MeasReg/Central/{wp}/fake_ptcorr_abseta"
    h = f.Get(histkey)
    if not h:
        raise ValueError(f"Histogram '{histkey}' not found in {fkey}")

    h.SetDirectory(0)
    f.Close()
    return h


def make_ratio2D(loose2D, tight2D, name):
    ratio = tight2D.Clone(name)
    ratio.SetDirectory(0)
    ratio.Divide(tight2D, loose2D, 1, 1, "B")
    return ratio


# Mu8 covers the low-pT region, Mu17 covers the high-pT region;
# switchPt is the ptCorr edge where we hand off from one trigger to the other
switchPt = 30.

hists = {path: {"loose": get_hist(path, "loose"), "tight": get_hist(path, "tight")} for path in paths}

ratio_Mu8 = make_ratio2D(hists["Mu8"]["loose"], hists["Mu8"]["tight"], "ratio_Mu8")
ratio_Mu17 = make_ratio2D(hists["Mu17"]["loose"], hists["Mu17"]["tight"], "ratio_Mu17")

# stitched 2D FR: this is the histogram bkg.cc reads (name must stay "FR")
fr = ratio_Mu8.Clone("FR")
fr.SetDirectory(0)
fr.SetTitle("")
for ix in range(1, fr.GetNbinsX() + 1):
    upedge = fr.GetXaxis().GetBinUpEdge(ix)
    ratio = ratio_Mu8 if upedge <= switchPt else ratio_Mu17
    for iy in range(1, fr.GetNbinsY() + 1):
        fr.SetBinContent(ix, iy, ratio.GetBinContent(ix, iy))
        fr.SetBinError(ix, iy, ratio.GetBinError(ix, iy))

# save the ROOT file for bkg.cc
outroot = f"/data6/Users/snuintern2/fake/{era}/FR2D_{sample}.root"
fout = ROOT.TFile(outroot, "RECREATE")
fr.Write("FR")
fout.Close()
print(f"Saved {outroot}")

# draw the 2D map for validation
cms = TLatex()
cms.SetTextSize(0.045)
cms.SetTextFont(61)
label = TLatex()
label.SetTextSize(0.035)
label.SetTextFont(42)
label.SetTextAlign(31)

c = TCanvas("c_fr2d", "", 800, 600)
c.SetLeftMargin(0.12)
c.SetTopMargin(0.08)
c.SetRightMargin(0.13)
c.SetLogx()   # low-pT bins are narrow; log axis keeps the bin texts readable

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
label.DrawLatexNDC(0.87, 0.93, f"{era.replace('_2', '')}, Mu8+Mu17 Prescaled (13TeV)")

c.RedrawAxis()

outname = os.path.join(outdir, f"FakeRate2D_{sample}_{era}.png")
c.SaveAs(outname)
print(f"Saved {outname}")
