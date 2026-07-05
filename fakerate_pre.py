import ROOT
import os
from ROOT import TCanvas, TLatex, TLegend, TGraphAsymmErrors
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# set global variables
era = os.environ.get('ERA', '2016preVFP_2')
sample = os.environ.get('SAMPLE', 'QCD')
paths = ["Mu8", "Mu17"]

etaBinColors = [ROOT.kAzure + 1, ROOT.kOrange + 1, ROOT.kRed]

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yes/33")
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


def make_ratio1D(loose2D, tight2D, iy):
    loose1D = loose2D.ProjectionX(f"loose_proj_{iy}_{loose2D.GetName()}", iy, iy)
    tight1D = tight2D.ProjectionX(f"tight_proj_{iy}_{tight2D.GetName()}", iy, iy)

    ratio = tight1D.Clone(f"ratio_{iy}_{tight2D.GetName()}")
    ratio.Divide(tight1D, loose1D, 1, 1, "B")
    return ratio


# Mu8 covers the low-pT region, Mu17 covers the high-pT region;
# switchPt is the ptCorr edge where we hand off from one trigger to the other
switchPt = 30.


def make_combined_graph(hists, iy):
    ratio_Mu8 = make_ratio1D(hists["Mu8"]["loose"], hists["Mu8"]["tight"], iy)
    ratio_Mu17 = make_ratio1D(hists["Mu17"]["loose"], hists["Mu17"]["tight"], iy)

    g = TGraphAsymmErrors()
    for ix in range(1, ratio_Mu8.GetNbinsX() + 1):
        upedge = ratio_Mu8.GetXaxis().GetBinUpEdge(ix)
        ratio = ratio_Mu8 if upedge <= switchPt else ratio_Mu17

        x = ratio.GetXaxis().GetBinCenter(ix)
        xlo = x - ratio.GetXaxis().GetBinLowEdge(ix)
        xhi = ratio.GetXaxis().GetBinUpEdge(ix) - x
        y = ratio.GetBinContent(ix)
        ey = ratio.GetBinError(ix)
        n = g.GetN()
        g.SetPoint(n, x, y)
        g.SetPointError(n, xlo, xhi, ey, ey)
    return g


cms = TLatex()
cms.SetTextSize(0.045)
cms.SetTextFont(61)
preliminary = TLatex()
preliminary.SetTextSize(0.035)
preliminary.SetTextFont(52)
label = TLatex()
label.SetTextSize(0.035)
label.SetTextFont(42)
label.SetTextAlign(31)

etaBins = [0.0, 0.9, 1.6, 2.4]

hists = {path: {"loose": get_hist(path, "loose"), "tight": get_hist(path, "tight")} for path in paths}

c = TCanvas("c_combined", "", 800, 600)
c.SetGrid()
c.SetLeftMargin(0.12)
c.SetTopMargin(0.08)

legend = TLegend(0.55, 0.72, 0.90, 0.88)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.03)

graphs = []
for iy in range(1, 4):
    g = make_combined_graph(hists, iy)
    g.SetLineColor(etaBinColors[iy - 1])
    g.SetMarkerColor(etaBinColors[iy - 1])
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.7)
    g.SetLineWidth(1)
    graphs.append(g)
    legend.AddEntry(g, f"{etaBins[iy-1]:.1f} < |#eta| < {etaBins[iy]:.1f}", "lep")

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
label.DrawLatexNDC(0.92, 0.93, f"2016preVFP, Mu8+Mu17 Prescaled (13TeV)")

c.RedrawAxis()

outname = os.path.join(outdir, f"FakeRate1D_combined_{sample}_{era}.png")
c.SaveAs(outname)
print(f"Saved {outname}")
