# Muon Fake Rate Measurement & Matrix Method Background Estimation

Measure the muon fake rate in 2016preVFP / 2016postVFP data,
then use it to estimate the fake-lepton background in the tt-bar
OS dimuon selection with the matrix method.

## References

- **AN-25-154** (`/data6/Users/snuintern2/ref/AN-25-154_temp.pdf`) §6.1.2:
  fake rate measurement method (Table 38: pT ranges per prescaled trigger).
  - Fig 44: FR measured in **data** (with prompt subtraction) ← comparison target for the data measurement
  - Fig 46: FR measured in **QCD MC** ← comparison target for the QCD MC measurement (do not confuse with Fig 44)
- **AN2010/261** (`/data6/Users/snuintern2/ref/AN2010_261_v1.pdf`) §9:
  fakeable object (matrix) method formulas. Prompt rate fixed to p = 1.

## Muon Working Points (fake.cc == bkg.cc, must be identical)

| | common | tight | loose |
|---|---|---|---|
| cuts | POG medium ID, \|dz\| < 0.1 cm, TkRelIso(R03) < 0.4 | \|SIP3D\| < 3, miniIso < 0.1 | \|SIP3D\| < 5, miniIso < 0.6 |

cone-corrected pT = pT × (1 + max(0, miniIso − 0.1)).
FR(ptcorr, |eta|) = N(tight) / N(loose), binning: ptcorr {10,12,14,17,20,30,50,100} × |eta| {0,0.9,1.6,2.4}.

## Analyzers (SKNanoAnalyzer, `Analyzers/src/`)

### fake.cc — FR measurement
- Two prescaled trigger paths: `Mu8` (HLT_Mu8_TrkIsoVVL, covers ptcorr 10–30),
  `Mu17` (HLT_Mu17_TrkIsoVVL, covers ptcorr 30–100). The final FR is stitched at 30 GeV.
- **MeasReg** (measurement region): exactly 1 loose muon + 0 electrons + away jet (pT>40, dR>0.7)
  + MET < 25 + MT < 25. → `{path}/MeasReg/Central/{loose,tight}/fake_ptcorr_abseta`
  (for MC, gen-matching also fills `/prompt` and `/fake` subdirectories)
- **ZEnriched** (normalization region): OS tight pair + |mll−91.2| < 15 + jet pT>40.
  → `{path}/ZEnriched/mll` — used to fix the MC normalization to data,
  accounting for the prescale.

### bkg.cc — matrix method background estimation
- Selection: IsoMu24(||IsoTkMu24) trigger, exactly 2 **loose** muons (OS,
  leading pT>26, subleading>10), 0 electrons, mll>20, Z veto(±15), jets ≥ 2, MET > 40.
- Events are classified TT / TL / LL by the tight decision of the two muons,
  then weighted (p=1, e = f/(1−f)):

  | estimate | leading term | full calculation |
  |---|---|---|
  | Npp | [TT] | [TT] − [TL]·e_fail + [LL]·e₁e₂ |
  | Nfp | [TL]·e_fail | [TL]·e_fail − 2[LL]·e₁e₂ |
  | Nff | [LL]·e₁e₂ | [LL]·e₁e₂ |
  | fake (=Nfp+Nff) | [TL]·e_fail + [LL]·e₁e₂ | [TL]·e_fail − [LL]·e₁e₂ |

- Output: `MuMu/{TT,TL,LL}/...` (observed; for MC an additional `/prompt`,`/fake` truth split),
  `MuMu/matrix/{leading,full}/{Npp,Nfp,Nff,fake}/...` (count, mll, pt, ptcorr, eta, nJets, MET).
  **The integral of `MuMu/matrix/full/fake/count` is the fake background estimate.**
- FR input (selected via userflag):
  - default: `{era}_2/FR2D_QCD.root` (QCD MC FR) — for MC closure test
  - `--userflags DataFR`: `{era}_data/FR2D_DATA.root` (data FR) — for the data estimate

## Directory layout (`/data6/Users/snuintern2/fake/`)

```
2016preVFP_2/ , 2016postVFP_2/     fake.cc output on QCD MuEnriched MC
    QCD_Pt-*_MuEnriched.root       (12 pT-binned samples)
    QCD.root                       hadd result
    FR2D_QCD.root                  fakerate_2d.py output (bkg.cc input)
2016preVFP_data/ , 2016postVFP_data/   fake.cc output on data / prompt MC
    DoubleMuon.root                hadd (DoubleMuon_B..F or F..H)
    DY.root                        hadd (DYJets + DYJets10to50)
    WJets.root
    FR2D_DATA.root                 fakerate_data.py output (bkg.cc input, DataFR)
Tools/                             this directory (post-processing scripts)
    yes/33/, yes/2d/, yes/data/    plot outputs
```

## Tools scripts (activate ROOT with `rot` before running)

| script | input | output | description |
|---|---|---|---|
| `fakerate_pre.py` / `fakerate_post.py` | `{era}/QCD.root` | 1D FR plots | QCD MC FR, 1D per eta bin (Mu8+Mu17 stitched) |
| `fakerate_2d.py` | `{era}/QCD.root` | `FR2D_QCD.root` + 2D plot | QCD MC FR 2D stitching, produces bkg.cc input |
| `fakerate_data.py` | `{era}/DoubleMuon.root, DY.root, WJets.root` | `FR2D_DATA.root` + 1D/2D plots | **data FR**: per-path MC scale from ZEnriched → prompt subtraction → FR |

```bash
rot && ERA=2016preVFP_2 python3 fakerate_2d.py       # QCD MC FR
rot && ERA=2016preVFP_data python3 fakerate_data.py  # data FR
```

### How fakerate_data.py works (prompt removal + normalization)

1. Per-path scale = N_data(Z peak) / N_MC(Z peak) — absorbs the prescale effective lumi.
   (preVFP: Mu8 4.5e-4, Mu17 9.2e-3 / postVFP: Mu8 7.6e-5, Mu17 1.4e-3)
2. In MeasReg, subtract scale × (**prompt component only** of DY+WJets) from data,
   separately for loose and tight.
   (The MC fake component is NOT subtracted — it is already part of the data fakes.)
3. FR = tight_sub / loose_sub, Mu8/Mu17 stitched at 30 GeV.

In the Mu17 tight 50–100 GeV bins the prompt contamination is 40–80% of the data,
so the subtraction is essential
(FR before subtraction ~0.25–0.36 → after subtraction 0.06–0.14, consistent with Fig 44).

## Full workflow

```bash
# 1. FR measurement jobs (fake analyzer)
SKNano.py -a fake -i 'QCD_Pt-*MuEnriched*' -e 2016preVFP -n 20   # for QCD MC FR
SKNano.py -a fake -i 'DoubleMuon*' -e 2016preVFP -n 50           # for data FR
SKNano.py -a fake -i 'DYJets*' -e 2016preVFP -n 20               #  + for prompt subtraction
SKNano.py -a fake -i 'WJets' -e 2016preVFP -n 20

# 2. hadd, then produce the FR files
rot && ERA=2016preVFP_2 python3 fakerate_2d.py
rot && ERA=2016preVFP_data python3 fakerate_data.py

# 3-a. MC closure test (QCD FR): compare matrix/full/fake vs TT/fake (truth)
SKNano.py -a bkg -i 'TTLL_powheg' -e 2016preVFP -n 20
SKNano.py -a bkg -i 'TTLJ_powheg' -e 2016preVFP -n 20    # dominant fake source
SKNano.py -a bkg -i 'WJets' -e 2016preVFP -n 20
SKNano.py -a bkg -i 'DYJets*' -e 2016preVFP -n 20
SKNano.py -a bkg -i 'SingleTop_tW_*_NoFullyHad' -e 2016preVFP -n 10

# 3-b. Data background estimate (data FR)
SKNano.py -a bkg -i 'SingleMuon*' -e 2016preVFP -n 50 --userflags DataFR
```

## Gotchas / lessons learned

- **Fig 44 vs Fig 46**: the FR measured in QCD MC must be compared to Fig 46.
  Fig 44 (data) rises at 50–100 GeV due to prompt contamination / composition
  differences — a monotonically falling QCD MC FR is expected.
- **Sumw2**: data histograms are filled with weight = 1 and carry no Sumw2 array.
  Without calling `Sumw2()` before `Divide`, the errors come out as sqrt(ratio).
- **Mu17 bins below ptcorr 30 being empty is by design** (ptCorrCut=30); the final FR
  is the Mu8(10–30) + Mu17(30–100) stitch.
- **TkRelIso < 0.4**: the framework's HcToWA muon ID has a bug making this cut
  effectively a no-op, so fake.cc / bkg.cc apply it correctly in their own
  `PassMuonWP()`.
- **FR universality**: applying the QCD-measured FR to b-decay fakes in tt-bar is
  the main systematic of this method (see AN-25-154 Fig 46 vs 47). If the closure
  test disagrees, suspect this first.
- In the matrix full calculation, LL events enter with negative weights, so
  individual bins can go negative — this is expected from the method.
