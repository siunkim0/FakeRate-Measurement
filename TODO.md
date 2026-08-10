# ==========================================================================
# Flavor-BDT for the muon FR — feasibility study        [PLAN 2026-07-29]
#   steps 1-3 (framework + ntuple + build) DONE & verified
#   steps A-E (submit / skim / train / eval) QUEUED — awaiting go-ahead
# ==========================================================================

## Why

`bkg.cc` consumes a single 2D map `FR(ptCorr, |eta|)` (a `TH2D` named literally `"FR"`).
Two parameters are not enough — the per-flavor study below (v3/v4) showed:

- FR depends strongly on the source-jet flavor. Run2-stacked QCD MuEnriched at
  ptcorr 50–100: FR(uds) ~ 0.20, FR(b) ~ 0.05, FR(pileup) ~ 0.57 (pileup *rises*
  with pT while everything else falls).
- The composition differs wildly by process: QCD muon fakes are 61–66% b,
  TTLJ 88–89% b.
- **But FR(flavor, pT, |eta|) is ~sample-universal** (QCD vs TTLJ b-FR agree to
  1.03–1.31 bin-by-bin) — so a flavor-weighted FR is well defined.

Flavor is MC truth, so to use it in data we need a **data-observable predictor of the
source-jet flavor**. That is what this branch builds: a **4-class** XGBoost classifier
(b / c / uds / g) over reconstructed muon + matched-jet + event variables,
label from the jet matched by `Muon_jetIdx` (same definition as `fake::RecoJetFlavor`):
`hadronFlavour` for b/c, `|partonFlavour|` for g/s/u/d, s+u+d merged to `uds`.

**Phase 1 goal: is the flavor separable at all?** — multiclass ROC/AUC, confusion
matrix, feature importance, sample-transfer test. Deploying a flavor-weighted FR in
`bkg.cc` is Phase 2, not this.

## Design decisions (agreed 2026-07-29)

| Decision | Choice |
|---|---|
| Ntuple source | flat `TTree` written by `fake.cc`, one row per **loose** muon in `MeasReg/Central` |
| Missing NanoAOD branches | **add them** to `SKNanoLoader` + `Muon.h` + `AnalyzerCore::GetAllMuons` |
| Target | **4-class** softprob `b, c, uds, g` |
| Dropped classes | **`pileup`** — 0.36% of loose fakes (~2.1k rows/era), ~100:1 against b, too thin to train (2026-07-29). **`unmatched`** (`jetIdx<0`, ~3.3%) — directly observable in data via `mu_hasJet`, so it stays a bucket outside the BDT. **No analyzer change**: `fake::FlavorCode` already emits `4=pileup` / `-1=unmatched`, the skim just filters them and codes 0-3 are already contiguous. |
| Eras | **full Run2** (2016preVFP + 2016postVFP + 2017 + 2018). With pileup gone, uds and g are the thin classes, and v4 below found single-era uds/gluon uninterpretable — stacking Run2 gave muon N_eff x3.8 (uds 361→1482, gen gluon 75→286). |
| Samples | QCD_Pt-*MuEnriched* + TTLJ_powheg (training) **and DoubleMuon** (data/MC input validation + eventual inference; data rows carry `flavor=-2` so they self-exclude from training) |
| Code location | new `fakeflavor/` subproject inside `/data6/Users/snuintern2/BDT` |
| Training host | the **GPU server**, not tamsa2 — everything is prepared and staged here first |

Training on the *loose* muons of the measurement region is deliberate: that is exactly
the population `FR = tight/loose` is measured on, and carrying `isTight` as a column
lets FR-per-BDT-category be computed offline with no new histogram code.

## Step 1 — load the missing muon branches

Verified against the repo's own dictionaries
(`data/Run3_v12_Run2_v9/NanoAODv9_Ref.json`, `NanoAODv12_Ref.json`):

| branch | type | v9 (Run2) | v12 (Run3) |
|---|---|---|---|
| `Muon_jetPtRelv2`, `Muon_jetRelIso` | Float_t | yes | yes |
| `Muon_jetNDauCharged` | UChar_t | yes | yes |
| `Muon_segmentComp`, `Muon_ptErr`, `Muon_tunepRelPt` | Float_t | yes | yes |
| `Muon_nStations` | UChar_t | yes | yes |
| `Muon_miniPFRelIso_chg`, `Muon_pfRelIso03_chg` | Float_t | yes | yes |
| `Muon_svIdx` | Short_t | **no** | yes |

- [ ] `Analyzers/include/SKNanoLoader.h` — `RVec<T> Muon_<x>;` in the muon block (~L231–277).
- [ ] `Analyzers/src/SKNanoLoader.cc` — `.resize(kMaxMuon)` (~L247–299),
      `BindArray("Muon_<x>", ..., &nMuon, "nMuon")` (~L1008–1055), legacy
      `SafeSetBranchAddress` path (~L1722) alongside the electron twins.
      The `Electron_jetPtRelv2` / `Electron_miniPFRelIso_chg` entries are exact templates.
- [ ] `Muon_svIdx` inside the existing `if (Run == 3)` guard (where `Muon_mvaMuID_WP`
      lives), `.resize(0)` on the Run2 side — else 2016 jobs die on a missing branch.
- [ ] `DataFormats/include/Muon.h` — setters/getters in the `SetJetIdx`/`JetIdx` style.
      **Only DataFormats file touched** ⇒ needs the full `scripts/build.sh` (rootcling),
      not `rebuild.sh`. Additive only, so no other analyzer is affected.
- [ ] `Analyzers/src/AnalyzerCore.cc::GetAllMuons` — set the new values on each `Muon`.

## Step 2 — `fake.cc` flat ntuple  [DONE 2026-07-29]

Tree API = `NewTree` / `SetBranch` / `FillTrees` (`AnalyzerCore.h:198-208`); template =
the per-jet ntuple in `Wtag.cc:13,357-414`. `SetBranch` is **scalar-only** (the
`std::vector` overload is marked unusable) ⇒ one flat row per muon.

Implemented as `fake::fillTreeRow(Event&, float evtSF)` + `fake::FlavorCode(TString)`,
behind the new userflag **`MakeTree`** (tree name `"mu"`, created in
`initializeAnalyzer`). Three non-obvious constraints drove the design:

1. **`SetBranch` hands `TBranch` the address of a `deque` element, and `FillTrees`
   does `clear() + shrink_to_fit()` on those deques.** ⇒ **every branch must be
   `SetBranch`'d on every single row.** A conditionally-skipped branch keeps a dangling
   address into freed memory. All Run2/Run3-only and no-matched-jet cases are therefore
   precomputed into sentinels (`-999.` float, `-99` int) above one unconditional
   straight-line `SetBranch` block.
2. **`fillTreeRow` is called OUTSIDE the trigger-path loop** (`fake.cc`, after the
   `for (path : paths)` block). Inside it, a muon passing both Mu8 and Mu17 would be
   written twice, and the duplicates would split across XGBoost CV folds and inflate the
   validation score. Path info survives as `pass_mu8`/`pass_mu17`/`w_mu8`/`w_mu17`.
   The per-path trigger lumi differs, so there is no single "the" weight — `mcweight`
   (path-independent) and the two path weights are separate columns.
3. **Away-jet gate opened to 30 GeV**, with `ev_hasAway30/40/60` + `ev_njet25/30/40/60`
   as columns. Central (=40) is then an offline cut and the AwayJetPt30/60 systematics
   are reproducible from the same rows — writing three syst passes instead would triple
   the rows with highly-correlated duplicates (leakage again).

Also: DATA rows are written (`isData` column) but get `flavor = -2`, because in data
`Jet::partonFlavour`/`genJetIdx` keep their ctor default `-999` and `FlavorTag` checks
`isPileup` first ⇒ every jet would come out labelled `"pileup"`.

Columns:
- **bookkeeping** — `isTight` (FR numerator), `flavor` (int code from `RecoJetFlavor(mu)`;
  u/d/s merged to `uds` offline), `genflav` (`GenJetFlavor(mu)`, cross-check),
  `isPrompt` (`GetLeptonType(mu,gens) > 0`), `path` (0=Mu8, 1=Mu17), `weight`,
  `run`/`event` (so the train/test split can be done **by event**).
- **muon** — `pt, eta, phi, conePt (= ptCorr), ptErr, ptErrRel, tunepRelPt, dxy, dxyErr,
  dz, dzErr, ip3d, sip3d, miniPFRelIso_all, miniPFRelIso_chg, pfRelIso03_all,
  pfRelIso03_chg, pfRelIso04_all, tkRelIso, jetRelIso, jetPtRelv2, jetNDauCharged,
  segmentComp, nStations, nTrackerLayers, isGlobal, isTracker, mediumId, tightId,
  softMva, mvaLowPt, mvaTTH`.
- **matched jet** — resolved exactly like `RecoJetFlavor`: scan the `rawJets` member
  (`fake.h:76`, pre-cleaning) for `jet.OriginalIndex() == mu.JetIdx()`; sentinel `-99`
  when absent. Accessors verified in `DataFormats/include/Jet.h`: `jet_pt, jet_eta,
  jet_dR (mu.DeltaR(jet)), jet_ptRatio = mu.Pt()/jet.Pt()`, `chHEF/neHEF/neEmEF/chEmEF/
  muEF` (L68-72), `nConstituents/nElectrons/nMuons/nSVs` (L81-84), `qgl()` (L88), and
  DeepFlav `B/CvB/CvL/QG` via `GetBTaggerResult` / `GetQvGTaggerResult` (L154-156).
- **event** — `MET, MT, nJets, awayJet_pt, awayJet_dR, nPV`.

**`conePt` and `MT` are BDT input features, not just bookkeeping.**
`conePt = pt*(1 + max(0, miniIso - 0.1))` is the FR's own x-axis, so the classifier gets
the isolation-corrected pT scale directly; `MT` is the prompt-W discriminant and
correlates with how the muon was produced. Both kept in `full` **and** `notag`. Since
they are the variables the FR is already binned in, evaluation must also report
performance *within* (conePt, |eta|) cells — a classifier that only rediscovers the
existing 2D binning adds nothing.

Notes:
- Run-dependent columns get sentinels and are dropped per era: `jet_qgl`/`jet_puIdDisc`
  are **Run2-only**; `Muon_svIdx` / `Jet_nSVs` / PNet / RobustParT are **Run3-only**.
  Deliberately excluded: `Jet_chMultiplicity`/`neMultiplicity` (v13-only, absent in v9
  *and* v12) and `Jet::nSVs()` (hardcoded `0` in Run2 at `AnalyzerCore.cc`, so it would
  look like a real measurement).
- Write **both prompt and fake** muons (`isPrompt` is a column, so the fake gate is
  applied offline and prompt muons stay as a control sample).

## Step 3 — build  [DONE 2026-07-29]

**`scripts/build.sh`** (rootcling regen — `Muon.h` gained members and its `ClassDef` went
1 → 2), not `rebuild.sh`. Exit 0; `fake.h` / `SKNanoLoader.h` reinstalled.

# --------------------------------------------------------------------------
# Steps A/C run 2026-07-29. Step D (training on the GPU box) not started.
# --------------------------------------------------------------------------

## Step A — submit  (tag `bdtflav`, Claude submits on tamsa2)   [MC SUBMITTED 2026-07-29 21:50]

**Submission log.** Pre-flight on tamsa2 before submitting:

- `HLT_Mu8_TrkIsoVVL` / `HLT_Mu17_TrkIsoVVL` and `QCD_Pt-*_MuEnriched` (12 pt bins),
  `TTLJ_powheg`, `DoubleMuon` all exist in **all four** eras. `fake.cc` has no 2016-only
  logic beyond `cuts.jet_eta_max` (2.4 vs 2.5), which is already era-switched.
- Single-file smoke runs pass for **2018 QCD**, **2018 DoubleMuon** and **2017 TTLJ**
  (container recipe in `~/.claude/CLAUDE.md`, `scratchpad/smoke2.py`): tree written, jet
  branches filled, `flavor` = -2 throughout in data, `w_mu8` = 1 for data.
- `build/redhat/*/lib*.so` (17:58) newer than all edited sources (16:16) — no rebuild needed.

**MC submitted** (8 DAGs, 2132 nodes total, no zero-node DAGs):
`QCD_Pt-*MuEnriched*` 492 nodes/era (12 samples x 40 + 12 hadd), `TTLJ_powheg` 41/era.
Runlogs `/gv0/Users/snuintern2/SKNanoRunlog/2026_07_29_2150{02,08,12,17,22,24,26,28}_fake_MakeTree`.

**DATA deliberately held back.** The `DoubleMuon` lines are in `fake.sh` but commented:
the pool already had 1649 idle jobs from other users, and 8 MC DAGs added ~2100 more.
Submit the 4 data DAGs (~920 more) once MC drains. Data is not needed for training at
all — only for the input-feature data/MC comparison — so it is not on the critical path.

**Gotcha found in the sample DB while checking:** `DoubleMuon` period files do not respect
the VFP split — 2016preVFP has B-F and 2016postVFP has B-H, so B-E are registered under
*both* 2016 eras, and 2018 has A-F where data only goes A-D. Harmless for a data/MC shape
comparison, but do **not** reuse these for a yield-level data number without checking.

**12 submissions** = {`QCD_Pt-*MuEnriched*`, `TTLJ_powheg`, `DoubleMuon`} x
{2016preVFP, 2016postVFP, 2017, 2018}. All three are registered in the sample DB for all
four eras (DoubleMuon as period files `DoubleMuon_{A..H}.json`).

Edit `SKNanoAnalyzer/fake.sh` **in place** (the established batch-list convention — that
file is a history of commented `SKNano.py` lines): comment out the two currently-active
`elecfake ... --tag QCE_1` lines, append per era

```
SKNano.py -a fake -i 'QCD_Pt-*MuEnriched*' -e <era> -n 40 --userflags MakeTree --tag bdtflav
SKNano.py -a fake -i TTLJ_powheg           -e <era> -n 40 --userflags MakeTree --tag bdtflav
SKNano.py -a fake -i DoubleMuon            -e <era> -n 40 --userflags MakeTree --tag bdtflav
```
then `source setup.sh >/dev/null 2>&1 && bash fake.sh`.

- The userflag becomes a **directory level** (`python/SKNano.py:423-429`), so outputs land
  in `/gv0/Users/snuintern2/SKNanoOutput/fake/MakeTree/{era}_bdtflav/<sample>.root` —
  *not* the usual `fake/{era}_bdtflav/`. `templates/hadd.sh` merges the `mu` tree across
  sub-jobs automatically.
- **MC hadd is part of the DAG; data is not** — DoubleMuon gives one file per period, so
  `rot && hadd` `DoubleMuon_*.root → DoubleMuon.root` afterwards.
- Watch for zero-node DAGs (= `-i` pattern matched nothing). Poll with
  `condor_q -af JobBatchName | grep -c fake` + output mtimes; `condor_q <dagman-id>
  -totals` reports 0 for a LIVE dag.

Sizing (measured, 2016preVFP QCD MuEnriched `MeasReg/Central/loose`, 587,701 muons):
b 64%, c 25%, g 3.7%, uds 3.2%, unmatched 3.3%, pileup 0.36%. 100 branches x 4 B with
ROOT compression ⇒ ~50-70 MB/era — disk is not the constraint, **class imbalance is**
(b vs uds is ~20:1 even after dropping pileup ⇒ per-class weights are mandatory).
**Both QCD and TTLJ are required** — QCD alone teaches the classifier the QCD flavor
prior, and TTLJ is the sample the closure test has to work on.

## Step B — skim to parquet (`BDT/fakeflavor/skim.py`)

This is the training-data skim: `uproot` reads the hadded `mu` trees → one parquet per
(sample, era) under `BDT/fakeflavor/data/ntuples/`.

Row selection:
- `ev_hasAway40` (Central measurement region);
- **path stitch** matching `fakerate_2d.py`: `mu_conept < 30` → require `pass_mu8`, use
  `w_mu8`; else require `pass_mu17`, use `w_mu17`. Emits a single `weight` + `path`
  column so each muon appears exactly once;
- MC training rows: `!isPrompt` and `flavor in {0,1,2,3}` (drops pileup and unmatched);
- data rows go to a separate parquet (`flavor == -2`), for data/MC plots only.

### [DONE 2026-07-30] full Run2 skim

All 12 DAGs finished (13/13 MC files per era; DoubleMuon periods hadded by hand — the data
hadd is **not** in the DAG). `python -m fakeflavor.skim --data`, ~90 s:

| class | rows | sum(w) | N_eff | N_eff in the 2026-07-29 partial skim |
|---|---|---|---|---|
| b   | 1654337 | 5.004e8 | 482558 | 292305 |
| c   |  644638 | 2.090e8 | 207748 | 125418 |
| uds |   87578 | 3.506e7 |  25802 |  15136 |
| g   |  111829 | 3.305e7 |  30177 |  17791 |

2498382 MC rows total + 345814 DoubleMuon rows. No `check_features` warnings (no MISSING,
all-NaN or newly CONSTANT column) — the feature list still holds on the full Run2 set.

Two things learned here, both worth keeping:

1. **The data glob must be `DoubleMuon.root`, not `DoubleMuon*.root`.** Once the periods are
   hadded, a `*` glob matches the hadded file *and* its own inputs, and silently doubles
   every data row (first run gave 691628 = 2 × 345814). Fixed in `config/samples.yaml`.
2. **TTLJ carries no physical weight next to QCD.** QCD sum(w) = 5.004e8 vs TTLJ 5979 — a
   factor 8e4. `ALL MC` b N_eff is 482558 against QCD-alone 482546, i.e. TTLJ's 133501 b
   rows are worth ~12 effective events. So `train.parquet` with `physical_weight: true`
   **is** a QCD-only training set, and the combined file cannot teach process-independence.
   Use it as such, and keep TTLJ as the held-out transfer test (its own N_eff is adequate:
   b 127243, c 12947, uds 1394, g 1026). A per-sample weight renormalization so both
   processes contribute equally is a legitimate variant — run it *second*, after seeing how
   far the QCD-trained model transfers.

## Step C — `BDT/fakeflavor/` training subproject   [WRITTEN 2026-07-29]

Written while the jobs ran; all modules `py_compile`-clean, not yet run on real input.
`BDT/fakeflavor/{__init__,features,skim,train,evaluate,cv}.py`,
`config/{samples,train}.yaml`, `requirements.txt`, `bootstrap.sh`, `README.md`;
`BDT/CLAUDE.md` gained a pointer. Package is imported as `fakeflavor.*` from the BDT
project root so `from src.utils import ...` keeps working.

Deviations from the sketch below, all deliberate:
- **`mu_eta` / `mu_phi` / `jet_eta` are NOT features** — only `mu_abseta` and a derived
  `jet_abseta`. The FR is eta-symmetric, so signed eta only lets the trees chase
  fluctuations. Also dropped: `ev_hasAway40` (constant after the Central cut),
  `ev_hasAway60` (redundant with `ev_awayjet_pt`), and **`ev_nTrueInt` (MC truth — does
  not exist in data)**, which the original feature sketch would have leaked in.
- **Jet sentinels → NaN in the skim.** `-999`/`-99` left in place would put unmatched-jet
  muons at the far end of every jet axis and let the trees carve out a spurious "flavor"
  region for them; as NaN, xgboost learns a default direction per split instead.
  `mu_hasJet` stays as the explicit flag.
- **Weighting is not `build_train_weight`.** That helper is per-*sample*-equal, which is
  wrong here: the 12 QCD pt bins are one physical sample, and equalizing them would
  destroy the pt spectrum. Instead `train.py::build_weights` uses the analyzer weight,
  clipped at the 99.9th percentile *within each class* (QCD pt-bin weights span ~6 orders
  of magnitude), then class-balanced. `evaluate.py --prior-correct` undoes the balancing
  when calibrated probabilities are wanted.
- **Train/test split is by `(run, lumi, evt)`**, not by row (`split_by_event`). A no-op
  guard today — the selection has exactly one loose muon per event — but the selection
  may loosen.

<details><summary>original sketch</summary>

Mirror the existing repo layout (`src/{skim,features,train,evaluate,cv}.py`,
`config/*.yaml`, `scripts/run_*.sh`), reuse `src/utils.py` (SEED=42, get_logger,
detect_gpu) and the weighting / ONNX helpers. Repo conventions (`BDT/CLAUDE.md`):
pathlib, parquet intermediates, `logging` not `print`, 300-dpi PNGs.

- `features.py` — `FEATURES` (**order is load-bearing** — the H→ZZ side was already
  bitten by a C++ feature-ordering bug), incl. `conePt`/`MT`, plus derived
  `dxySig = dxy/dxyErr`, `miniIso_neutral = all - chg`, `log(jet_ptRatio)`.
  Excluded: label/bookkeeping columns (`flavor*`, `isTight`, `isPrompt`, `leptonType`,
  `isData`, `pass_*`, `w_*`, `mcweight`, `run/lumi/evt`) and the branches the smoke test
  showed are **constant because the loose WP implies them** — `mu_pogMedium`,
  `mu_isTracker`, `mu_isPFcand`, `ev_hasAway30`.
- `train.py` — `XGBClassifier(objective="multi:softprob", num_class=4,
  eval_metric="mlogloss", tree_method="hist", device="cuda")`; reuse `build_train_weight`
  (per-sample-equal within class, class-balanced, mean ~ 1).
- `evaluate.py` — one-vs-rest ROC/AUC per class, row-normalized confusion matrix,
  gain importance, per-true-class score distributions, predicted-vs-truth composition
  vs ptcorr.
- `cv.py` — 5-fold StratifiedKFold, reusing the `src/cv.py` pattern.
- `requirements.txt` + `bootstrap.sh` — self-contained venv (xgboost>=2 CUDA build,
  uproot>=5, awkward>=2, sklearn, pandas, pyarrow, matplotlib).

Two feature sets, because the b/c-taggers on the muon's *own* jet encode nearly the same
information as `hadronFlavour` and are the most likely thing to be mismodeled in data:
- **`full`** — everything.
- **`notag`** — `jet_deepjet_*` + `jet_qgl` removed (kinematics incl. conePt/MT, IP,
  isolation, jet energy fractions and multiplicities only).

</details>

## Step D — handover to the GPU server

Steps A-C run on tamsa2. Training does **not**: once the parquet is staged and the code
is written, stop and report ready; the user starts Claude on the GPU box to run
`train.py` / `evaluate.py` / `cv.py` there.

**Env caveat.** The LCG_105 el9 view (`rot`) = python 3.9, xgboost 2.0.2, sklearn 1.2.2,
pandas 1.5.3, pyarrow 11 — enough for CPU training — but **uproot 4.3.7 / awkward 1.10.3**,
older than `BDT/requirements.txt` (uproot>=5, awkward>=2, py>=3.10), and `onnxmltools` is
missing. Hence the self-contained venv. *Open question:* if the GPU box already has a
conda env worth reusing, point Claude at it instead of bootstrapping a new one.

## Leakage diagnostic — do not skip

`mu_miniiso`, `mu_sip3d`, `mu_tkreliso` and `mu_conept` are the very variables that define
the **tight WP**, i.e. the FR numerator (`conePt = pt*(1+max(0, miniiso-0.1))` literally
encodes miniiso). Using them to predict *flavor* is legitimate — flavor is the label — but
a flavor-weighted FR built on top could be biased if the predicted flavor correlates with
tightness *within* a true flavor.

⇒ Plot **FR vs predicted-flavor probability, within each true flavor**. It should be flat.
If it is not, retrain excluding the WP-defining variables and compare.

### It is NOT flat. [measured 2026-07-29, partial QCD skim]

First real training (993k rows, ~40% of the eventual QCD skim, no TTLJ yet, CPU) —
`evaluate.py::leakage_check`, chi2 vs a flat line:

| true flavor | inclusive FR | chi2/ndf vs flat | shape |
|---|---|---|---|
| b   | 0.289 | 1346 / 9 | mild, non-monotonic |
| c   | 0.190 | **9215 / 9** | **FR 0.41 → 0.00 as p(c) goes 0.15 → 0.95** |
| uds | 0.323 | 12.5 / 7 | flat ✓ |
| g   | 0.333 | 56 / 8 | falling, 0.37 → 0.26 |

The `c` panel is the problem: within *true charm*, the muons the model is most confident
are charm are ~never tight. So `Σ_f p_f(x)·FR_f` would be badly biased for c — the model
is partly learning "how prompt-muon-like is this object", which is the FR numerator, not
the jet flavor. Consistent with the gain ranking, where **`mu_softMva` is the #2 feature**
(after `jet_deepjet_b`) and `mu_sip3d` / `mu_miniiso` / `mu_miniiso_neu` are all top-10.

⇒ Added a **third feature set `nowp`** (`fakeflavor/features.py::WP_FEATURES`): drops
`mu_conept`, all isolation (`mu_miniiso*`, `mu_pfiso*`, `mu_tkreliso`, `mu_jetRelIso`),
all IP significances (`mu_sip3d`, `mu_ip3d`, `mu_dxysig`, `mu_dzsig`) and the lepton MVAs
(`mu_softMva`, `mu_mvaTTH`, `mu_mvaLowPt`); `mu_pt` replaces `mu_conept`. `full` − `nowp`
is the size of the bias. **Note this overrides the earlier instruction to use conePt as an
input** — conePt cannot be an input to the model whose output reweights the FR, because
conePt *is* miniiso and miniiso *is* half the tight WP. It stays an input in `full`/`notag`
purely as the upper bound on separability.

Caveat: partial statistics, and these chi2 values are per-bin binomial with N_eff from the
weights, so they are indicative rather than a goodness-of-fit. Re-run on the full skim.

### `nowp` does NOT fix it. [measured 2026-07-29, same partial skim]

The control model was trained (58 features) and the diagnostic re-run. Dropping the
WP-defining variables costs real separation and **does not remove the leakage**:

| true flavor | `full` chi2/ndf | `nowp` chi2/ndf | verdict |
|---|---|---|---|
| b   | 1346 / 9 | **2024 / 9** | *worse* |
| c   | **9215 / 9** | **7425 / 9** | −20%, still catastrophic |
| uds | 12.5 / 7 | 19.6 / 7 | both ~flat |
| g   | 56 / 8 | 35.0 / 7 | mildly better |

Test one-vs-rest AUC (class-balanced), `full` → `nowp`:
b 0.875 → 0.851, c 0.802 → 0.752, uds 0.770 → 0.735, g 0.775 → 0.749.

So `nowp` pays ~0.03-0.05 AUC across the board and buys a 20% reduction in the charm chi2 —
which was ~1000x too large to begin with — while making b worse. This is the caveat below
coming true: the jet-side variables (`jet_pt`, `jet_ptratio`, `jet_muEF`, `jet_nmuons`)
track muon isolation too, so removing the *muon-level* WP variables cannot decouple the
prediction from tightness. Cutting deeper would mean dropping the jet kinematics as well,
i.e. most of the discrimination.

⇒ **The flavor-weighted-FR route looks structurally compromised, not merely contaminated.**
Tightness and "which flavor does this look like" are entangled in the observables
themselves, not in a removable subset of them. Prefer the direct route below. Confirm on
the full skim before treating this as final, but do not expect a different answer.

### Full-statistics confirmation [measured 2026-07-30 on ai-tamsa1, 2x RTX 3090]

Env: the `ml` conda env (`/home/snuintern2/miniconda3/envs/ml`, py3.11, xgboost 3.2.0,
uproot 5.7.2, awkward 2.9.0) — no bootstrap needed, `/data6` is mounted so the parquet is
read in place. GPU is real and worth using: 0.49 s vs 86 s for the same 50 rounds (176x),
and `fakeflavor.train::detect_gpu` correctly returns True here / False on tamsa2.

Test one-vs-rest AUC (class-balanced) on 2498382 rows, and `best_iteration`:

| feature set | b | c | uds | g | best_iter |
|---|---|---|---|---|---|
| `full` (74)  | 0.8789 | 0.8048 | 0.7749 | 0.7798 | 534 |
| `notag` (69) | 0.8722 | 0.7925 | 0.7704 | 0.7737 | 602 |
| `nowp` (58)  | 0.8586 | 0.7585 | 0.7466 | 0.7526 | 650 |

5-fold CV, `full`: mean OvR AUC **0.8112 ± 0.0019**, mean n_trees 556. Stable.

**1. Separability: yes.** In-cell AUC (inside each existing (conePt,|eta|) FR cell) is
0.70-0.92 and rises monotonically with conePt — e.g. `full` at [50,100)x[0,0.9):
b 0.919, c 0.870, uds 0.914, g 0.902; worst is c at [10,12)x[1.6,2.4) = 0.701. The
classifier is **not** just rediscovering the 2D binning. Confusion diagonal
0.662/0.605/0.433/0.465, with uds<->g mixing ~0.27 both ways.

**2. Leakage: worse, and now clearly a bias rather than noise.** chi2/ndf vs flat grew
roughly in proportion to the dataset (993k -> 2.5M rows), which is what a genuine
fractional deviation does — noise would have stayed put:

| true flavor | `full` @993k | `full` @2.5M | `notag` | `nowp` |
|---|---|---|---|---|
| b   | 1346 / 9 | 1914 / 9 | 2138 / 9 | 3020 / 9 |
| c   | 9215 / 9 | **13483 / 9** | 16285 / 8 | 11881 / 9 |
| uds | 12.5 / 7 | 30.1 / 8 | 32.5 / 8 | 42.7 / 7 |
| g   | 56 / 8   | 91.8 / 8 | 93.3 / 8 | 77.4 / 7 |

**Why no feature-set swap can fix it — the mechanism, now measured.** The flavor
information is *duplicated* across the muon-level and jet-level blocks, and each copy
correlates with tightness. Compare the gain rankings:
- `full`: `jet_deepjet_b` 695, `mu_softMva` 523, `jet_deepjet_cvl` 370, `jet_deepjet_cvb` 320
- `notag` (taggers gone): top list is *entirely muon-level* — `mu_softMva` 595,
  `mu_mvaTTH` 311, `mu_miniiso` 225 — and costs only 0.007-0.012 AUC
- `nowp` (muon WP vars gone): top list is *entirely jet-level* — `jet_deepjet_b` 546,
  `jet_deepjet_cvl` 322 — and costs 0.02-0.05 AUC

Either block alone recovers most of the separation. So dropping the tightness-correlated
variables cannot decouple p_f from tightness: there is always a second copy of the flavor
signal, equally entangled.

**3. Sample transfer: b transfers, light flavor does NOT.** (Step E item 2, class-balanced
AUC. `train.parquet` is QCD by weight, so "QCD-trained" = the `full` model above.)

| direction | b | c | uds | g |
|---|---|---|---|---|
| QCD -> QCD (native) | 0.8791 | 0.8049 | 0.7738 | 0.7802 |
| QCD -> TTLJ         | 0.8805 | 0.7872 | 0.7548 | **0.6937** |
| TTLJ -> QCD         | 0.8546 | 0.7657 | 0.7324 | **0.6850** |

b is essentially unchanged in both directions; c and uds lose 0.02-0.04; **gluon loses
0.09-0.10 and collapses**. The TTLJ-trained model's confusion on QCD is the clearest
symptom: true g predicted as g only **0.088**, as uds **0.506** — it cannot identify
gluons at all, having seen 1076 of them. QCD->TTLJ is milder but still degraded
(true uds -> predicted g 0.385 vs 0.279 natively).

This is not merely a statistics artifact of TTLJ's thin light-flavor classes: it lands on
exactly the classes whose *composition* differs most between processes (QCD fakes 61-66% b,
TTLJ 88-89% b), i.e. precisely where a flavor-weighted FR would need the prediction to be
trustworthy. A ~0.02 AUC error for uds/g will not be recovered by the closure test.

⇒ **Phase 1 verdict: flavors are distinguishable (Q1 answered: yes), but the
flavor-weighted FR now fails on two independent grounds — the leakage bias and light-flavor
non-transfer.** Nothing here softens the recommendation below; both new pieces of evidence
point the same way.

### Physics closure: the flavor route's *ceiling* is worse than the flat map
[measured 2026-07-30, `fakeflavor/closure.py`]

Step E item 3, and it settles the question. Per-flavor FR maps measured on QCD
MuEnriched (Run2-summed, **recoflav** to match the parquet's label), applied to TTLJ,
predicting N(tight) three ways and comparing to the observed count. Population = the
parquet rows (non-prompt loose muons, flavor in {b,c,uds,g}), so pileup/unmatched are
excluded from maps *and* observation, and `flat` is built as `Sum_f tight_f / Sum_f loose_f`
over those same four flavors rather than from `FR2D_QCD.root` (whose denominator is all
loose muons) — otherwise the method comparison would be confounded by a population
difference.

| method | QCD FR -> TTLJ | TTLJ FR -> TTLJ (control) |
|---|---|---|
| OBSERVED | 1114.42 ± 7.22 | (same) |
| truth-weighted `Sum_f n_f FR_f` | **1.1270** | 0.9987 |
| flat FR (what bkg.cc does) | **1.0949** | 0.9989 |
| bdt, prior->QCD mixture (deployable) | 1.0950 | 0.9918 |
| bdt, prior->TTLJ mixture (cheating) | 1.1301 | 1.0025 |
| bdt, raw class-balanced (diagnostic) | 1.1344 | 1.0633 |

**The control validates the machinery**: measuring and applying in the same process closes
to 0.1% inclusively and 0.996-1.016 in every conePt bin, so the lookup, the Mu8/Mu17
stitch, the overflow convention and the weighting are all correct. The QCD->TTLJ numbers
are therefore physics, not a bug.

**Three things follow, in order of importance:**

1. **`truth-weighted` (1.127) does not beat `flat` (1.095) — it is worse.** Truth-weighted
   is the ceiling of the entire flavor route: it uses the MC-truth flavor, i.e. a perfect
   classifier. Since the ceiling loses to the map we already have, *no* classifier can make
   the flavor-weighted FR worthwhile. This is the single decisive number of Phase 1.
2. **The premise fails: per-flavor FR is not sample-universal.** The v4 study's
   "`FR(flavor, pT, |eta|)` is approximately sample-universal" is what the whole route
   rested on. Comparing the two map sets directly at Mu8 [10,12)x[0,0.9):
   FR_b 0.4831 (QCD) vs 0.4501 (TTLJ), FR_c 0.4507 vs 0.3540 (-21%). The QCD-measured maps
   overpredict TTLJ by ~13% even *given perfect truth flavor*, and decomposing by flavor
   does not absorb the sample dependence — it slightly amplifies it. The flat map's own
   composition mismatch (QCD 64% b vs TTLJ 89% b) happens to partly cancel against the
   within-flavor difference, which is why it looks better.
3. **The classifier itself is fine — it is the method that fails.** Given the correct
   prior, the BDT reproduces the truth-weighted answer to 0.3% (1.1301 vs 1.1270 on QCD
   maps; 1.0025 vs 0.9987 in the control). And note `bdt [qcd]` = 1.0950 vs `flat` = 1.0949:
   prior-corrected to the composition you would actually know without MC truth, the
   flavor-weighted FR is *numerically indistinguishable from the flat map*. It adds nothing
   while costing an ONNX inference path inside `bkg.cc`.

Differentially, `flat` is also the more stable of the two: 1.05-1.14 across the whole
ptcorr range, while `truth` drifts 1.08 -> 1.20 -> 0.87 -> 0.76, i.e. the flavor
decomposition introduces a ptcorr-dependent trend that the flat map does not have.

Caveats, stated plainly: one process pair (QCD vs TTLJ), MC only; TTLJ's light-flavor
statistics are thin, so FR_uds/FR_g in the control carry ±0.13/±0.10 at low ptcorr — but
uds+g are only 1.7% of TTLJ fakes, so this cannot move the conclusion. The
truth-vs-flat gap (35.8 events on 1114) is large against both the observed stat (7.2) and
the map stat (3.8, 3.4, positively correlated).

⇒ **Do not build the flavor-weighted FR.** Three independent failures now point the same
way: the leakage bias, light-flavor non-transfer, and — decisively — a ceiling that loses
to the existing flat map. Keep the flavor classifier as an interpretation tool.

### [2026-07-30] The direct route works — `P(tight | loose, x)` transports 4x better

Measured on `ai-tamsa1` with `BDT/fakeflavor/train.py --target tight` (new mode) and the
extended `closure.py`. Trained on **QCD alone** (2.35M loose muons; TTLJ is weightless
beside it), applied to TTLJ. Same population, maps, stitch and weights as the flavor
closure above, so the numbers are directly comparable to it.

Every method reduces to one per-muon number `f_i`, reported two ways — and **the two are
not interchangeable**:

| form | formula | what it tests |
|---|---|---|
| `rate` | `sum_{all loose} w_i f_i` | does the parametrization transport between processes |
| `matrix` | `sum_{loose & !tight} w_i f_i/(1-f_i)` | what `bkg.cc` actually computes |

Both are unbiased when `f` is the true conditional (`sum_i w_i (1-f_i) f_i/(1-f_i) =
sum_i w_i f_i` identically), so `matrix` is not a second physics test — it is an
**efficiency** test, and it is the one that exposes WP leakage.

| method (rate form) | QCD self-closure | QCD -> TTLJ | tightness AUC |
|---|---|---|---|
| `flat` FR — what bkg.cc does | 1.0060 | 1.0949 | — |
| `direct_map` = BDT on (conePt,\|eta\|) only | 1.0085 | 1.0900 | 0.692 |
| **`direct` = BDT on 53 WP-free variables** | **1.0007** | **1.0227** | 0.888 |
| truth-flavor ceiling (for reference) | 1.0058 | 1.1270 | — |
| `direct_full` = all 74 variables (leak demo) | 0.9558 | 0.9583 | 1.0000 |

**1. The transfer bias drops from +9.5% to +2.3%, a 4x reduction** — and it is uniform, not
a cancellation: on \|eta\| (see the axis note below) `direct` gives 1.034 / 1.021 / 1.010
against `flat`'s 1.101 / 1.100 / 1.079. Self-closure on held-out QCD is 1.0007, so the
machinery is verified the same way the flavor closure was.

**2. `direct_map` is the control that makes this a real result.** A BDT given only
(conePt, \|eta\|) reproduces the histogram map to 0.5% (1.0900 vs 1.0949, AUC 0.692). So the
improvement comes from the *added variables*, not from replacing a histogram with a BDT.

**3. The WP-leakage trap is real, quantified, and invisible in the rate form.**
`direct_full` (all 74 variables) reaches tightness AUC **1.0000** and a rate-form closure of
0.9583 that looks perfectly respectable — while its `matrix` form is **0.0186**, i.e. it
predicts 2% of the fake background. Quoting the rate form alone would have shipped this.
The exact mechanism, worth stating because it is not obvious:
`conePt = pt x (1 + max(0, miniiso - 0.1))` means **conePt == pt exactly iff
miniiso <= 0.1**. Measured on QCD: that indicator agrees with the isolation cut for
99.9999% of rows, `P(tight | conePt > pt) = 0.0000` exactly, and conePt/pt alone has AUC
0.933 for tightness. So (`mu_pt`, `mu_conept`) is a *lossless* tag for half the tight WP,
and pt is recoverable many other ways (`jet_ptratio x jet_pt`, `mu_ptErr/mu_ptErrRel`, ...).
Dropping `mu_conept` closes all routes at once and costs nothing, since raw `mu_pt` carries
the same scale WP-independently. **This is also the exact mechanism behind the flavor
route's leakage-diagnostic failure** — every flavor feature set contains both columns, so
those models had an exact tightness tag available throughout. The rule now enforced in
`features.py`: *no input may be a function of miniPFRelIso or of SIP3D*; correlation with
them is fine and useful (`jet_muEF`, `jet_ptratio` reach single-variable AUC 0.65 and stay).

**4. The new blocker is conditioning, not physics.** `direct`'s *matrix*-form self-closure
on held-out QCD is **0.9384**, not 1.0, and 39% of the whole estimate comes from the
largest 1% of fail-side terms (`flat`: 15%), with 251 muons hitting bkg.cc's `f < 0.95`
clamp. Cause: `e = f/(1-f)` diverges as `f -> 1`, so a fail-side muon the model was
confident about enters with weight >= 19. **A sharper FR is an intrinsically
worse-conditioned input to the matrix method**, and the fail-side form is not optional —
in data the tight leptons are prompt-dominated, which is the whole reason the estimate is
built from the L side. So this must be fixed before the +2.3% can be banked; note
1.0233/0.9384 ~ 1.09 is what you get if the two defects are assumed multiplicative, i.e.
exactly `flat` — the conditioning defect can eat the entire gain.

### [2026-07-30] Sharpness/conditioning scan — the "trade AUC for conditioning" idea fails

Scanned `max_depth` 6/4/3/2 and score quantization q10..q200 with an optional cap that
merges the top score bins until their *measured* FR is bounded (`closure.py --quantize N
--fr-cap C`; merging rather than clipping, because the merged value is still the mean
measured rate of its level set, which is exactly the condition for the matrix form to be
unbiased — clipping would put an uncalibrated value there, which is what bkg.cc's blind
0.95 clamp does today). `corr` = transfer / self-closure, i.e. the method's own defect
divided out.

| config | AUC | rate self | rate TTLJ | rate corr | mat self | mat TTLJ | mat corr | top1% |
|---|---|---|---|---|---|---|---|---|
| `flat` (bkg.cc) | — | 1.0060 | 1.0949 | 1.0884 | 1.0108 | 1.1320 | 1.1199 | 0.150 |
| depth 6 | 0.8877 | 1.0007 | 1.0227 | 1.0220 | 0.9384 | 1.0233 | 1.0905 | 0.392 |
| depth 4 | 0.8839 | 1.0021 | 1.0330 | 1.0308 | 0.9316 | 1.0353 | 1.1113 | 0.377 |
| depth 3 | 0.8793 | 1.0027 | 1.0428 | 1.0400 | 0.9226 | 1.0469 | 1.1347 | 0.360 |
| depth 2 | 0.8695 | 1.0034 | 1.0652 | 1.0616 | 0.9096 | 1.0658 | 1.1717 | 0.330 |
| d6 ~q100 | — | 0.9984 | **1.0046** | **1.0063** | 0.9856 | 1.0494 | 1.0648 | 0.420 |
| d6 ~q100 cap 0.95 | — | 1.0021 | 1.0074 | 1.0053 | 1.0678 | 1.1204 | **1.0492** | 0.339 |
| d6 ~q100 cap 0.80 | — | 1.0024 | 1.0189 | 1.0164 | 1.0442 | 1.1377 | 1.0895 | 0.212 |
| d6 ~q100 cap 0.70 | — | 1.0019 | 1.0376 | 1.0355 | 1.0279 | 1.1537 | 1.1224 | 0.135 |

**1. Reducing capacity is the wrong lever — it hurts both things at once.** Transfer degrades
monotonically (rate 1.0227 -> 1.0652 from depth 6 to 2) for almost no AUC (0.888 -> 0.870),
*and* the matrix self-closure gets **worse** (0.9384 -> 0.9096), not better. Underfitting
shrinks the high-`f` predictions toward the mean and the matrix form amplifies error at high
`f` by `1/(1-f)`, so a blunter model is a *worse*-calibrated matrix input even though
`top1pct` and `n_clamp` improve. Sharpness was never the disease. **Keep depth 6.**

**2. Quantizing the score into a measured FR lookup is the right move, and is the deployment
format anyway.** It recalibrates by construction (matrix self-closure 0.92-0.94 -> 0.99-1.03)
and gives the best rate-form transfer of anything measured: **1.0046 vs flat's 1.0949**, i.e.
+0.6% against +8.8% corrected — a **14x** reduction in transfer bias. q50/q100/q200 agree to
0.0004, so 50 bins is enough; this ships as a 1D `TH1D` FR(score), or 2D against |eta|.

**3. The matrix form is still the blocker, and the cause is now split in two.**
   - `e = f/(1-f)` with a top-bin FR of 0.99 gives `e = 99`, truncated to 19 by bkg.cc's
     clamp: unclamped 1.0786 vs clamped 0.9856, i.e. *the clamp decides the answer*. Merging
     the top bins removes the clamp entirely (`n_clamp` 0) but costs resolution, and the
     matrix-corrected transfer only reaches ~1.05 against flat's 1.120 — the bias halves,
     it does not vanish.
   - **The score -> FR calibration map must be built out-of-fold, and currently is not.**
     This is the likely explanation for matrix self-closure sitting at 1.02-1.07 instead of
     1.0 while the rate form stays at 1.002: the map is measured on the full QCD parquet,
     80% of which the model trained on, so its high-score FR is inflated by overfitting —
     and the matrix form multiplies that inflation by `1/(1-f) ~ 20`. The rate form only
     sees it linearly, which is exactly the observed pattern.

*Next, in order:* **cross-fit the calibration map** (build FR(score) from held-out folds
only) and re-measure — until that is done the matrix-form numbers are not trustworthy and
only the rate-form result (+0.6%) should be quoted. Then reverse transfer (train TTLJ ->
apply QCD), then DY/WJets. Deployment still needs the ONNX path in `bkg.cc` — or, better,
the quantized FR(score) as a plain histogram, which needs no ONNX at all, only the 53
features at FR-lookup time — and the input data/MC comparison (Step E item 4).

*Axis note (matters for reading any differential table):* the differential identity needs
`E[tight | x, axis in bin] = f(x)`, so the axis must add nothing about tightness beyond the
method's own inputs. Because (pt, conePt) determine miniiso, **a conePt axis favours the map
methods and a mu_pt axis favours `direct`** — measured: `direct` swings 0.82-1.37 across
conePt bins while closing to 1.0007 inclusively, and `flat` swings 1.41-0.38 across mu_pt
bins in its *own self-closure*. Only `|eta|` is in every parametrization compared, so only
`|eta|` is fair; `closure.py` prints all three and labels which is which.

### [2026-07-30] Reference paper adopted: arXiv:2511.06972 (Neural Fake Factor / DBI)

`ref/2511.06972v2-2.pdf`, Gavranovič et al., *Neural Fake Factor Estimation Using Data-Based
Inference*. It is the same estimator we built, arrived at independently, plus three things we
did not have. Worth reading §2–3 before touching this code again.

**It is literally our model.** Their method trains a binary classifier to separate tight from
loose-and-not-tight and uses the logit `q` via the likelihood-ratio trick, so
`F(x) = exp(q) = p^T(x)/p^L(x)` is the **fake factor** (their Eq. 10–12). Our `--target tight`
model trains on all loose muons with `y = isTight`, so class 0 *is* loose-and-not-tight, and
`sigma(q) = f`, `exp(q) = f/(1-f) = F`. Same fit, two readings. Consequence: `f/(1-f)`
diverging is **not** a defect of our route — it is the published method's central object. What
was wrong was working in `f`-space; `q = log F` is the natural variable, because the matrix
sum is *linear* in `F` and errors in `q` are multiplicative in `F`, whereas errors in `f` get
amplified by `1/(1-f)`.

**What they have that we don't**, in decreasing order of value to us:
1. **Data-based inference with a learned prompt subtraction** (§3.1, Eq. 15–16). They train the
   fake factor *on data*, removing prompt contamination with two auxiliary data/MC density-ratio
   classifiers `r^T, r^L` and reweighting per event by `w(1 - 1/r)`, with a soft-absolute output
   activation forcing `r >= 1` so **the weights cannot go negative by construction**. That
   directly targets our dominant systematic: we use a per-path `N_data/N_MC` scale from the
   Z-enriched region, subtract gen-matched prompt MC, and **zero negative bins with a warning**,
   with the residual absorbed into `PROMPT_UNC = 0.3`.
2. **The validation protocol** (Figs. 12 vs 13, 14 vs 15). Closure in the CR *and* extrapolation
   to an orthogonal SR, shown in variables the binned method is **not** parameterized in. Their
   binned FF closes in `pT`/`|eta|` (its own axes) and mis-models `m_T`; the ML one closes in
   all five. This is the answer to our "no neutral pT axis" problem: the honest differential
   axes are the ones *neither* method uses — for us `ev_mt`, `ev_met`, `ev_njet30`, `ev_ht`,
   `ev_awayjet_pt`, all already columns in the parquet.
3. A `lambda*q^2` penalty on the log-density-ratio — scanned below, and it does **not**
   reproduce for us.

Not worth copying: the ResNet + feature-embedding + AdamW architecture (an artifact of putting
NNs on tabular data; xgboost has no vanishing-gradient problem to mitigate), and their
uncertainty treatment, which they leave as future work while we already have a full model.

### [2026-07-30] **BUG — `base_score` is silently dropped on model load** (fixed)

Found while implementing the q-space change; it had corrupted nothing that was reported, but
only by luck, and it would have.

`xgboost 2.1.4` does not round-trip `base_score` through the sklearn wrapper. The saved JSON
holds the fitted value (`learner_model_param.base_score = 0.25796`, the weighted tight
fraction), but `XGBClassifier().load_model(path)` leaves the booster at the constructor default
**0.5**, with no error and no warning. Passing `base_score=` to the constructor does not help —
`load_model` overwrites it. The fix, now in `evaluate.load_model()`, is to re-read the value
from the JSON and `booster.set_param("base_score", v)` after loading.

Why it matters here specifically: a constant offset to the margin does not change the ranking,
so **AUC, ROC and every ranking metric are unaffected** — the bug is invisible to the usual
checks. But the direct model's output *is* the fake rate, and the offset is
`logit(0.5) - logit(0.258) = +1.06` in `q`, inflating every prediction. Measured on QCD
held-out rows: sum rule `sum w*p / sum_tight w` = **1.5307 unpatched vs 1.0029 patched**, the
latter reproducing exactly what `train.py` recorded in-memory at training time. Verified
identical for the raw `xgb.Booster` loader, so it is not a wrapper-only issue.

Two lessons kept: (a) `train.py`'s in-memory metrics and any metric recomputed after a reload
must be cross-checked, since only the reload path is affected — the stored `_sum_rule` in the
`.meta.json` was right while the reloaded model was wrong; (b) multiclass models store a
*vector* `base_score` that `set_param` cannot take, so the flavor models still carry a
per-class offset — O(1e-3) against class priors of O(0.1), and irrelevant after
`prior_correct`, but `load_model` now warns rather than passing silently.

### [2026-07-30] q-space rewrite + the paper's `lambda` regularization scan — also negative

`closure.py` now takes the **raw margin** (`predict(X, output_margin=True)`) rather than
`predict_proba`, carries `q` alongside `f`, computes the matrix form as `sum_fail w*exp(q)`
exactly, reports bkg.cc's clamped value as a separate column, quantizes in `q`, and prints a
new per-score-bin **matrix-form calibration table** (`matrix_calibration()`). Two of those are
real changes, one is not, and the difference is worth stating so nobody re-derives it:

- **Real:** precision. Recovering `F` from a float32 probability is worthless once `f -> 1`.
  The leak-demo model lives entirely there, and its matrix number moves 0.0186 -> **0.0160**
  when computed from `q` instead of `f`. (Conclusion unchanged: it still predicts ~2% of the
  fake background. The leak demo stands.)
- **Real:** the calibration table. The rate form needs `f` to equal the mean tight fraction of
  its own level set; the matrix form needs `F` to equal `n/(d-n)` there, and by Jensen those
  are *different* conditions once `f` varies within the bin. That is why `direct` closes at
  1.0029 in `rate` and 0.9585 in `matrix` at the same time. The table localizes it: on QCD
  held-out the `F` ratio is 0.950/0.963/0.969 in the three bins that carry the sum, i.e. a
  systematic few-% `F` deficit, not noise.
- **Not real:** the quantization grid. Quantile edges are invariant under the monotone map
  `q <-> f`, so binning in `q` puts exactly the same muons in the same bins. What it fixes is
  what each bin *stores* — measuring both `n/d` and `n/(d-n)` from the same two sums makes one
  lookup simultaneously calibrated for both forms.

**The `lambda` scan (the paper's §3.2 lever) fails, in the same direction the depth scan did.**
`reg_lambda` is L2 on leaf weights, which are additive contributions to `q`, so it is the
closest xgboost analogue of their `lambda*q^2`. It needs to be huge here: per-leaf Hessian sums
are O(1e5) after the weight normalization, so `reg_lambda` up to 1e4 changes nothing at all
(AUC 0.8877 -> 0.8877). Scanning where it does bite:

| config | AUC | rate self | rate TTLJ | mat self | mat TTLJ | mat corr | top1% self |
|---|---|---|---|---|---|---|---|
| `flat` (bkg.cc) | — | 1.0060 | 1.0949 | 1.0108 | 1.1320 | 1.1199 | 0.150 |
| lambda = 1 (baseline) | 0.8877 | 1.0029 | 1.0239 | 0.9585 | 1.0388 | 1.0838 | 0.404 |
| lambda = 1e4 | 0.8877 | 1.0031 | 1.0406 | 0.9403 | 1.0365 | 1.1023 | 0.379 |
| lambda = 1e5 | 0.8677 | 1.0046 | 1.0903 | 0.9123 | 1.0703 | 1.1732 | 0.319 |
| lambda = 1e6 | 0.8334 | 1.0026 | 1.2082 | 0.8713 | 1.1588 | 1.3300 | 0.187 |
| lambda = 1e7 | 0.7848 | 1.0204 | 1.4784 | 0.9742 | 1.5806 | 1.6224 | 0.085 |
| lambda = 1, ~q100 | — | 1.0021 | **1.0066** | 1.0793 | 1.1085 | **1.0271** | 0.469 |
| lambda = 1e4, ~q100 | — | 1.0026 | 1.0190 | 1.0478 | 1.0952 | 1.0452 | 0.442 |
| lambda = 1e6, ~q100 | — | 1.0028 | 1.0728 | 1.0154 | 1.1247 | 1.1076 | 0.333 |
| `max_delta_step` 3 | 0.8884 | 1.0028 | 1.0241 | 0.9569 | 1.0384 | 1.0851 | 0.404 |

**1. More regularization monotonically destroys transfer** — rate TTLJ 1.0239 -> 1.4784, i.e.
worse than the flat map by lambda = 1e5. `max_delta_step` does nothing measurable.

**2. But it does exactly what it advertises for *conditioning*** — quantized matrix
self-closure improves monotonically with lambda (1.0793 -> 1.0478 -> 1.0154) and `top1%` falls
0.469 -> 0.333. So the knob is real; it just costs far more in transport than it buys in
conditioning. Same verdict as the depth scan, reached by a different route: **the transport
gain lives in the model's fine structure, and every attempt to blunt that structure loses more
than it gains.** Keep lambda = 1, depth 6.

**3. Why the paper sees the opposite is not a contradiction.** Their extrapolation is CR -> SR
*within one dataset* (`m_T < 60` -> `m_T > 60`), where the target is a sparsely-populated
extension of the training region and shrinkage buys stability. Ours is QCD -> TTLJ, a different
*process* whose fake composition differs (61-66% b vs 88-89% b), where the fine structure is
precisely what tracks the change. Their footnote to §4.2 says the same thing from the other
side: a better parameterization (mother-parton pT) would remove the topology dependence they
regularize against. Do not expect their lambda = 0.01 to transfer to a cross-process test.

**Current best config, both forms, corrected:** `direct` (53 WP-free vars, depth 6, lambda 1)
quantized to 100 score bins — rate **+0.5%**, matrix **+2.7%**, against the flat map's +8.8%
and +12.0%. The matrix number is better than the +4.9% reported yesterday because it is now
computed exactly instead of through bkg.cc's clamp. It is still **not** a number to quote:
the calibration map is measured on the full QCD parquet, 80% of which the model trained on
(visible as the un-quantized matrix self-closure 0.9585 *overshooting* to 1.0793 once
quantized). Cross-fitting that map is the next job and is unchanged by any of the above.

*Plot-file caveat found 2026-08-02:* `evaluate.py` names its diagnostics
`<plot>_<target>_<feature_set>.png`, and every `reg_lambda` / `max_delta_step` variant shares
`feature_set = direct`. So `plots/training_history_tight_direct.png` is **not** the baseline —
it is whichever variant was trained last (currently lambda = 1e7, titled "mean OvR AUC 0.7848").
Same for any other `*_tight_direct.png`. Read the `.meta.json` next to the model, not the plot,
when attributing a number to a config.

### [2026-08-02] Deck: `BDT/fakeflavor/plots/direct_fr/20260802_SiunKim_directFR.{pptx,pdf}`

18 slides on the direct route — how it is trained, the two report forms, the transfer result,
the two problems (WP leakage, `f/(1-f)` conditioning), both failed fixes, the `base_score` bug,
and what is not yet quotable. Summary figures next to it in the same directory, all redrawn
from the measured tables above (`scratchpad/deck/make_figs.py`, not committed).

### The finding cuts both ways — read this before deciding what Phase 2 is

The non-flatness says **FR is not constant within a true flavor** at fixed (conePt, |eta|):
there is real, exploitable structure in the muon-level variables *beyond* flavor. That is
good news for the original question ("can more variables give a more accurate FR?") and
bad news only for the specific mechanism of routing it through a flavor label.

`Σ_f p_f(x)·FR_f` is a *projection* of `E[tight | x]` onto the flavor basis, and it is only
correct if FR depends on x solely through flavor. The diagnostic shows it does not. So:

- **the flavor route needs p_f uncorrelated with tightness at fixed flavor**, or the
  projection is biased — and `nowp` was measured and **does not deliver that** (see the
  section above: c 9215/9 → 7425/9, b got worse). Taking this route anyway means either
  accepting a known bias for c, or finding a decorrelation scheme (adversarial /
  distance-correlation penalty) — a much bigger project than a feature-set swap.
- **the direct route is strictly stronger and has no leakage problem by construction** —
  train `P(tight | loose, x)` on the same ntuple (`isTight` is already a column) and use
  the calibrated probability *as* the fake rate. No intermediate label, no projection, and
  ~~every variable is fair game including conePt and miniiso~~. The cost is that it is no
  longer interpretable as a flavor composition, and the closure/transfer tests have to be
  redone against data rather than against the per-flavor FR maps.

  > **Struck through 2026-07-30 — that clause was wrong, and wrong in the direction that
  > would have silently broken the result.** "No leakage problem by construction" holds
  > only for the *bias* of the estimator. `bkg.cc` evaluates the FR on the muon that
  > **failed** tight (`Analyzers/src/bkg.cc:237-252`, categories TL/LL) as `e = f/(1-f)`,
  > so a model that is allowed to see miniiso or SIP3D returns f ~ 0 on every muon it is
  > ever asked about and the fake estimate collapses. conePt is a function of miniiso and
  > is therefore also excluded. See the section below for the measured numbers.

Recommendation: finish Phase 1 as scoped (it answers "are flavors distinguishable?" —
provisionally **yes**, in-cell AUC 0.75-0.93), but decide Phase 2 between these two routes
on the evidence, rather than defaulting to the flavor-weighted FR.

**Updated recommendation after the `nowp` control (2026-07-29): go direct.** The flavor
label has been shown to be inseparable from tightness in the available observables, so it
is a lossy and biased intermediate. `P(tight | loose, x)` needs no new ntuple — `isTight`,
the physical weight and every feature are already columns in the same parquet — so the
switch costs one new training script, not a new production run. Keep the flavor classifier
as the *interpretation* tool (it is what tells the story "QCD fakes are 61-66% b, TTLJ
88-89% b, and FR(uds) ≫ FR(b)"), not as the thing in the FR path.

## Step E — what "success" means

1. **Separability** — per-class AUC + confusion matrix, inclusive **and inside each
   (conePt, |eta|) FR cell** (conePt is itself an input; the gain over the existing 2D
   binning is the real figure of merit). b-vs-light should be easy; the interesting
   questions are **c-vs-b and g-vs-uds** — those two confusions are what decide whether
   this is usable.
2. **Sample transfer** — train QCD → test TTLJ and vice versa. If the model only learned
   the sample prior, this collapses. Decisive Phase-1 test.
3. **Physics closure (the number that matters)** — using the per-flavor FR maps that
   already exist in `{era}_flavor*/` (read as `fakerate_flavor.py` does), compare in TTLJ:
   truth-weighted `Σ_f n_f·FR_f`, BDT-weighted `Σ_f p_f(x)·FR_f`, and the flat QCD
   `FR(pt,|eta|)`. If BDT-weighted tracks truth-weighted and both beat the flat map,
   Phase 2 is worth doing.
4. **Data/MC of the inputs** — gate on *deployment*, not Phase 1. `track.cc`'s `NoSel/`
   block already fills the muon-level ones for data+MC; matched-jet features would need
   adding there.

Note on the dropped classes when closing the loop: pileup and unmatched still *exist* in
the data, they are just not BDT classes. In the flavor-weighted FR they have to be handled
explicitly — `unmatched` is directly taggable in data (`mu_hasJet == false`) and can carry
its own measured FR; `pileup` has no data-side handle, so at Run2 rates (0.36% of fakes,
but FR ~0.7) it is a ~0.3% effect on the weighted FR and belongs in the systematic, not
the model.

## Phase 2 (not now)

`bkg.cc` hard-codes a 2D `TH2D` `"FR"` on (ptCorr, |eta|) and `GetFakeRate(const Muon&)`
takes only a muon — so the soft-weighted FR `Σ_f p_f(x)·FR_f`, a *per-muon scalar*,
cannot ship as a `TH2D`. Needs either ONNX inference inside `bkg.cc` (template: the
existing `HiggsBDT` ONNX path + `BDT/scripts/export_onnx.py`) or a 2D map collapsed under
an assumed composition. Also note `bkg.cc`'s hard-coded FR paths (`{era}_2/`,
`{era}_data/`) are **stale** — those dirs moved to `fake/0706/`.

## Verification  [DONE for steps 1-2, 2026-07-29]

- `g++ -fsyntax-only` clean on `fake.cc`, `SKNanoLoader.cc`, `AnalyzerCore.cc`, `Muon.cc`;
  `scripts/build.sh` exit 0 (rootcling regenerated, `fake.h`/`SKNanoLoader.h` reinstalled).
- Single-file interactive run on `QCD_Pt-170to300_MuEnriched` / 2016preVFP with
  `Userflags = ["MakeTree"]`: no `Branch ... not found` warnings, no Run2 crash on the
  Run3-only `Muon_svIdx`, `WriteHist` reports `Writing tree: mu`.
  → **tree `mu`, 100 branches**, 109 rows on that slice.
- **Ntuple vs histogram (decisive check) — exact match.** Selecting
  `pass_muX && ev_hasAway40` and weighting by `w_muX` reproduces
  `{Mu8,Mu17}/MeasReg/Central/{loose,tight}/fake_ptcorr_abseta` to 3e-8 relative,
  and every `.../recoflav/<f>/fake_ptcorr_abseta` sum bin-for-bin
  (b/c/uds/g/pileup/unmatched, both paths). The tree therefore describes exactly the
  population the histogram FR is measured on.
- Branches that came out **constant** (and are useless as features — exclude in
  `features.py`): `mu_pogMedium`, `mu_isTracker`, `mu_isPFcand` (all implied by the
  loose WP's POG-medium requirement), `ev_hasAway30` (the gate itself), and in a
  pure-QCD file `isPrompt`/`isData`/`mcweight`.
- Still to do: **BDT** 5-fold StratifiedKFold (reuse `src/cv.py`); split **by event**
  (`run`/`lumi`/`evt` columns), not by muon. Report to `BDT/fakeflavor/{plots,reports}/`;
  deck via the python-pptx + PdfPages recipe.

# ==========================================================================
# ttbar 4-class + QCD-vs-ttbar origin classifier            [DONE 2026-08-09]
#   goal: (a) repeat the flavor BDT on ttbar, (b) ask whether a fake from a
#   b jet in ttbar is a different object from a fake from a b jet in QCD
# ==========================================================================

Full write-up: `BDT/fakeflavor/RESULTS.md` §12-14. Nothing was submitted and
nothing was re-skimmed — `ntuples_sv/TTLJ.parquet` (149,636 rows) was already on
disk, and the QCD-vs-ttbar label is the `sample` string the skim already writes.
21 models trained on ai-tamsa1 in ~12 min total.

## `mu_pogTight` removed from every feature set — it was worth nothing

It was in every set except `direct_map`, including `nowp` and `direct`, whose
purpose is to hold no tight-WP information, and it ranked 9th by gain in `slim`.
Removing it costs +0.0001 / -0.0002 / -0.0002 on slim / no-ML / with-ML. A high
gain rank measures *use*, not irreplaceability. Sizes are now full 85, noml 75,
slim 32 (assert updated), nowp 69, direct 64, plus the new `local` 71.
**Side effect: every `direct_*` model is stale** — they were trained with a hard
muon-ID flag inside `direct`. Retraining them is the cheapest open item there is.

## (a) ttbar 4-class: it transfers

mean OvR AUC, QCD `_v3` -> TTLJ `_v3`: slim 0.7434 -> 0.7484, no-ML 0.7563 ->
0.7589, with-ML 0.8104 -> 0.8081. All within 0.005 on a process with a 16x
smaller sample and an almost inverted flavor prior (ttbar fakes 89% b vs QCD
65%), so **the separation is a property of jets, not of the QCD prior**. The
tagger ablation survives too: full - noml is -0.0493 in ttbar vs -0.0541 in QCD.
Per class the two trade: ttbar b easier (.8903 vs .8800), ttbar c harder (.7835
vs .8050). TTLJ has 1,463 uds and 1,076 g rows in ALL of Run 2 (~290/~215 in the
test split, so ~±0.03) — do not read those two columns as physics.

## (b) origin classifier: at fixed flavor, QCD and ttbar fakes differ

New `--target origin --flavor {b,c,uds,g}`, binary QCD vs TT with class_balance
undoing the 1.1e5 physical-weight ratio. AUC for TT:

Same three nested feature sets as the flavor decks, plus two control arms:

  slice   withML85  noML75   SLIM32   (local71)  (conePt,|eta|)
  b        .9532    .9532    .7581     .7773      .7082
  c        .9556    .9563    .6717     .6630      .6091
  uds      .9431    .9433    .5904     .5899      .5483
  g        .9578    .9579    .7238     .7571      .5606
  mean     .9524    .9527    .6860     .6968      .6065

The ablation reads OPPOSITE to the flavor target: the ML outputs are worth
+0.0003 (nothing — flavor is held fixed, so the taggers have nothing to say),
and the 75->32 step costs -0.2667 because that step drops the EVENT block.
with-ML and no-ML both keep ev_ht / ev_njet* and their gain is ev_ht (3241) +
ev_njet30 (2171) in every slice: at 0.95 they tag the event, not the fake.
**`slim` is the physics arm**, and `local` (= full minus every ev_*) agrees with
it to 0.02-0.03, so the ev_met/ev_mt still in slim are not doing the work.

The decisive number is per FR cell, not inclusive: inside every one of the 21
(conePt, |eta|) cells `bkg.cc` uses, `origin_b_local` scores **0.66-0.72** —
barely below its inclusive 0.777 (`origin_b_slim` mean cell AUC 0.662, c 0.630).
uds and g have NO per-cell table: under 50 ttbar rows land in any one cell
(training-split effective N on the ttbar side is 903 and 689). So the QCD/ttbar difference is NOT mostly the
conePt spectrum and survives conditioning on the exact binning the FR is
measured in. Controls: `origin_b_direct_map` gives 0.486-0.551 inside the five
narrow conePt bins (= 0.5 by construction, so the machinery is not manufacturing
separation) and 0.58-0.60 in [30,50) and [50,100), which are wide enough that
conePt still discriminates inside the cell — still far below local's 0.66-0.72; and
the weighted era composition of the two samples agrees to a percentage point, so
no era-sensitive variable is carrying it.

This is the **mechanism behind the §7/"per-flavor FR is not sample-universal"
finding** (FR_b 0.4831 QCD vs 0.4501 TTLJ; FR_c 0.4507 vs 0.3540): the
populations being averaged are not the same population. It does NOT revive the
flavor-weighted FR — that died on its truth-flavor ceiling (1.127 vs flat 1.095)
and better flavor information cannot fix a ceiling.

**It is a warning for the live direct route.** `P(tight|loose,x)`'s whole claim
is its QCD->TTLJ transfer (1.023 vs flat 1.095), and this says the two domains
are separable at 0.78 in the very variables that model trains on. Next: reverse
transfer (train on ttbar, apply to QCD), and check whether the transfer degrades
in the cells where `origin_*_local` is largest.

# ==========================================================================
# Classifier optimization round — SV features + gen label   [DONE 2026-08-01]
#   goal: raise the 4-class accuracy itself, decoupled from any FR application
# ==========================================================================

Headline: mean OvR AUC **0.8094 → 0.8170**. Two changes survived; everything else
measured <= 0. Deck + figures: `BDT/fakeflavor/plots/flavor_opt/`
(`20260801_SiunKim_flavorBDT.{pptx,pdf}`).

## What was measured (QCD-only A/B, identical event splits, so deltas are attributable)

| change | Δ mean OvR AUC | verdict |
|---|---|---|
| label `flavor` → `genflavor` | **+0.0054** | KEPT |
| add 12 `sv_*` features | **+0.0008** (b-vs-c **+0.0041**) | KEPT |
| drop `jet_qgl` | −0.0001 | rejected — it is *noise*, not dead weight |
| hierarchical 3-model (HF/LF → b:c, uds:g) | 0.0000 | rejected |
| max_depth 6 → 8 / → 10 | −0.0015 / −0.0029 | rejected, depth 6 already optimal |
| max_bin 512 + colsample 0.6 + mcw 30 | −0.0011 | rejected |

Production model `BDT/fakeflavor/data/models/flavor_genlabel_sv.json` (QCD+TTLJ,
gen label, 86 features, 549 trees): b 0.8848 / c 0.8081 / uds 0.7839 / g 0.7913.

## Hard-won facts (do not re-derive)

- **The reco-jet label is the single biggest error source, and only for g.**
  reco-b and reco-c are ~100% pure against the gen jet; reco-uds is 90%; **reco-g is
  54% gen-g, 24% gen-uds, 22% no gen jet at all.** Switching to `genflavor` costs only
  1.33% of rows and moves the g class +0.015. Adding gen-unmatched as a 5th class is
  *worse* than dropping those rows — do not retry it.
- **SV pays only where b/c physics lives.** b-vs-c 0.8404 → 0.8445; uds-vs-g
  0.6928 → 0.6925 (nothing). `sv_masssum` ranks 3rd–4th overall in gain importance.
  Weighted fraction of rows with a matched SV: b 0.39, c 0.11, uds 0.04, g 0.05 —
  (the 0.318 inclusive number is unweighted and dominated by b; don't quote it per class).
- **uds-vs-g is at its information ceiling, and the ceiling is kinematic.** 86 features
  give 0.693 vs 0.670 from `jet_deepjet_qg` alone. AUC rises 0.667 (jet pT 20–25) →
  0.737 (60–100) and the sample's median jet pT is 30 GeV. No jet-level tagger will
  move this; only more uds/g statistics or Run 3 ParticleNet would.
- **The model was never the bottleneck** — every architecture/HP knob is exhausted.
  b-vs-c is the only sub-problem that still responds to new information.
- **Caveat that blocks reuse:** `mu_softMva` is the 2nd most important feature and the
  isolation / IP variables sit right behind it. Those *define* the tight WP, so these
  AUCs are NOT transferable to a fake-rate application — same leakage that killed the
  flavor-weighted-FR route. Pure-classifier numbers only.

## Plumbing changes

- `fake.cc::fillTreeRow` fills 12 `sv_*` branches (leading SV in dR<0.4 of the jet axis,
  chosen by max `dlenSig`; sentinels when none).
- `SKNanoLoader.{h,cc}`: SV binding moved out of the `Run == 3` branch — **the SV
  collection exists in NanoAOD v9 too**; only the counter type differs (v9 `UInt_t`
  `nSV_RunII` vs v12 `Int_t`), which `BindScalarWithRunCheck` absorbs. Run2 previously
  did `SV_*.resize(0)`.
- `SV_pAngle` in v9 clusters at ~π (median 3.121), not ~0. That is the v9 convention,
  verified against the raw branch — not a misaligned read.
- Ntuple tag **`bdtflav2`** = `bdtflav` + the sv_* block, same selection, row-for-row
  comparable (2,498,382 rows both). Parquet in `BDT/fakeflavor/data/ntuples_sv/`;
  the old `ntuples/` set is left intact.
- `features.py`: `SV_FEATURES` appended at the **end** of `FEATURES_FULL` (order is
  load-bearing) and folded into `TAGGER_FEATURES` so `notag` stays a real control.
- `train.py`: new `--target genflavor`, which drops the no-gen-jet rows.

## If another round is wanted

Remaining unread jet branches, in expected-value order: `Jet_cRegCorr`/`cRegRes`
(c-specific by construction), `Jet_bRegCorr`/`bRegRes`, then DeepCSV
(`btagDeepB`/`btagDeepC`) and `btagCSVV2` as independently-trained taggers. Expect the
same shape as SV — a few ×0.001 on b-vs-c, ~0 on uds-vs-g — for one production cycle.

# ==========================================================================
# Per-flavor FR v3 — pileup category, uds merge, 2016+2022  [DONE 2026-07-28]
# ==========================================================================

- `FlavorTag(parton, hadron, isRecoJet)`: matched **reco** jet with parton==0 &&
  hadron==0 → **`pileup`** (no generator counterpart). Reco-only on purpose — the
  GenJet collection has no pileup jets, so 0/0 there is just "no parton match".
  Measured: pileup = 0.0% (gen) vs 0.2–4.7% (reco). `unmatched` = no jet in dR<0.4.
- Plots now draw **b / c / uds / g** (u+d+s merged) →
  `plots/{QCD,TTLJ}/{muon,electron}/{gen,reco}/{era}_{etaInt,eta1..3}.pdf` (128 PDFs).
- Ran **2016preVFP/postVFP + 2022/2022EE**, QCD & TTLJ, muon & electron (16 DAGs).
  2022 needed no code change (`Run == 2` guards already there; Run3 electron MVA WPs
  at elecfake.cc:44). QCD rerun ⇒ now shares the fake-gate with TTLJ.

## Findings
- Composition (loose fakes, low-pT path): muon QCD = b 61–66%, c 25–28%;
  muon TTLJ = **b 88–89%**. Electron QCD(EMEnriched) = uds 62–66%, g 20–23%;
  electron TTLJ = **b 63–69%**. ttbar is the heavy-flavor complement to QCD.
- **Integrated per-flavor FR must NOT be compared across samples.** FR(b) reads
  0.31 (QCD) vs 0.18 (TTLJ) integrated, but bin-by-bin the two agree to 1.03–1.31
  over ptcorr 10–50 GeV — the gap is purely the pT spectrum (QCD b-fakes pile into
  the low-ptcorr bins where FR≈0.54). ⇒ FR(flavor, pT, |eta|) is ~universal across
  samples, which is what the flavor-weighted FR assumes.
- Run3 caveat: reco `unmatched` jumps 4% → 13% (QCD muon) in 2022 vs 2016.
- Electron QCD has ~0 b statistics (zero tight events in several ptcorr bins) →
  take the electron b-flavor FR from TTLJ, not from EMEnriched.
- Ops gotcha: `condor_q <dagman-id> -totals` reports 0 for a LIVE dag — poll
  `condor_q -af JobBatchName | grep -c fake` and check output mtimes instead.

## Deck
- `plots/20260728_flavorFR_QCDvsTTLJ.{pptx,pdf}` — 65 slides, one 2-up
  QCD | TTLJ comparison per (lepton, gen/reco, era, eta slice).
- NOTE: `plots/*/*/*/{preVFP,postVFP}_*.pdf` (64 files, 2026-07-16) are **stale**
  pre-v3 plots kept from the old era-label naming; the deck excludes them.

# ==========================================================================
# Per-flavor FR v4 — full-Run2 stacking            [DONE 2026-07-28]
# ==========================================================================

Single-era TTLJ light/gluon FR was uninterpretable: uds/gluon jumped between 0
and 1. Three separate causes, all now addressed.

1. **Genuinely tiny statistics.** ttbar fake muons are 89% b / 9% c; a light or
   gluon jet faking a muon needs pi/K decay-in-flight or punch-through, and TTLJ
   has no muon filter while QCD MuEnriched is built for one. 2016preVFP gen
   gluon had N_eff = 42 spread over 24 cells (<2 effective events/cell).
2. **Plotting defects that amplified it.** Empty bins (loose=0) were drawn as
   `0.000 +- 0.000`; the binomial error collapsed to exactly 0 when T=0 or T=L.
   Fixed: `make_ratio1D` returns the raw projections and `make_combined_graph`
   builds the ratio per bin with a **Clopper-Pearson** interval on N_eff, and
   skips bins below `MIN_NEFF` (default **10** — at N_eff<=2 a bin lands on
   exactly 0.000 or 1.000 with a meaningless bar).
3. **Why gen scattered ~1.9x more than reco.** Same muons, same total N_eff
   (20785 gen vs 20786 reco) — only the categorization differs. A b-jet that
   fails ghost b-tagging picks up partonFlavour=21 from nearby radiation, so
   gluon gets N_eff 42 (gen) vs 152 (reco).

**Fix: stack whole run periods.** `ERAS` (comma-separated era dirs) + `ERA_LABEL`
in both `fakerate_flavor.py` / `elecfake_flavor.py`. Stack only WITHIN a period —
across Run2/Run3 the b FR shifts ~10% and uds 20-30 GeV differs 0.07 vs 0.19
(~2 sigma); within a period b agrees to ~2%.

- Ran TTLJ + QCD for **2017 and 2018** (2016 already existed) ⇒ full Run2.
  Muon N_eff x3.8 (uds 361→1482, gen gluon 75→286); QCD x2.7.
  Result: uds is now smooth and monotonic; gluon usable to ~50 GeV.
- `undef` (unmatched) **dropped from the drawn list** — its statistics collapse
  with pt (N_eff 1746 → 2 across the muon range), so the high-pt end was noise.
  Definition + palette entry kept; re-add to `flavors` to restore.
- Flavor palette aligned with the track/rawjet scripts: uds kAzure+1, c kGreen+2,
  b kMagenta+1, g kRed+1, pileup kOrange+7, undef kGray+2 (dashed).
- Deck: `plots/20260728_flavorFR_Run2_QCDvsTTLJ.{pptx,pdf}` — 17 slides.

## Findings (Run2)
- QCD muon **pileup FR ~0.7 and rising with pt**, opposite to every other flavor.
  Consistent with the physics: if the jet matched to the muon is a pileup jet the
  muon is not inside hard-scatter hadronic activity, so it tends to be isolated
  and passes tight.
- Electron pileup is well measured (N_eff 126–1102/bin) and sits ~0.05 above
  gluon; muon pileup is thin (N_eff 37–72) — read it with care.

# ==========================================================================
# TODO — ELECTRON fake rate per parton flavor (elecfake.cc)   [DONE v1, 2026-07-09]
# ==========================================================================

## Result summary (v1)
- `elecfake.cc`/`elecfake.h`: added rawJets member + FlavorTag / GenJetFlavor /
  RecoJetFlavor(Electron); fills MC-only `{path}/MeasReg/Central/{loose,tight}/{genflav,recoflav}/{b,c,s,d,u,g,unmatched}/fake_ptcorr_abseta`.
- Submitted `elecfake -i 'QCD_Pt-*EMEnriched*'` both eras `--tag eflavor`; hadded to
  `/data6/Users/snuintern2/fake/{era}_eflavor/QCD.root` (8 pt bins each).
- Plots: `Tools/elecfake_flavor.py` → `Tools/plots/eFR_flavor_{genflav,recoflav}_{era}_{etaInt,eta1..3}.png`.
- **Finding:** EMEnriched is dominated by light-quark/gluon fakes (conversions / π0
  overlap) → g,s,u,d,c well-measured (~0.10–0.25 at mid ptcorr), but **b (and c at
  high ptcorr) have poor stats / large errors** (heavy flavor rare in EM-enriched).
  Reco agrees with gen; reco puts gluon a bit higher (same as muon). For a
  flavor-weighted electron FR the heavy-flavor bins may need merging or more MC.

--- original electron plan below (for reference) ---

Mirror the (done) muon per-flavor study onto the **electron** chain: split the
QCD **EMEnriched** electron fake rate FR(ptcorr, |scEta|) by the flavor of the
fake electron's **source jet** — separate FR maps for **b, c, s, d, u, gluon**
(+ `unmatched`). MC-only (needs gen truth). Same design decisions as muon:
**both** gen- and reco-jet tags, **six** categories kept separate, aimed at a
flavor-weighted electron FR.

## Electron-specific facts (grounded in current elecfake.cc / elecfake.h)
- QCD sample = **EMEnriched** (not MuEnriched): `-i 'QCD_Pt-*EMEnriched*'`.
- **|scEta| everywhere** for binning (`el.scEta()`), but spatial jet matching
  uses ordinary `el.DeltaR(jet)` (physics eta/phi), dR<0.4 — same as muon.
- Binning (current code): ptcorr `{15,17,20,25,35,50,100,200}` (meas starts 15
  after the pt>15 alignment), |scEta| `{0,0.8,1.479,2.5}`. Overflow → last ptcorr bin.
- 3 trigger paths Ele8/Ele12/Ele23 → **two** stitch boundaries: Ele8 [15,20],
  Ele12 [20,35], Ele23 [35,100] (muon had a single boundary at 30).
- `elecfake_pre.py` reads `{path}/MeasReg/Central/{wp}/fake_ptcorr_abseta` — so fill
  per-flavor in **MeasReg/Central** (same region choice as muon).
- Source jet is removed from `jets` by `JetsVetoLeptonInside` → must match against
  **rawJets** (make it a member; currently a local at elecfake.cc:119) and `genJets`.

## Steps (mirror the muon work exactly)
1. **`elecfake.cc` / `elecfake.h`** (edit in place; only Analyzers/):
   - [ ] Add `RVec<Jet> rawJets;` member; assign `rawJets = GetAllJets();` at top of executeEvent.
   - [ ] Add helpers `FlavorTag(parton,hadron)`, `GenJetFlavor(const Electron&)`,
         `RecoJetFlavor(const Electron&)` — identical mapping to fake.cc
         (hadronFlavour 5→b,4→c; |partonFlavour| 21→g,3→s,2→u,1→d; else unmatched).
   - [ ] In the MeasReg fill block (elecfake.cc ~line 299), add MC-only, Central-only:
         `base + "/" + tag + "/{genflav,recoflav}/" + flavor + "/fake_ptcorr_abseta"`.
   - [ ] Syntax-check, then **build** in container (`rebuild.sh`, Analyzers-only) per global CLAUDE.md recipe.
2. **Submit** (tamsa2): `elecfake -i 'QCD_Pt-*EMEnriched*' -e 2016preVFP/2016postVFP -n 40 --tag eflavor`.
   (tag `eflavor` → gv0 `elecfake/{era}_eflavor/`, keeps it distinct from muon `_flavor`.)
3. **hadd** QCD_Pt-*_EMEnriched → `/data6/Users/snuintern2/fake/{era}_eflavor/QCD.root`.
4. **Plots** → `Tools/plots/`: new `elecfake_flavor.py` (twin of `fakerate_flavor.py`),
   with the 3-path/two-boundary stitch and |scEta| bins. Gen + reco, eta-int + per-|scEta| slices.

## Confirm before I start (so we're aligned)
- **Output/tag naming**: submit `--tag eflavor`, hadd to `fake/{era}_eflavor/QCD.root`
  — OK? (Can't reuse `_flavor`: that's the muon QCD.root.)
- **New script** `elecfake_flavor.py` (like the muon got `fakerate_flavor.py`) vs
  folding electron mode into the existing one — which do you prefer?
- Anything else different for electrons (e.g. also fill the **Inclusive** region,
  since the official electron QCD-MC FR was noted as Inclusive)? Default: MeasReg/Central only, matching muon + what elecfake_pre.py reads.
- Note: elecfake.cc currently **needs a rerun anyway** (pt>15 alignment + EMEnriched
  120to170 xsec fix) — this flavor run will pick those up. Prompt/data elecfake FR is unaffected.

# ==========================================================================
# (DONE) Muon fake rate per parton flavor  — reference for the above
# ==========================================================================

# TODO — Muon fake rate per parton flavor (b / c / s / d / u / gluon)

Status: **DONE (v1, 2026-07-09).** Muon FR matches official MeasFakeRateV4; this
branch measures how the QCD-MC fake rate depends on the source-parton flavor.

## Result summary (v1)
- Analyzer `fake.cc` fills both gen- and reco-jet flavor trees at
  `{path}/MeasReg/Central/{loose,tight}/{genflav,recoflav}/{b,c,s,d,u,g,unmatched}/fake_ptcorr_abseta`
  (MC only). Muon matched to nearest gen/**raw**-reco jet dR<0.4 (rawJets = before
  lepton cleaning, since `jets` has the source jet removed).
- Submitted `fake -i 'QCD_Pt-*MuEnriched*'` both eras `--tag flavor`; hadded to
  `/data6/Users/snuintern2/fake/{era}_flavor/QCD.root`.
- Plots: `Tools/fakerate_flavor.py` → `Tools/plots/FR_flavor_{genflav,recoflav}_{era}_{etaInt,eta1..3}.png`.
- **Ordering at high ptcorr (eta-integrated, gen):** u,d (~0.20–0.24) > s (~0.17) >
  g (~0.11) > c (~0.09) > b (~0.05). Reco agrees on b<c and light-quark ordering;
  reco puts gluon higher (~0.24) since reco partonFlavour is noisier for light jets.
- Next: use these per-flavor FRs as the flavor-weighting inputs for a
  flavor-weighted FR (original motivation #3).

--- original plan below ---

## Goal

Split the QCD MuEnriched fake rate FR(ptcorr, |eta|) by the flavor of the
**source jet** of the fake muon: separate FR maps for **b, c, s, d, u, gluon**
(+ an `unmatched` catch-all). This is a **MC-only** study (needs gen truth →
data has none). Purpose: see how strongly the fake rate depends on quark
generation / gluon origin.

## Method / key facts (grounded in the current code)

- Truth handles that already exist in the framework:
  - `Jet::partonFlavour()` — PDG-style: `21`=gluon, `1`=d, `2`=u, `3`=s, `4`=c, `5`=b (sign = quark/antiquark).
  - `Jet::hadronFlavour()` — `5`=b, `4`=c, `0`=light (ghost-hadron matching, robust for b/c).
  - `Muon::GenPartFlav()` — `1`=prompt, `5`=from-b, `4`=from-c, `3`=light/tau, `0`=unmatched (coarse; b/c cross-check only).
- **Flavor assignment plan**: match the fake muon to its *source jet* (nearest
  selected jet, dR < 0.4 — NOT the away jet). Category from that jet:
  - b / c ← `hadronFlavour()` (5 / 4), more reliable than partonFlavour for HF.
  - g / s / u / d ← `|partonFlavour()|` (21 / 3 / 2 / 1) when hadronFlavour == 0 (light).
  - no matched jet → `unmatched`.
- Only meaningful for **fake** muons: gate on the existing `ltype`
  (`GetLeptonType(mu, gens)` → fake) in QCD MuEnriched MC.
- Cross-check b/c category against `Muon::GenPartFlav()` before trusting u/d/s/g split.

## Steps

### 1. Analyzer — `SKNanoAnalyzer/Analyzers/src/fake.cc`  (edit in place; ask before rerun)
- [ ] Add a helper: given the fake muon + selected `jets`, return a flavor tag
      string (`b/c/s/d/u/g/unmatched`) via the assignment plan above.
- [ ] In the muon FR fill block (`MeasReg/<syst>/<tag>/...`, around fake.cc:277),
      add a per-flavor subdirectory for fake muons, e.g.
      `.../MeasReg/Central/<tag>/flav/<flavor>/fake_ptcorr_abseta`
      (both loose & tight so FR = tight/loose downstream). Fill Central only to
      start; extend to away-jet systs only if needed.
- [ ] Keep binning identical to nominal: ptcorr {10,12,14,17,20,30,50,100} × |eta| {0,0.9,1.6,2.4}; overflow → last ptcorr bin.
- [ ] Syntax-check with the `rot && g++ -fsyntax-only ...` recipe (global CLAUDE.md) before handing back build/submit.
- [ ] Hand user exact `SKNano.py -a fake -i <QCD_MuEnriched> -e <era> -n <N>` for both eras (do NOT submit myself).

### 2. Post-processing — `Tools/`  (new twins, follow existing pattern)
- [ ] Add `fakerate_flavor_pre.py` / `fakerate_flavor_post.py` twins (preVFP/postVFP),
      modeled on `fakerate_pre.py`/`fakerate_post.py` — loop flavors, `Sumw2()`
      before `Divide()`, plain Divide, produce per-flavor FR2D + comparison plots.
      (Confirm naming with user — skeleton copies are usually pre-created.)
- [ ] Keep `_pre`/`_post` in sync (only ERA default / labels / suffix differ).
- [ ] Output FR maps + overlay of FR vs ptcorr per flavor (per |eta| slice) under `Tools/yes/`.

### 3. Validation / interpretation
- [ ] Sanity: sum of per-flavor loose/tight yields == inclusive fake histogram.
- [ ] Compare FR magnitude & pt-dependence across b vs c vs light(u/d/s) vs gluon.
- [ ] Check statistics per flavor (light quarks / gluon may be low) — merge bins if needed.

## Open questions for the user (before coding)
- Source-jet match: nearest selected jet within dR<0.4 — OK? Or prefer `GenJet` matching / `GenPartFlav`-only (coarser, no u/d/s/g split)?
- Which subdirectory layout do you want under `MeasReg` (`flav/<flavor>` vs a parallel tree)?
- Group u+d (hard to separate cleanly) or keep all six categories fully separate?
- Data has no truth — is this purely a MC dependence study, or feed into a flavor-weighted FR later?
