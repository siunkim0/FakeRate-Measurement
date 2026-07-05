# Muon Fake Rate 측정 & Matrix Method Background 추정

2016preVFP / 2016postVFP 데이터에서 muon fake rate 를 측정하고,
이를 이용해 tt-bar OS dimuon selection 의 fake lepton background 를
matrix method 로 추정하는 프로젝트.

## References

- **AN-25-154** (`/data6/Users/snuintern2/ref/AN-25-154_temp.pdf`) §6.1.2:
  fake rate 측정 방법 (Table 38: prescaled trigger 별 pT 영역).
  - Fig 44: **데이터**에서 잰 FR (prompt subtraction 포함) ← 데이터 측정의 비교 대상
  - Fig 46: **QCD MC**에서 잰 FR ← QCD MC 측정의 비교 대상 (Fig 44 와 헷갈리지 말 것)
- **AN2010/261** (`/data6/Users/snuintern2/ref/AN2010_261_v1.pdf`) §9:
  fakeable object (matrix) method 공식. prompt rate 는 p = 1 로 고정.

## Muon Working Points (fake.cc == bkg.cc, 반드시 동일해야 함)

| | 공통 | tight | loose |
|---|---|---|---|
| 컷 | POG medium ID, \|dz\| < 0.1 cm, TkRelIso(R03) < 0.4 | \|SIP3D\| < 3, miniIso < 0.1 | \|SIP3D\| < 5, miniIso < 0.6 |

cone-corrected pT = pT × (1 + max(0, miniIso − 0.1)).
FR(ptcorr, |eta|) = N(tight) / N(loose), binning: ptcorr {10,12,14,17,20,30,50,100} × |eta| {0,0.9,1.6,2.4}.

## Analyzers (SKNanoAnalyzer, `Analyzers/src/`)

### fake.cc — FR 측정용
- Prescaled trigger 경로 2개: `Mu8` (HLT_Mu8_TrkIsoVVL, ptcorr 10–30 담당),
  `Mu17` (HLT_Mu17_TrkIsoVVL, ptcorr 30–100 담당). 최종 FR 은 30 GeV 에서 stitching.
- **MeasReg** (측정 영역): loose muon 딱 1개 + electron 0 + away jet(pT>40, dR>0.7)
  + MET < 25 + MT < 25. → `{path}/MeasReg/Central/{loose,tight}/fake_ptcorr_abseta`
  (MC 는 gen-matching 으로 `/prompt`, `/fake` 서브디렉토리도 채움)
- **ZEnriched** (normalization 영역): OS tight pair + |mll−91.2| < 15 + jet pT>40.
  → `{path}/ZEnriched/mll` — prescale 로 인한 MC normalization 을 데이터로 정하는 데 사용.

### bkg.cc — matrix method background 추정용
- Selection: IsoMu24(||IsoTkMu24) trigger, **loose** muon 정확히 2개 (OS,
  leading pT>26, subleading>10), electron 0, mll>20, Z veto(±15), jet ≥ 2, MET > 40.
- 두 muon 의 tight 통과 여부로 TT / TL / LL 분류 후 weight (p=1, e = f/(1−f)):

  | 추정치 | leading term | full calculation |
  |---|---|---|
  | Npp | [TT] | [TT] − [TL]·e_fail + [LL]·e₁e₂ |
  | Nfp | [TL]·e_fail | [TL]·e_fail − 2[LL]·e₁e₂ |
  | Nff | [LL]·e₁e₂ | [LL]·e₁e₂ |
  | fake (=Nfp+Nff) | [TL]·e_fail + [LL]·e₁e₂ | [TL]·e_fail − [LL]·e₁e₂ |

- 출력: `MuMu/{TT,TL,LL}/...` (관측, MC 는 `/prompt`,`/fake` truth 분리 추가),
  `MuMu/matrix/{leading,full}/{Npp,Nfp,Nff,fake}/...` (count, mll, pt, ptcorr, eta, nJets, MET).
  **`MuMu/matrix/full/fake/count` 의 integral 이 fake background 추정치.**
- FR 입력 (userflag 로 선택):
  - 기본: `{era}_2/FR2D_QCD.root` (QCD MC FR) — MC closure test 용
  - `--userflags DataFR`: `{era}_data/FR2D_DATA.root` (데이터 FR) — 데이터 추정용

## 디렉토리 구조 (`/data6/Users/snuintern2/fake/`)

```
2016preVFP_2/ , 2016postVFP_2/     fake.cc 를 QCD MuEnriched MC 에 돌린 출력
    QCD_Pt-*_MuEnriched.root       (12개 pT bin 샘플)
    QCD.root                       hadd 결과
    FR2D_QCD.root                  fakerate_2d.py 출력 (bkg.cc 입력)
2016preVFP_data/ , 2016postVFP_data/   fake.cc 를 데이터/prompt MC 에 돌린 출력
    DoubleMuon.root                hadd (DoubleMuon_B..F 또는 F..H)
    DY.root                        hadd (DYJets + DYJets10to50)
    WJets.root
    FR2D_DATA.root                 fakerate_data.py 출력 (bkg.cc 입력, DataFR)
Tools/                             이 디렉토리 (후처리 스크립트)
    yes/33/, yes/2d/, yes/data/    플롯 출력
```

## Tools 스크립트 (실행 전 `rot` 로 ROOT 활성화)

| 스크립트 | 입력 | 출력 | 설명 |
|---|---|---|---|
| `fakerate_pre.py` / `fakerate_post.py` | `{era}/QCD.root` | 1D FR 플롯 | QCD MC FR, eta 구간별 1D (Mu8+Mu17 stitched) |
| `fakerate_2d.py` | `{era}/QCD.root` | `FR2D_QCD.root` + 2D 플롯 | QCD MC FR 2D stitching, bkg.cc 입력 생성 |
| `fakerate_data.py` | `{era}/DoubleMuon.root, DY.root, WJets.root` | `FR2D_DATA.root` + 1D/2D 플롯 | **데이터 FR**: ZEnriched 로 path 별 MC scale 추출 → prompt 성분 subtraction → FR |

```bash
rot && ERA=2016preVFP_2 python3 fakerate_2d.py      # QCD MC FR
rot && ERA=2016preVFP_data python3 fakerate_data.py  # 데이터 FR
```

### fakerate_data.py 동작 (prompt 제거 + normalization)

1. path 별 scale = N_data(Z peak) / N_MC(Z peak) — prescale 유효 lumi 반영.
   (preVFP: Mu8 4.5e-4, Mu17 9.2e-3 / postVFP: Mu8 7.6e-5, Mu17 1.4e-3)
2. MeasReg 에서 data − scale × (DY+WJets 의 **prompt 성분만**) 을 loose/tight 각각 빼기.
   (MC 의 fake 성분은 데이터 fake 에 이미 포함된 것이므로 빼지 않음)
3. FR = tight_sub / loose_sub, 30 GeV 에서 Mu8/Mu17 stitching.

Mu17 tight 의 50–100 GeV bin 은 prompt 오염이 데이터의 40–80% 라 subtraction 이 필수
(빼기 전 FR ~0.25–0.36 → 빼면 0.06–0.14, Fig 44 와 일치).

## 전체 workflow

```bash
# 1. FR 측정용 잡 (fake analyzer)
SKNano.py -a fake -i 'QCD_Pt-*MuEnriched*' -e 2016preVFP -n 20   # QCD MC FR 용
SKNano.py -a fake -i 'DoubleMuon*' -e 2016preVFP -n 50           # 데이터 FR 용
SKNano.py -a fake -i 'DYJets*' -e 2016preVFP -n 20               #  + prompt subtraction 용
SKNano.py -a fake -i 'WJets' -e 2016preVFP -n 20

# 2. hadd 후 FR 파일 생성
rot && ERA=2016preVFP_2 python3 fakerate_2d.py
rot && ERA=2016preVFP_data python3 fakerate_data.py

# 3-a. MC closure test (QCD FR): matrix/full/fake vs TT/fake(truth) 비교
SKNano.py -a bkg -i 'TTLL_powheg' -e 2016preVFP -n 20
SKNano.py -a bkg -i 'TTLJ_powheg' -e 2016preVFP -n 20    # 주된 fake source
SKNano.py -a bkg -i 'WJets' -e 2016preVFP -n 20
SKNano.py -a bkg -i 'DYJets*' -e 2016preVFP -n 20
SKNano.py -a bkg -i 'SingleTop_tW_*_NoFullyHad' -e 2016preVFP -n 10

# 3-b. 데이터 background 추정 (data FR)
SKNano.py -a bkg -i 'SingleMuon*' -e 2016preVFP -n 50 --userflags DataFR
```

## 주의사항 / 삽질 기록

- **Fig 44 vs Fig 46**: QCD MC 로 잰 FR 은 Fig 46 과 비교해야 한다. Fig 44(데이터)는
  50–100 GeV 에서 올라가는데 이는 prompt 오염/조성 차이 때문 — QCD MC FR 이
  단조감소하는 것은 정상.
- **Sumw2**: 데이터 히스토그램은 weight=1 로 채워져 Sumw2 가 없다.
  `Divide` 전에 `Sumw2()` 를 켜지 않으면 에러가 sqrt(ratio) 로 잘못 나온다.
- **Mu17 의 ptcorr < 30 bin 이 0 인 것**은 설계 (ptCorrCut=30); 최종 FR 은
  Mu8(10–30) + Mu17(30–100) stitching.
- **TkRelIso < 0.4**: 프레임워크의 HcToWA muon ID 는 이 컷이 사실상 no-op 인 버그가
  있어, fake.cc / bkg.cc 는 자체 `PassMuonWP()` 에서 올바르게 적용한다.
- **FR universality**: QCD 에서 잰 FR 을 tt-bar 의 b-decay fake 에 적용하는 가정이
  이 방법의 주요 systematic (AN-25-154 Fig 46 vs 47 참고). closure 가 어긋나면
  이것부터 의심할 것.
- matrix full calculation 에서 LL 이벤트가 음수 weight 로 들어가므로 bin 별로
  음수가 나올 수 있음 — 방법상 정상.
