#include "elecfake.h"

//==== Constructor and Destructor
elecfake::elecfake() : MeasFakeEle8(false), MeasFakeEle12(false), MeasFakeEle23(false) {}
elecfake::~elecfake() {}

//==== Initialize variables
void elecfake::initializeAnalyzer() {

    //==== Userflags
    MeasFakeEle8  = HasFlag("MeasFakeEle8");    // Ele8 트리거 경로만 측정
    MeasFakeEle12 = HasFlag("MeasFakeEle12");   // Ele12 트리거 경로만 측정
    MeasFakeEle23 = HasFlag("MeasFakeEle23");   // Ele23 트리거 경로만 측정

    //==== 플래그가 없으면 세 경로 모두 측정
    if (!MeasFakeEle8 && !MeasFakeEle12 && !MeasFakeEle23) {
        MeasFakeEle8 = true;
        MeasFakeEle12 = true;
        MeasFakeEle23 = true;
    }

    //==== prescaled single electron trigger 경로 정의 (analysis note Table 38)
    //==== {히스토그램 prefix, HLT 이름, electron pT 컷, cone-corrected pT 컷}
    if (MeasFakeEle8)  paths.push_back({"Ele8",  "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30",  10., 10.});
    if (MeasFakeEle12) paths.push_back({"Ele12", "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30", 15., 20.});
    if (MeasFakeEle23) paths.push_back({"Ele23", "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30", 25., 35.});

    //==== systematic 목록: Central = away jet pT > 40 GeV
    //==== away jet pT variation 은 2D 히스토그램 몇 개만 더 채우는 것이라
    //==== 부담이 없으므로 항상 켠다
    systs = {"Central", "AwayJetPt30", "AwayJetPt60"};

    //==== fake rate 2D 히스토그램 binning: cone-corrected pT x |scEta|
    //==== pT 경계는 트리거 stitching 지점 (20, 35) 을 포함해야 한다
    //==== 공식 측정처럼 [100,200] 을 따로 둔다 (FR 파일을 만들 때는 100 까지만 사용)
    ptCorrBins = {15., 17., 20., 25., 35., 50., 100., 200.};
    absEtaBins = {0., 0.8, 1.479, 2.5};

    //==== loose WP 의 raw MVANoIso 컷: etaRegion (IB, OB, EC) 별,
    //==== Run 2 / Run 3 에서 MVA 가 retraining 되어 값이 다르다
    //==== 주의: Electron::Pass_HcToWALooseRun2() 는 이 컷의 부호가 반전된
    //====       버그가 있어 PassElectronWP 에서 직접 구현한다
    if (Run == 2) { looseMvaIB = 0.985; looseMvaOB = 0.96; looseMvaEC = 0.85; }
    else          { looseMvaIB = 0.8;   looseMvaOB = 0.5;  looseMvaEC = -0.8; }

    //==== era 별 jet |eta| 컷
    if (DataEra == "2016preVFP" || DataEra == "2016postVFP") {
        cuts.jet_eta_max = 2.4;
    } else {
        cuts.jet_eta_max = 2.5;
    }

    myCorr = new MyCorrection(DataEra, DataPeriod, IsDATA ? DataStream : MCSample, IsDATA);
}

//==============================================================
// Electron WP (electron ID 표)
// 공통 (baseline) : trigger emulation (CaloIdL_TrackIdL_IsoVL: sieie, dEtaIn,
//                   dPhiIn, H/E, ECAL/HCAL PF cluster iso (EA), tracker iso;
//                   GAP 영역 제외) + conversion veto + missing inner hits <= 1
//                   + |dz| < 0.1 cm
// tight           : MVANoIso POG wp90, |SIP3D| < 4, rel. mini-iso < 0.1
// loose           : raw MVANoIso > (IB, OB, EC) 컷 (또는 wp90 통과),
//                   |SIP3D| < 8, rel. mini-iso < 0.4
//==============================================================
bool elecfake::PassElectronWP(const Electron &el, const TString &wp) const {
    //==== 공통 컷
    if (!el.Pass_CaloIdL_TrackIdL_IsoVL()) return false;   // trigger emulation
    if (!el.ConvVeto()) return false;
    if (!(el.LostHits() <= cuts.electron_losthits_max)) return false;
    if (!(fabs(el.dZ()) < cuts.electron_dz_max)) return false;

    //==== tight / loose 별 컷
    if (wp == "tight") {
        if (!el.isMVANoIsoWP90()) return false;
        if (!(fabs(el.SIP3D()) < cuts.tight_sip3d_max)) return false;
        if (!(el.MiniPFRelIso() < cuts.tight_miniiso_max)) return false;
    } else {
        //==== tight ⊂ loose 를 보장하기 위해 wp90 통과도 loose MVA 컷으로 인정
        float mvaCut = looseMvaEC;
        if (el.etaRegion() == Electron::ETAREGION::IB)      mvaCut = looseMvaIB;
        else if (el.etaRegion() == Electron::ETAREGION::OB) mvaCut = looseMvaOB;
        if (!(el.isMVANoIsoWP90() || el.MvaNoIso() > mvaCut)) return false;
        if (!(fabs(el.SIP3D()) < cuts.loose_sip3d_max)) return false;
        if (!(el.MiniPFRelIso() < cuts.loose_miniiso_max)) return false;
    }
    return true;
}

//==============================================================
// Muon veto: fake.cc 의 loose muon WP 와 동일
// (POG medium ID, |dz| < 0.1, tkRelIso < 0.4, |SIP3D| < 5, mini-iso < 0.6)
//==============================================================
bool elecfake::PassMuonVeto(const Muon &mu) const {
    if (mu.Pt() < cuts.muon_pt_min || fabs(mu.Eta()) > cuts.muon_eta_max) return false;
    if (!mu.isPOGMediumId()) return false;
    if (!(fabs(mu.dZ()) < cuts.muon_dz_max)) return false;
    if (!(mu.TkRelIso() < cuts.muon_tkiso_max)) return false;
    if (!(fabs(mu.SIP3D()) < cuts.muon_sip3d_max)) return false;
    if (!(mu.MiniPFRelIso() < cuts.muon_miniiso_max)) return false;
    return true;
}

//==============================================================
// jet 단위 veto map: 공식 framework 의 로직을 직접 구현 (fake.cc 와 동일)
// EM fraction > 0.9 또는 muon dR < 0.2 인 jet 은 검사 대상에서 제외
//==============================================================
bool elecfake::PassVetoMapJet(const Jet &jet, const RVec<Muon> &muons) const {
    if (jet.chEmEF() + jet.neEmEF() > 0.9) return true;
    for (const auto &mu : muons)
        if (jet.DeltaR(mu) < 0.2) return true;
    return !myCorr->IsJetVetoZone(jet.Eta(), jet.Phi(), "jetvetomap");
}

void elecfake::executeEvent() {

    //==== 이벤트와 raw physics object 읽기
    Event ev = GetEvent();
    rawJets = GetAllJets();   // flavor 매칭용으로 lepton cleaning 이전 컬렉션을 보관

    //==== noise (MET) filter 통과 요구
    if (!PassNoiseFilter(rawJets, ev)) return;

    RVec<Muon> rawMuons = GetAllMuons();

    //==== jet veto map (공식 측정과 동일하게 event 단위로 적용; Run2 는 no-op)
    if (!PassVetoMap(rawJets, rawMuons, "jetvetomap")) return;

    RVec<Electron> rawElectrons = GetAllElectrons();
    sort(rawElectrons.begin(), rawElectrons.end(), PtComparing);

    //==== Step 1: veto / loose / tight electron 선택 (tight ⊂ loose ⊂ veto)
    //====         공식 측정처럼 veto 는 pT > 10, 측정(loose) 은 pT > 15;
    //====         object |eta| 컷은 공식(SelectElectrons)과 같이 일반 eta 를 쓰고
    //====         FR binning 만 supercluster eta 를 쓴다
    vetoElectrons.clear();
    looseElectrons.clear();
    tightElectrons.clear();
    for (const auto &el : rawElectrons) {
        if (fabs(el.Eta()) > cuts.electron_eta_max) continue;
        if (el.Pt() < cuts.electron_veto_pt_min) continue;
        if (!PassElectronWP(el, "loose")) continue;
        vetoElectrons.push_back(el);
        if (el.Pt() < cuts.electron_pt_min) continue;   // 측정 대상은 pT > 15
        looseElectrons.push_back(el);
        if (PassElectronWP(el, "tight")) tightElectrons.push_back(el);
    }

    //==== Step 2: muon veto 선택 (pT > 10, loose muon WP)
    vetoMuons.clear();
    for (const auto &mu : rawMuons) {
        if (PassMuonVeto(mu)) vetoMuons.push_back(mu);
    }

    //==== Step 3: jet 선택 (공식 측정과 같은 순서)
    //====         tight ID → veto lepton (pT>10 electron + muon) 과 dR<0.4 제거
    //====         → jet 단위 veto map → loose PU ID (Run2)
    jets = SelectJets(rawJets, "tight", cuts.jet_pt_min, cuts.jet_eta_max);
    jets = JetsVetoLeptonInside(jets, vetoElectrons, vetoMuons, cuts.jet_lep_dr);
    RVec<Jet> jetsNoPuId;   // PU ID SF 계산용 (PU ID 적용 전)
    if (Run == 2) {
        RVec<Jet> vetoMapped;
        for (const auto &jet : jets)
            if (PassVetoMapJet(jet, rawMuons)) vetoMapped.push_back(jet);
        jetsNoPuId = vetoMapped;
        jets = SelectJets(vetoMapped, "loosePuId", cuts.jet_pt_min, cuts.jet_eta_max);
    }
    sort(jets.begin(), jets.end(), PtComparing);

    //==== Step 4: PUPPI MET + Type-I correction
    METv = ApplyTypeICorrection(ev.GetMETVector(Event::MET_Type::PUPPI),
                                rawJets, rawElectrons, rawMuons);

    //==== Step 5: gen 정보 (MC 에서 prompt / fake 구분 + PU ID SF 용)
    gens = IsDATA ? RVec<Gen>() : GetAllGens();
    genJets = IsDATA ? RVec<GenJet>() : GetAllGenJets();

    //==== 트리거 경로 공통 event weight 보정 (MC only, 공식 측정과 동일한 세트)
    float evtSF = 1.;
    if (!IsDATA) {
        //==== top pT reweight (TT 샘플만 해당)
        if (MCSample.Contains("TTLL") || MCSample.Contains("TTLJ"))
            evtSF *= myCorr->GetTopPtReweight(gens);

        //==== pileup jet ID SF: PU ID 적용 전, away jet pT 컷(40)을 넘는 jet 대상
        if (Run == 2) {
            RVec<Jet> sfJets;
            for (const auto &jet : jetsNoPuId)
                if (jet.Pt() > cuts.awayjet_pt) sfJets.push_back(jet);
            unordered_map<int, int> matchedIdx =
                GenJetMatching(sfJets, genJets, fixedGridRhoFastjetAll, 0.4, 10.);
            evtSF *= myCorr->GetPileupJetIDSF(sfJets, matchedIdx, "loose");
        }
    }

    //==== Step 6: 트리거 경로(Ele8 / Ele12 / Ele23)마다 측정
    for (const auto &path : paths) {

        //==== 해당 트리거 통과 요구
        if (!ev.PassTrigger(path.trigger)) continue;

        //==== weight 계산 (MC only)
        float weight = 1.;
        if (!IsDATA) {
            //==== prescale 은 trigger lumi 에 반영되어 있지 않으므로 여기의
            //==== normalization 은 임시값; 최종 MC normalization 은
            //==== ZEnriched 영역에서 offline 으로 뽑는다
            weight = MCweight() * ev.GetTriggerLumi(path.trigger);
            weight *= GetL1PrefireWeight();
            weight *= myCorr->GetPUWeight(ev.nTrueInt());
            weight *= evtSF;   // top pT reweight + pileup jet ID SF
        }

        measureFakeRate(path, weight);    // fake rate 측정 영역
        fillZEnriched(path, ev, weight);  // MC normalization 용 Z 영역
    }
}

//==============================================================
// Fake rate 측정 영역
// selection: loose electron 딱 1개 + muon 0개
//            + away jet (pT > 40, dR > 0.7) + MET < 25 + MT < 25
// FR(ptcorr, |scEta|) = tight 히스토그램 / loose 히스토그램 (offline 에서 계산)
//==============================================================
void elecfake::measureFakeRate(const TriggerPath &path, float weight) {

    //==== Step 1: single lepton 이벤트 요구
    //====         측정 electron 1개 + soft(pT>10) 전자도 그 1개뿐 + muon 0개
    if (looseElectrons.size() != 1) return;
    if (vetoElectrons.size() != 1) return;
    if (vetoMuons.size() != 0) return;

    const Electron &el = looseElectrons.at(0);

    //==== electron RECO SF (공식 측정과 동일하게 측정 대상 전자에 적용)
    if (!IsDATA) weight *= myCorr->GetElectronRECOSF(looseElectrons);

    //==== Step 2: 트리거별 electron pT 컷 (Ele8: 10, Ele12: 15, Ele23: 25)
    if (el.Pt() < path.ptCut) return;

    //==== Step 3: cone-corrected pT = pT * (1 + max(0, miniIso - 0.1))
    //====         tight iso 기준(0.1)을 넘는 만큼 pT 를 보정해 준다
    const float ptCorr = el.Pt() * (1. + max(0.f, el.MiniPFRelIso() - cuts.tight_miniiso_max));
    if (ptCorr < path.ptCorrCut) return;   // Ele8: 10, Ele12: 20, Ele23: 35

    //==== tight WP 통과 여부, prompt/fake 구분 (MC only)
    const bool isTight = PassElectronWP(el, "tight");
    const TString ltype = IsDATA ? "" : LeptonTypeToString(GetLeptonType(el, gens));

    //==== transverse mass MT(el, MET)
    const float MT = sqrt(2. * el.Pt() * METv.Pt() * (1. - cos(el.DeltaPhi(METv))));

    //==== loose 는 항상 채우고, tight 통과 시 tight 도 채운다 → FR = tight / loose
    RVec<TString> tags = {"loose"};
    if (isTight) tags.push_back("tight");

    //==== overflow 는 마지막 bin 에 넣는다 (binning 은 |scEta|)
    const float xval = min(ptCorr, ptCorrBins.back() - 0.1f);
    const float yval = fabs(el.scEta());

    //==== away jet pT 컷을 바꿔가며 반복 (Central = 40, syst = 30 / 60)
    for (const auto &syst : systs) {
        float awayJetPtCut = cuts.awayjet_pt;
        if (syst == "AwayJetPt30") awayJetPtCut = 30.;
        else if (syst == "AwayJetPt60") awayJetPtCut = 60.;

        //==== Step 4: away jet 요구 (pT > cut && dR(el, jet) > 0.7)
        int nJets = 0;
        bool hasAwayJet = false;
        for (const auto &jet : jets) {
            if (jet.Pt() < awayJetPtCut) continue;
            nJets++;
            if (jet.DeltaR(el) > cuts.awayjet_dr) hasAwayJet = true;
        }
        if (!hasAwayJet) continue;

        //==== Inclusive (validation) 영역: MET / MT 컷 없이 kinematics + 2D FR
        //==== 공식(MeasFakeRateV4) QCD MC FR 은 이 영역에서 측정되므로
        //==== 비교용 2D FR 히스토그램도 같이 채운다
        if (syst == "Central") {
            for (const auto &tag : tags) {
                fillElectronKinematics(path.name + "/Inclusive/" + tag,
                                       el, ptCorr, MT, nJets, weight);
                FillHist(path.name + "/Inclusive/" + tag + "/fake_ptcorr_abseta",
                         xval, yval, weight, ptCorrBins, absEtaBins);
                if (!IsDATA) {
                    fillElectronKinematics(path.name + "/Inclusive/" + tag + "/" + ltype,
                                           el, ptCorr, MT, nJets, weight);
                    FillHist(path.name + "/Inclusive/" + tag + "/" + ltype + "/fake_ptcorr_abseta",
                             xval, yval, weight, ptCorrBins, absEtaBins);
                }
            }
        }

        //==== Step 5: 측정 영역 컷 (W/Z 의 prompt electron 오염 억제)
        if (!(METv.Pt() < cuts.met_max)) continue;
        if (!(MT < cuts.mt_max)) continue;

        //==== FR 계산용 2D 히스토그램 (cone-corrected pT x |scEta|)
        const TString base = path.name + "/MeasReg/" + syst;
        for (const auto &tag : tags) {
            FillHist(base + "/" + tag + "/fake_ptcorr_abseta",
                     xval, yval, weight, ptCorrBins, absEtaBins);
            if (!IsDATA)
                FillHist(base + "/" + tag + "/" + ltype + "/fake_ptcorr_abseta",
                         xval, yval, weight, ptCorrBins, absEtaBins);
        }

        //==== per-parton-flavor 2D 히스토그램 (flavor-weighted FR 용, MC only)
        //==== source jet 을 gen / reco 두 방식으로 매칭해 b/c/s/d/u/g 로 분류
        //==== FR_flavor = tight_flavor / loose_flavor (offline)
        if (!IsDATA && syst == "Central") {
            const TString gf = GenJetFlavor(el);
            const TString rf = RecoJetFlavor(el);
            for (const auto &tag : tags) {
                FillHist(base + "/" + tag + "/genflav/" + gf + "/fake_ptcorr_abseta",
                         xval, yval, weight, ptCorrBins, absEtaBins);
                FillHist(base + "/" + tag + "/recoflav/" + rf + "/fake_ptcorr_abseta",
                         xval, yval, weight, ptCorrBins, absEtaBins);
            }
        }

        //==== 측정 영역 kinematics (Central only)
        if (syst == "Central") {
            for (const auto &tag : tags) {
                fillElectronKinematics(base + "/" + tag, el, ptCorr, MT, nJets, weight);
                if (!IsDATA)
                    fillElectronKinematics(base + "/" + tag + "/" + ltype,
                                           el, ptCorr, MT, nJets, weight);
            }
        }
    }
}

//==============================================================
// Z-enriched 영역: prescaled trigger 경로별 MC normalization 추출용
// selection: OS tight electron pair + |M(ll) - 91.2| < 15 + jet pT > 40 하나 이상
//==============================================================
void elecfake::fillZEnriched(const TriggerPath &path, const Event &ev, float weight) {

    //==== Step 1: loose 2개 == tight 2개 (딱 tight pair 만 있는 이벤트)
    if (!(looseElectrons.size() == 2 && tightElectrons.size() == 2)) return;
    if (!vetoMuons.empty()) return;

    const Electron &el1 = tightElectrons.at(0);
    const Electron &el2 = tightElectrons.at(1);

    //==== electron RECO SF (공식 측정과 동일하게 두 전자 모두에 적용)
    if (!IsDATA) weight *= myCorr->GetElectronRECOSF(tightElectrons);

    //==== Step 2: opposite sign + pT 컷 (leading 은 트리거별 컷, subleading 은 10)
    if (el1.Charge() + el2.Charge() != 0) return;
    if (el1.Pt() < path.ptCut || el2.Pt() < cuts.electron_pt_min) return;

    //==== Step 3: jet pT > 40 하나 이상 (측정 영역과 같은 topology 맞추기)
    bool hasJet = false;
    for (const auto &jet : jets) {
        if (jet.Pt() > cuts.zjet_pt) { hasJet = true; break; }
    }
    if (!hasJet) return;

    //==== Step 4: Z mass window |M(ll) - 91.2| < 15
    const float mll = (el1 + el2).M();
    if (fabs(mll - cuts.z_mass) > cuts.z_window) return;

    //==== MC 는 두 electron 모두 prompt 인지에 따라 prompt / fake 로도 나눠 채운다
    RVec<TString> prefixes = {path.name + "/ZEnriched"};
    if (!IsDATA) {
        const bool isPrompt = (GetLeptonType(el1, gens) > 0) && (GetLeptonType(el2, gens) > 0);
        prefixes.push_back(prefixes[0] + (isPrompt ? "/prompt" : "/fake"));
    }

    for (const auto &prefix : prefixes) {
        FillHist(prefix + "/mll", mll, weight, 60, 76.2, 106.2);
        FillHist(prefix + "/el1_pt", el1.Pt(), weight, 200, 0., 200.);
        FillHist(prefix + "/el2_pt", el2.Pt(), weight, 200, 0., 200.);
        FillHist(prefix + "/el1_eta", el1.scEta(), weight, 50, -2.5, 2.5);
        FillHist(prefix + "/el2_eta", el2.scEta(), weight, 50, -2.5, 2.5);
        FillHist(prefix + "/nPV", ev.nPV(), weight, 70, 0., 70.);
    }
}

//==============================================================
// electron kinematics 히스토그램 묶음 (prefix 아래 7개)
//==============================================================
void elecfake::fillElectronKinematics(const TString &prefix, const Electron &el,
                                      float ptCorr, float MT, int nJets, float weight) {
    FillHist(prefix + "/pt", el.Pt(), weight, 200, 0., 200.);
    FillHist(prefix + "/ptcorr", ptCorr, weight, 200, 0., 200.);
    FillHist(prefix + "/eta", el.scEta(), weight, 50, -2.5, 2.5);
    FillHist(prefix + "/phi", el.Phi(), weight, 64, -3.2, 3.2);
    FillHist(prefix + "/MET", METv.Pt(), weight, 100, 0., 100.);
    FillHist(prefix + "/MT", MT, weight, 100, 0., 100.);
    FillHist(prefix + "/nJets", nJets, weight, 10, 0., 10.);
}

//==============================================================
// GetLeptonType > 0 : prompt (EW/BSM prompt, tau daughter, internal conversion)
// GetLeptonType <= 0: hadron 기원 / external conversion / unmatched → fake
//==============================================================
TString elecfake::LeptonTypeToString(int leptonType) const {
    return (leptonType > 0) ? "prompt" : "fake";
}

//==============================================================
// Source jet parton flavor 분류 (fake.cc 와 동일 로직)
//   hadronFlavour (ghost B/C hadron matching) 로 b/c 를 먼저 잡고,
//   light (hadronFlavour==0) 는 |partonFlavour| 로 g/s/u/d 세분화한다.
//   partonFlavour: 21=gluon, 1=d, 2=u, 3=s (부호 = quark/antiquark)
//   매칭 실패(partonFlavour==0 등)는 unmatched
//==============================================================
TString elecfake::FlavorTag(int partonFlavour, int hadronFlavour) const {
    if (hadronFlavour == 5) return "b";
    if (hadronFlavour == 4) return "c";
    const int ap = abs(partonFlavour);
    if (ap == 21) return "g";
    if (ap == 3)  return "s";
    if (ap == 2)  return "u";
    if (ap == 1)  return "d";
    return "unmatched";
}

//==== electron 을 가장 가까운 reco jet (lepton cleaning 이전 rawJets) 에 dR<0.4 매칭
TString elecfake::RecoJetFlavor(const Electron &el) const {
    float bestDR = 0.4f;
    int bestParton = 0, bestHadron = 0;
    bool matched = false;
    for (const auto &jet : rawJets) {
        const float dr = jet.DeltaR(el);
        if (dr < bestDR) {
            bestDR = dr;
            bestParton = jet.partonFlavour();
            bestHadron = jet.hadronFlavour();
            matched = true;
        }
    }
    return matched ? FlavorTag(bestParton, bestHadron) : "unmatched";
}

//==== electron 을 가장 가까운 gen jet 에 dR<0.4 매칭 (source jet 의 진짜 flavor)
TString elecfake::GenJetFlavor(const Electron &el) const {
    float bestDR = 0.4f;
    int bestParton = 0, bestHadron = 0;
    bool matched = false;
    for (const auto &gjet : genJets) {
        const float dr = gjet.DeltaR(el);
        if (dr < bestDR) {
            bestDR = dr;
            bestParton = gjet.partonFlavour();
            bestHadron = gjet.hadronFlavour();
            matched = true;
        }
    }
    return matched ? FlavorTag(bestParton, bestHadron) : "unmatched";
}
