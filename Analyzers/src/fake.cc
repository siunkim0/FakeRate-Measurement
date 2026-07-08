#include "fake.h"

//==== Constructor and Destructor
fake::fake() : MeasFakeMu8(false), MeasFakeMu17(false) {}
fake::~fake() {}

//==== Initialize variables
void fake::initializeAnalyzer() {

    //==== Userflags
    MeasFakeMu8  = HasFlag("MeasFakeMu8");   // Mu8 트리거 경로만 측정
    MeasFakeMu17 = HasFlag("MeasFakeMu17");  // Mu17 트리거 경로만 측정

    //==== 플래그가 없으면 두 경로 모두 측정
    if (!MeasFakeMu8 && !MeasFakeMu17) {
        MeasFakeMu8 = true;
        MeasFakeMu17 = true;
    }

    //==== prescaled single muon trigger 경로 정의 (analysis note Table 38)
    //==== {히스토그램 prefix, HLT 이름, muon pT 컷, cone-corrected pT 컷}
    if (MeasFakeMu8)  paths.push_back({"Mu8",  "HLT_Mu8_TrkIsoVVL",  10., 10.});
    if (MeasFakeMu17) paths.push_back({"Mu17", "HLT_Mu17_TrkIsoVVL", 20., 30.});

    //==== systematic 목록: Central = away jet pT > 40 GeV
    //==== away jet pT variation 은 2D 히스토그램 몇 개만 더 채우는 것이라
    //==== 부담이 없으므로 항상 켠다
    systs = {"Central", "AwayJetPt30", "AwayJetPt60"};

    //==== fake rate 2D 히스토그램 binning: cone-corrected pT x |eta|
    //==== 공식 측정처럼 [100,200] 을 따로 둔다 (FR 파일을 만들 때는 100 까지만 사용)
    ptCorrBins = {10., 12., 14., 17., 20., 30., 50., 100., 200.};
    absEtaBins = {0., 0.9, 1.6, 2.4};

    //==== era 별 jet |eta| 컷
    if (DataEra == "2016preVFP" || DataEra == "2016postVFP") {
        cuts.jet_eta_max = 2.4;
    } else {
        cuts.jet_eta_max = 2.5;
    }

    myCorr = new MyCorrection(DataEra, DataPeriod, IsDATA ? DataStream : MCSample, IsDATA);
}

//==============================================================
// Muon WP (analysis note 의 WP 표)
// 공통       : POG medium ID, |dz| < 0.1 cm, rel. tracker iso (R03) < 0.4
// tight      : |SIP3D| < 3, rel. mini-iso < 0.1
// loose(Run2): |SIP3D| < 5, rel. mini-iso < 0.6
//==============================================================
bool fake::PassMuonWP(const Muon &mu, const TString &wp) const {
    //==== 공통 컷
    if (!mu.isPOGMediumId()) return false;
    if (!(fabs(mu.dZ()) < cuts.muon_dz_max)) return false;
    if (!(mu.TkRelIso() < cuts.muon_tkiso_max)) return false;   // trigger emulation

    //==== tight / loose 별 컷
    if (wp == "tight") {
        if (!(fabs(mu.SIP3D()) < cuts.tight_sip3d_max)) return false;
        if (!(mu.MiniPFRelIso() < cuts.tight_miniiso_max)) return false;
    } else {
        if (!(fabs(mu.SIP3D()) < cuts.loose_sip3d_max)) return false;
        if (!(mu.MiniPFRelIso() < cuts.loose_miniiso_max)) return false;
    }
    return true;
}

//==============================================================
// Electron veto WP: 공식 HcToWALooseRun2 (Run 2 전용)
// framework Electron::Pass_HcToWALooseRun2() 는 raw MVA 컷 부호가
// 반전된 버그가 있어 직접 구현한다 (elecfake.cc 의 loose WP 와 동일)
//==============================================================
bool fake::PassElectronVeto(const Electron &el) const {
    if (!el.Pass_HcToWABaseline()) return false;
    if (!(el.SIP3D() < 8.)) return false;
    if (!(el.MiniPFRelIso() < 0.4)) return false;
    //==== raw MVANoIso 컷 (IB, OB, EC); wp90 통과도 인정 (tight ⊂ loose)
    float mvaCut = 0.85;
    if (el.etaRegion() == Electron::ETAREGION::IB)      mvaCut = 0.985;
    else if (el.etaRegion() == Electron::ETAREGION::OB) mvaCut = 0.96;
    if (!(el.isMVANoIsoWP90() || el.MvaNoIso() > mvaCut)) return false;
    return true;
}

//==============================================================
// jet 단위 veto map: 공식 framework 의 로직을 직접 구현
// (우리 fork 의 AnalyzerCore::PassVetoMap 은 muon dR < 0.2 이거나
//  loose PU ID 를 통과 못 한 jet 자체를 veto 해 버려 동작이 다르다)
// EM fraction > 0.9 또는 muon dR < 0.2 인 jet 은 검사 대상에서 제외
//==============================================================
bool fake::PassVetoMapJet(const Jet &jet, const RVec<Muon> &muons) const {
    if (jet.chEmEF() + jet.neEmEF() > 0.9) return true;
    for (const auto &mu : muons)
        if (jet.DeltaR(mu) < 0.2) return true;
    return !myCorr->IsJetVetoZone(jet.Eta(), jet.Phi(), "jetvetomap");
}

void fake::executeEvent() {

    //==== 이벤트와 raw physics object 읽기
    Event ev = GetEvent();
    RVec<Jet> rawJets = GetAllJets();

    //==== noise (MET) filter 통과 요구
    if (!PassNoiseFilter(rawJets, ev)) return;

    RVec<Muon> rawMuons = GetAllMuons();

    //==== jet veto map (공식 측정과 동일하게 event 단위로 적용)
    if (!PassVetoMap(rawJets, rawMuons, "jetvetomap")) return;

    RVec<Electron> rawElectrons = GetAllElectrons();
    sort(rawMuons.begin(), rawMuons.end(), PtComparing);

    //==== Step 1: loose / tight muon 선택 (tight 는 loose 의 부분집합)
    looseMuons.clear();
    tightMuons.clear();
    for (const auto &mu : rawMuons) {
        if (mu.Pt() < cuts.muon_pt_min || fabs(mu.Eta()) > cuts.muon_eta_max) continue;
        if (!PassMuonWP(mu, "loose")) continue;
        looseMuons.push_back(mu);
        if (PassMuonWP(mu, "tight")) tightMuons.push_back(mu);
    }

    //==== Step 2: veto 용 loose electron 선택 (framework loose ID 의
    //====         raw MVA 컷 반전 버그 때문에 PassElectronVeto 로 직접 선택)
    looseElectrons.clear();
    for (const auto &el : rawElectrons) {
        if (!(el.Pt() > cuts.electron_pt_min)) continue;
        if (!(fabs(el.Eta()) < cuts.electron_eta_max)) continue;
        if (PassElectronVeto(el)) looseElectrons.push_back(el);
    }

    //==== Step 3: jet 선택 (공식 측정과 같은 순서)
    //====         tight ID → loose lepton 과 dR < 0.4 로 겹치는 jet 제거
    //====         → jet 단위 veto map → loose PU ID (Run2)
    jets = SelectJets(rawJets, "tight", cuts.jet_pt_min, cuts.jet_eta_max);
    jets = JetsVetoLeptonInside(jets, looseElectrons, looseMuons, cuts.jet_lep_dr);
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

    //==== Step 6: 트리거 경로(Mu8 / Mu17)마다 측정
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
// selection: loose muon 딱 1개 + electron 0개
//            + away jet (pT > 40, dR > 0.7) + MET < 25 + MT < 25
// FR(ptcorr, |eta|) = tight 히스토그램 / loose 히스토그램 (offline 에서 계산)
//==============================================================
void fake::measureFakeRate(const TriggerPath &path, float weight) {

    //==== Step 1: single lepton 이벤트 요구 (loose muon 1개, electron 0개)
    if (looseMuons.size() != 1) return;
    if (looseElectrons.size() != 0) return;

    const Muon &mu = looseMuons.at(0);

    //==== muon RECO SF (공식 측정과 동일하게 측정 대상 muon 에 적용)
    if (!IsDATA) weight *= myCorr->GetMuonRECOSF(mu);

    //==== Step 2: 트리거별 muon pT 컷 (Mu8: 10, Mu17: 20)
    if (mu.Pt() < path.ptCut) return;

    //==== Step 3: cone-corrected pT = pT * (1 + max(0, miniIso - 0.1))
    //====         tight iso 기준(0.1)을 넘는 만큼 pT 를 보정해 준다
    const float ptCorr = mu.Pt() * (1. + max(0.f, mu.MiniPFRelIso() - cuts.tight_miniiso_max));
    if (ptCorr < path.ptCorrCut) return;   // Mu8: 10, Mu17: 30

    //==== tight WP 통과 여부, prompt/fake 구분 (MC only)
    const bool isTight = PassMuonWP(mu, "tight");
    const TString ltype = IsDATA ? "" : LeptonTypeToString(GetLeptonType(mu, gens));

    //==== transverse mass MT(mu, MET)
    const float MT = sqrt(2. * mu.Pt() * METv.Pt() * (1. - cos(mu.DeltaPhi(METv))));

    //==== loose 는 항상 채우고, tight 통과 시 tight 도 채운다 → FR = tight / loose
    RVec<TString> tags = {"loose"};
    if (isTight) tags.push_back("tight");

    //==== overflow 는 마지막 bin 에 넣는다
    const float xval = min(ptCorr, ptCorrBins.back() - 0.1f);
    const float yval = fabs(mu.Eta());

    //==== away jet pT 컷을 바꿔가며 반복 (Central = 40, syst = 30 / 60)
    for (const auto &syst : systs) {
        float awayJetPtCut = cuts.awayjet_pt;
        if (syst == "AwayJetPt30") awayJetPtCut = 30.;
        else if (syst == "AwayJetPt60") awayJetPtCut = 60.;

        //==== Step 4: away jet 요구 (pT > cut && dR(mu, jet) > 0.7)
        int nJets = 0;
        bool hasAwayJet = false;
        for (const auto &jet : jets) {
            if (jet.Pt() < awayJetPtCut) continue;
            nJets++;
            if (jet.DeltaR(mu) > cuts.awayjet_dr) hasAwayJet = true;
        }
        if (!hasAwayJet) continue;

        //==== Inclusive (validation) 영역: MET / MT 컷 없이 kinematics 채움
        //==== 공식(MeasFakeRateV4) QCD MC FR 은 이 영역에서 측정되므로
        //==== 비교용 2D FR 히스토그램도 같이 채운다
        if (syst == "Central") {
            for (const auto &tag : tags) {
                fillMuonKinematics(path.name + "/Inclusive/" + tag,
                                   mu, ptCorr, MT, nJets, weight);
                FillHist(path.name + "/Inclusive/" + tag + "/fake_ptcorr_abseta",
                         xval, yval, weight, ptCorrBins, absEtaBins);
                if (!IsDATA) {
                    fillMuonKinematics(path.name + "/Inclusive/" + tag + "/" + ltype,
                                       mu, ptCorr, MT, nJets, weight);
                    FillHist(path.name + "/Inclusive/" + tag + "/" + ltype + "/fake_ptcorr_abseta",
                             xval, yval, weight, ptCorrBins, absEtaBins);
                }
            }
        }

        //==== Step 5: 측정 영역 컷 (W/Z 의 prompt muon 오염 억제)
        if (!(METv.Pt() < cuts.met_max)) continue;
        if (!(MT < cuts.mt_max)) continue;

        //==== FR 계산용 2D 히스토그램 (cone-corrected pT x |eta|)
        const TString base = path.name + "/MeasReg/" + syst;
        for (const auto &tag : tags) {
            FillHist(base + "/" + tag + "/fake_ptcorr_abseta",
                     xval, yval, weight, ptCorrBins, absEtaBins);
            if (!IsDATA)
                FillHist(base + "/" + tag + "/" + ltype + "/fake_ptcorr_abseta",
                         xval, yval, weight, ptCorrBins, absEtaBins);
        }

        //==== 측정 영역 kinematics (Central only)
        if (syst == "Central") {
            for (const auto &tag : tags) {
                fillMuonKinematics(base + "/" + tag, mu, ptCorr, MT, nJets, weight);
                if (!IsDATA)
                    fillMuonKinematics(base + "/" + tag + "/" + ltype,
                                       mu, ptCorr, MT, nJets, weight);
            }
        }
    }
}

//==============================================================
// Z-enriched 영역: prescaled trigger 경로별 MC normalization 추출용
// selection: OS tight muon pair + |M(ll) - 91.2| < 15 + jet pT > 40 하나 이상
//==============================================================
void fake::fillZEnriched(const TriggerPath &path, const Event &ev, float weight) {

    //==== Step 1: loose 2개 == tight 2개 (딱 tight pair 만 있는 이벤트)
    if (!(looseMuons.size() == 2 && tightMuons.size() == 2)) return;
    if (!looseElectrons.empty()) return;

    const Muon &mu1 = tightMuons.at(0);
    const Muon &mu2 = tightMuons.at(1);

    //==== muon RECO SF (공식 측정과 동일하게 두 muon 모두에 적용)
    if (!IsDATA) weight *= myCorr->GetMuonRECOSF(tightMuons);

    //==== Step 2: opposite sign + pT 컷 (leading 은 트리거별 컷, subleading 은 10)
    if (mu1.Charge() + mu2.Charge() != 0) return;
    if (mu1.Pt() < path.ptCut || mu2.Pt() < cuts.muon_pt_min) return;

    //==== Step 3: jet pT > 40 하나 이상 (측정 영역과 같은 topology 맞추기)
    bool hasJet = false;
    for (const auto &jet : jets) {
        if (jet.Pt() > cuts.zjet_pt) { hasJet = true; break; }
    }
    if (!hasJet) return;

    //==== Step 4: Z mass window |M(ll) - 91.2| < 15
    const float mll = (mu1 + mu2).M();
    if (fabs(mll - cuts.z_mass) > cuts.z_window) return;

    //==== MC 는 두 muon 모두 prompt 인지에 따라 prompt / fake 로도 나눠 채운다
    RVec<TString> prefixes = {path.name + "/ZEnriched"};
    if (!IsDATA) {
        const bool isPrompt = (GetLeptonType(mu1, gens) > 0) && (GetLeptonType(mu2, gens) > 0);
        prefixes.push_back(prefixes[0] + (isPrompt ? "/prompt" : "/fake"));
    }

    for (const auto &prefix : prefixes) {
        FillHist(prefix + "/mll", mll, weight, 60, 76.2, 106.2);
        FillHist(prefix + "/mu1_pt", mu1.Pt(), weight, 200, 0., 200.);
        FillHist(prefix + "/mu2_pt", mu2.Pt(), weight, 200, 0., 200.);
        FillHist(prefix + "/mu1_eta", mu1.Eta(), weight, 48, -2.4, 2.4);
        FillHist(prefix + "/mu2_eta", mu2.Eta(), weight, 48, -2.4, 2.4);
        FillHist(prefix + "/nPV", ev.nPV(), weight, 70, 0., 70.);
    }
}

//==============================================================
// muon kinematics 히스토그램 묶음 (prefix 아래 7개)
//==============================================================
void fake::fillMuonKinematics(const TString &prefix, const Muon &mu,
                              float ptCorr, float MT, int nJets, float weight) {
    FillHist(prefix + "/pt", mu.Pt(), weight, 200, 0., 200.);
    FillHist(prefix + "/ptcorr", ptCorr, weight, 200, 0., 200.);
    FillHist(prefix + "/eta", mu.Eta(), weight, 48, -2.4, 2.4);
    FillHist(prefix + "/phi", mu.Phi(), weight, 64, -3.2, 3.2);
    FillHist(prefix + "/MET", METv.Pt(), weight, 100, 0., 100.);
    FillHist(prefix + "/MT", MT, weight, 100, 0., 100.);
    FillHist(prefix + "/nJets", nJets, weight, 10, 0., 10.);
}

//==============================================================
// GetLeptonType > 0 : prompt (EW/BSM prompt, tau daughter, internal conversion)
// GetLeptonType <= 0: hadron 기원 / external conversion / unmatched → fake
//==============================================================
TString fake::LeptonTypeToString(int leptonType) const {
    return (leptonType > 0) ? "prompt" : "fake";
}
