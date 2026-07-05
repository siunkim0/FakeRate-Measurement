#include "bkg.h"

//==== Constructor and Destructor
bkg::bkg() {}
bkg::~bkg() {
    if (h_FR) delete h_FR;
}

//==== Initialize variables
void bkg::initializeAnalyzer() {

    //==== era 별 unprescaled single muon trigger + trigger-safe pT 컷
    if (DataEra == "2016preVFP" || DataEra == "2016postVFP") {
        triggers = {"HLT_IsoMu24", "HLT_IsoTkMu24"};
    } else if (DataEra == "2017") {
        triggers = {"HLT_IsoMu27"};
        cuts.mu1_pt_min = 29.;
    } else {
        triggers = {"HLT_IsoMu24"};
    }

    //==== era 별 jet |eta| 컷
    if (DataEra == "2016preVFP" || DataEra == "2016postVFP") {
        cuts.jet_eta_max = 2.4;
    } else {
        cuts.jet_eta_max = 2.5;
    }

    //==== electron veto 용 loose ID
    electronLooseID = (Run == 2) ? "HcToWALooseRun2" : "HcToWALooseRun3";

    //==== 측정된 FR(ptcorr, |eta|) 읽기 (stitched: ptcorr < 30: Mu8, >= 30: Mu17)
    //====   기본            : QCD MC FR    (Tools/fakerate_2d.py 출력) - MC closure 용
    //====   --userflags DataFR: 데이터 FR (Tools/fakerate_data.py 출력, prompt 제거)
    //====                       - 데이터에서 background 추정할 때 사용
    UseDataFR = HasFlag("DataFR");
    fakeRateFile = UseDataFR
        ? "/data6/Users/snuintern2/fake/" + DataEra + "_data/FR2D_DATA.root"
        : "/data6/Users/snuintern2/fake/" + DataEra + "_2/FR2D_QCD.root";
    TFile *f_FR = TFile::Open(fakeRateFile);
    if (!f_FR || f_FR->IsZombie()) {
        cerr << "[bkg::initializeAnalyzer] cannot open " << fakeRateFile << endl;
        cerr << "[bkg::initializeAnalyzer] run Tools/fakerate_2d.py (QCD) or "
             << "Tools/fakerate_data.py (data) first" << endl;
        exit(EXIT_FAILURE);
    }

    h_FR = (TH2D *)f_FR->Get("FR");
    if (!h_FR) {
        cerr << "[bkg::initializeAnalyzer] cannot find 'FR' in " << fakeRateFile << endl;
        exit(EXIT_FAILURE);
    }
    h_FR->SetDirectory(nullptr);   // 파일을 닫아도 히스토그램이 살아있도록 분리
    f_FR->Close();

    myCorr = new MyCorrection(DataEra, DataPeriod, IsDATA ? DataStream : MCSample, IsDATA);
}

//==============================================================
// Muon WP: fake.cc 와 동일 (analysis note 의 WP 표)
// 공통       : POG medium ID, |dz| < 0.1 cm, rel. tracker iso (R03) < 0.4
// tight      : |SIP3D| < 3, rel. mini-iso < 0.1
// loose(Run2): |SIP3D| < 5, rel. mini-iso < 0.6
//==============================================================
bool bkg::PassMuonWP(const Muon &mu, const TString &wp) const {
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
// cone-corrected pT = pT * (1 + max(0, miniIso - 0.1)) : fake.cc 와 동일
//==============================================================
float bkg::GetPtCorr(const Muon &mu) const {
    return mu.Pt() * (1. + max(0.f, mu.MiniPFRelIso() - cuts.tight_miniiso_max));
}

//==============================================================
// FR lookup: (ptcorr, |eta|) 로 h_FR 에서 읽는다
// binning 범위를 벗어나면 마지막 bin 값을 쓴다 (overflow 방지)
//==============================================================
float bkg::GetFakeRate(const Muon &mu) const {
    const TAxis *ax = h_FR->GetXaxis();
    const TAxis *ay = h_FR->GetYaxis();
    const float x = min(max(GetPtCorr(mu), float(ax->GetXmin() + 0.01)),
                        float(ax->GetXmax() - 0.01));
    const float y = min(float(fabs(mu.Eta())), float(ay->GetXmax() - 0.01));

    float f = h_FR->GetBinContent(h_FR->FindBin(x, y));
    //==== 병적인 bin 보호: e = f/(1-f) 가 발산하지 않도록 clamp
    if (f < 0.001) f = 0.001;
    if (f > 0.950) f = 0.950;
    return f;
}

void bkg::executeEvent() {

    //==== 이벤트와 raw physics object 읽기
    Event ev = GetEvent();
    RVec<Jet> rawJets = GetAllJets();

    //==== noise (MET) filter 통과 요구
    if (!PassNoiseFilter(rawJets, ev)) return;

    //==== 트리거 통과 요구 (era 별 IsoMu 경로 중 하나)
    bool passTrigger = false;
    for (const auto &trig : triggers) {
        if (ev.PassTrigger(trig)) { passTrigger = true; break; }
    }
    if (!passTrigger) return;

    RVec<Muon> rawMuons = GetAllMuons();
    RVec<Electron> rawElectrons = GetAllElectrons();
    sort(rawMuons.begin(), rawMuons.end(), PtComparing);

    //==== Step 1: loose muon 선택 (selection 은 loose 기준으로 잡는다)
    looseMuons.clear();
    for (const auto &mu : rawMuons) {
        if (mu.Pt() < cuts.muon_pt_min || fabs(mu.Eta()) > cuts.muon_eta_max) continue;
        if (!PassMuonWP(mu, "loose")) continue;
        looseMuons.push_back(mu);
    }

    //==== Step 2: loose electron 선택 (mumu channel 을 위한 veto 용)
    looseElectrons = SelectElectrons(rawElectrons, electronLooseID,
                                     cuts.electron_pt_min, cuts.electron_eta_max);

    //==== Step 3: jet 선택 - tight ID (+ Run2 는 loose PU ID),
    //====         loose lepton 과 dR < 0.4 로 겹치는 jet 은 제거
    jets = SelectJets(rawJets, "tight", cuts.jet_pt_min, cuts.jet_eta_max);
    if (Run == 2) jets = SelectJets(jets, "loosePuId", cuts.jet_pt_min, cuts.jet_eta_max);
    jets = JetsVetoLeptonInside(jets, looseElectrons, looseMuons, cuts.jet_lep_dr);
    sort(jets.begin(), jets.end(), PtComparing);

    //==== Step 4: PUPPI MET + Type-I correction
    METv = ApplyTypeICorrection(ev.GetMETVector(Event::MET_Type::PUPPI),
                                rawJets, rawElectrons, rawMuons);

    //==== Step 5: gen 정보 (MC 에서 prompt / fake 구분용)
    gens = IsDATA ? RVec<Gen>() : GetAllGens();

    //==== Step 6: tt-bar OS dimuon selection (loose 기준)
    //====   loose muon 딱 2개 + electron 0개 + OS
    //====   leading pT > 26 (trigger-safe), subleading pT > 10
    //====   mll > 20 + Z veto + jet >= 2 + MET > 40
    if (looseMuons.size() != 2) return;
    if (!looseElectrons.empty()) return;

    const Muon &mu1 = looseMuons.at(0);
    const Muon &mu2 = looseMuons.at(1);

    if (mu1.Charge() + mu2.Charge() != 0) return;
    if (mu1.Pt() < cuts.mu1_pt_min || mu2.Pt() < cuts.mu2_pt_min) return;

    const float mll = (mu1 + mu2).M();
    if (mll < cuts.mll_min) return;
    if (fabs(mll - cuts.z_mass) < cuts.z_window) return;   // Z veto

    if ((int)jets.size() < cuts.njet_min) return;
    if (METv.Pt() < cuts.met_min) return;

    //==== Step 7: weight 계산 (MC only)
    float weight = 1.;
    if (!IsDATA) {
        weight = MCweight() * ev.GetTriggerLumi("Full");
        weight *= GetL1PrefireWeight();
        weight *= myCorr->GetPUWeight(ev.nTrueInt());
    }

    //==== Step 8: tight 통과 여부로 이벤트를 TT / TL / LL 로 분류
    //====   T = tight 통과, L = loose 만 통과 (fail)
    const bool tight1 = PassMuonWP(mu1, "tight");
    const bool tight2 = PassMuonWP(mu2, "tight");
    const int nTight = int(tight1) + int(tight2);
    const TString cat = (nTight == 2) ? "TT" : (nTight == 1) ? "TL" : "LL";

    //==== 관측된 category 별 히스토그램 (Nt2 / Nt1 / Nt0)
    fillDilepton("MuMu/" + cat, mu1, mu2, weight);

    //==== MC 는 truth 로도 나눠 채운다 (closure test 용):
    //====   두 muon 모두 prompt 면 prompt, 하나라도 아니면 fake
    if (!IsDATA) {
        const bool bothPrompt = (GetLeptonType(mu1, gens) > 0) && (GetLeptonType(mu2, gens) > 0);
        fillDilepton("MuMu/" + cat + (bothPrompt ? "/prompt" : "/fake"), mu1, mu2, weight);
    }

    //==== Step 9: matrix method weight (p = 1 고정)
    //====   e_i = f_i / (1 - f_i), f_i = FR(ptcorr_i, |eta_i|)
    const float f1 = GetFakeRate(mu1);
    const float f2 = GetFakeRate(mu2);
    const float e1 = f1 / (1. - f1);
    const float e2 = f2 / (1. - f2);

    //==== category 별로 각 추정치에 들어갈 weight 목록 (AN2010/261 section 9)
    //====   leading term : Npp = [TT], Nfp = [TL] e_fail, Nff = [LL] e1*e2
    //====   full calc.   : Npp = [TT] - [TL] e_fail + [LL] e1*e2
    //====                  Nfp = [TL] e_fail - 2 [LL] e1*e2
    //====                  Nff = [LL] e1*e2
    //====   fake = Nfp + Nff (tight-tight 영역에 남는 fake background)
    vector<pair<TString, float>> estimates;
    if (cat == "TT") {
        estimates = {{"leading/Npp", 1.},
                     {"full/Npp", 1.}};
    } else if (cat == "TL") {
        const float eFail = tight1 ? e2 : e1;   // fail 한 muon 의 e
        estimates = {{"leading/Nfp", eFail}, {"leading/fake", eFail},
                     {"full/Npp", -eFail},
                     {"full/Nfp", eFail},
                     {"full/fake", eFail}};
    } else {   // LL
        const float ee = e1 * e2;
        estimates = {{"leading/Nff", ee}, {"leading/fake", ee},
                     {"full/Npp", ee},
                     {"full/Nfp", -2.f * ee},
                     {"full/Nff", ee},
                     {"full/fake", -ee}};
    }

    for (const auto &[name, matrixWeight] : estimates) {
        fillDilepton("MuMu/matrix/" + name, mu1, mu2, weight * matrixWeight);
    }
}

//==============================================================
// dimuon kinematics 히스토그램 묶음 (prefix 아래 9개)
// count 의 integral 이 그 category / 추정치의 event yield
//==============================================================
void bkg::fillDilepton(const TString &prefix, const Muon &mu1, const Muon &mu2,
                       float weight) {
    FillHist(prefix + "/count", 0.5, weight, 1, 0., 1.);
    FillHist(prefix + "/mll", (mu1 + mu2).M(), weight, 80, 0., 400.);
    FillHist(prefix + "/mu1_pt", mu1.Pt(), weight, 200, 0., 200.);
    FillHist(prefix + "/mu2_pt", mu2.Pt(), weight, 200, 0., 200.);
    FillHist(prefix + "/mu1_ptcorr", GetPtCorr(mu1), weight, 200, 0., 200.);
    FillHist(prefix + "/mu2_ptcorr", GetPtCorr(mu2), weight, 200, 0., 200.);
    FillHist(prefix + "/mu1_eta", mu1.Eta(), weight, 48, -2.4, 2.4);
    FillHist(prefix + "/mu2_eta", mu2.Eta(), weight, 48, -2.4, 2.4);
    FillHist(prefix + "/nJets", jets.size(), weight, 10, 0., 10.);
    FillHist(prefix + "/MET", METv.Pt(), weight, 100, 0., 200.);
}
