#ifndef fake_h
#define fake_h

#include "AnalyzerCore.h"

//==============================================================
// Muon fake rate 측정 (Run 2 2016, prescaled single muon trigger)
//
// 방법: QCD-enriched 영역에서 loose muon 중 tight 를 통과하는 비율을 잰다
//   FR(ptcorr, |eta|) = N(tight) / N(loose)
//
// Userflags:
//   MeasFakeMu8  - Mu8 트리거 경로만 측정
//   MeasFakeMu17 - Mu17 트리거 경로만 측정
//   (플래그 없음) - 두 경로 모두 측정
//   RunSyst      - away jet pT 변화 (30 / 60 GeV) 시스테마틱도 채움
//==============================================================
class fake : public AnalyzerCore {
public:
    fake();
    ~fake();

    void initializeAnalyzer();
    void executeEvent();

    // rootcling 이 vector<fake::TriggerPath> dictionary 를 만들기 때문에 public 필요
    struct TriggerPath {
        TString name;      // 히스토그램 prefix: Mu8, Mu17
        TString trigger;   // HLT 경로 이름
        float ptCut;       // offline muon pT 컷
        float ptCorrCut;   // cone-corrected pT 영역 컷
    };

    // Userflags
    bool MeasFakeMu8, MeasFakeMu17, RunSyst;

    // 트리거 경로 / 시스테마틱 / binning
    vector<TriggerPath> paths;
    RVec<TString> systs;                 // Central (+ AwayJetPt30, AwayJetPt60 if RunSyst)
    RVec<float> ptCorrBins, absEtaBins;  // FR 2D 히스토그램 binning
    TString electronLooseID;             // electron veto 용 loose ID

    // Analysis cuts (analysis note 의 WP 표 + 측정영역 정의)
    struct AnalysisCuts {
        // muon 공통: POG medium ID + 아래 컷
        float muon_pt_min    = 10.0;
        float muon_eta_max   =  2.4;
        float muon_dz_max    =  0.1;   // cm
        float muon_tkiso_max =  0.4;   // rel. tracker iso R03 (trigger emulation)
        // tight WP
        float tight_sip3d_max   = 3.0;
        float tight_miniiso_max = 0.1;
        // loose WP (Run 2)
        float loose_sip3d_max   = 5.0;
        float loose_miniiso_max = 0.6;
        // electron veto
        float electron_pt_min  = 15.0;
        float electron_eta_max =  2.5;
        // jet
        float jet_pt_min   = 25.0;
        float jet_eta_max  =  2.4;   // 2016: 2.4, 그 외: 2.5 (initializeAnalyzer 에서 설정)
        float jet_lep_dr   =  0.4;   // lepton 과 겹치는 jet 제거
        float awayjet_pt   = 40.0;   // Central away jet pT 컷
        float awayjet_dr   =  0.7;   // dR(mu, away jet)
        // 측정 영역 (W/Z prompt 오염 억제)
        float met_max = 25.0;
        float mt_max  = 25.0;
        // Z-enriched 영역 (MC normalization 용)
        float z_mass    = 91.2;
        float z_window  = 15.0;
        float zjet_pt   = 40.0;
    } cuts;

    // 이벤트마다 executeEvent 에서 새로 채우는 physics objects
    RVec<Muon> looseMuons, tightMuons;   // tight ⊂ loose
    RVec<Electron> looseElectrons;
    RVec<Jet> jets;
    Particle METv;
    RVec<Gen> gens;                      // prompt / fake 구분용 (MC only)

    // Helper functions
    bool PassMuonWP(const Muon &mu, const TString &wp) const;
    void measureFakeRate(const TriggerPath &path, float weight);
    void fillZEnriched(const TriggerPath &path, const Event &ev, float weight);
    void fillMuonKinematics(const TString &prefix, const Muon &mu,
                            float ptCorr, float MT, int nJets, float weight);
    TString LeptonTypeToString(int leptonType) const;
};

#endif
