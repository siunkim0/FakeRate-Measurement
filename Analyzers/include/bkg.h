#ifndef bkg_h
#define bkg_h

#include "AnalyzerCore.h"
#include "TFile.h"
#include "TH2D.h"

//==============================================================
// tt-bar dilepton (mumu) fake lepton background 추정 (matrix method)
//
// reference: AN2010/261 "fakeable object" method, section 9 (Example of ttbar)
//
// 방법: OS dimuon tt-bar selection 을 loose muon 으로 잡은 뒤,
//   두 muon 의 tight 통과 여부로 이벤트를 TT / TL / LL 로 나누고
//   fake analyzer 에서 측정한 FR(ptcorr, |eta|) 로 weight 를 걸어
//   tight-tight 영역의 fake background 수를 추정한다
//
// prompt rate 는 없으므로 p = 1 로 고정 (note 의 "p=1" column)
//   e = f / (1 - f)
//   leading term : Npp = [TT],  Nfp = [TL] e_fail,        Nff = [LL] e1*e2
//   full calc.   : Npp = [TT] - [TL] e_fail + [LL] e1*e2
//                  Nfp = [TL] e_fail - 2 [LL] e1*e2
//                  Nff = [LL] e1*e2
//   fake bkg (tight 영역) = Nfp + Nff
//     leading: [TL] e_fail + [LL] e1*e2
//     full   : [TL] e_fail - [LL] e1*e2
//==============================================================
class bkg : public AnalyzerCore {
public:
    bkg();
    ~bkg();

    void initializeAnalyzer();
    void executeEvent();

    // FR 입력 선택
    //   기본             : FR2D_QCD.root  (Tools/fakerate_2d.py, QCD MC)
    //   --userflags DataFR: FR2D_DATA.root (Tools/fakerate_data.py, 데이터 - prompt MC)
    bool UseDataFR = false;
    TString fakeRateFile;
    TH2D *h_FR = nullptr;   // stitched FR: ptcorr < 30 은 Mu8, >= 30 은 Mu17

    // Trigger (era 별 unprescaled single muon trigger)
    RVec<TString> triggers;
    TString electronLooseID;   // electron veto 용 loose ID

    // Analysis cuts
    struct AnalysisCuts {
        // muon WP: fake.cc 와 완전히 동일해야 FR 적용이 유효하다
        float muon_pt_min    = 10.0;
        float muon_eta_max   =  2.4;
        float muon_dz_max    =  0.1;   // cm
        float muon_tkiso_max =  0.4;   // rel. tracker iso R03 (trigger emulation)
        float tight_sip3d_max   = 3.0;
        float tight_miniiso_max = 0.1;
        float loose_sip3d_max   = 5.0;
        float loose_miniiso_max = 0.6;
        // tt-bar OS dimuon selection
        float mu1_pt_min = 26.0;   // trigger-safe (2017 은 29 로 변경)
        float mu2_pt_min = 10.0;
        float mll_min    = 20.0;
        float z_mass     = 91.2;   // Z veto: |mll - 91.2| > 15
        float z_window   = 15.0;
        int   njet_min   = 2;
        float met_min    = 40.0;
        // electron veto
        float electron_pt_min  = 15.0;
        float electron_eta_max =  2.5;
        // jet
        float jet_pt_min  = 25.0;
        float jet_eta_max =  2.4;   // 2016: 2.4, 그 외: 2.5 (initializeAnalyzer 에서 설정)
        float jet_lep_dr  =  0.4;
    } cuts;

    // 이벤트마다 executeEvent 에서 새로 채우는 physics objects
    RVec<Muon> looseMuons;       // selection 은 loose 기준
    RVec<Electron> looseElectrons;
    RVec<Jet> jets;
    Particle METv;
    RVec<Gen> gens;              // prompt / fake 구분용 (MC only)

    // Helper functions
    bool PassMuonWP(const Muon &mu, const TString &wp) const;
    float GetPtCorr(const Muon &mu) const;
    float GetFakeRate(const Muon &mu) const;
    void fillDilepton(const TString &prefix, const Muon &mu1, const Muon &mu2,
                      float weight);
};

#endif
