// Open a NanoAOD file and extract Gen-level condtioning AND reco targets for
// trainings Working fine with ROOT 6.22


auto DeltaPhiMEFJ(ROOT::VecOps::RVec<float> &PhiJ) {

  /* Calculates the DeltaPhi between most energetic FakeJet and
  all others FakeJets
   */

  auto size = PhiJ.size();
  ROOT::VecOps::RVec<float> dphis;
  dphis.reserve(size);
  Double_t dphi0 = -0.1; 
  dphis.emplace_back(dphi0);
  for (size_t i = 1; i < size; i++) {
    Double_t dphi = TVector2::Phi_mpi_pi(PhiJ[0] - PhiJ[i]);
    dphis.emplace_back(dphi);
  }
  return dphis;
}

auto ABSDeltaEtaMEFJ(ROOT::VecOps::RVec<float> &EtaJ) {
  /* Calculates the absolute value of DeltaEta between most energetic FakeJet and
  all others FakeJets
  */
  auto size = EtaJ.size();
  ROOT::VecOps::RVec<float> detas;
  detas.reserve(size);
  Double_t deta0 = -0.1; 
  detas.emplace_back(deta0);
  for (size_t i = 1; i < size; i++) {
    Double_t deta = TMath::Abs(EtaJ[0] - EtaJ[i]);
    detas.emplace_back(deta);
  }
  return detas;
}
  

auto DeltaPhi(ROOT::VecOps::RVec<float> &Phi1,
              ROOT::VecOps::RVec<float> &Phi2) {

  /* Calculates the DeltaPhi between two RVecs
   */

  auto size = Phi1.size();
  ROOT::VecOps::RVec<float> dphis;
  dphis.reserve(size);
  for (size_t i = 0; i < size; i++) {
    Double_t dphi = TVector2::Phi_mpi_pi(Phi1[i] - Phi2[i]);
    dphis.emplace_back(dphi);
  }
  return dphis;
}

auto closest_muon_dr(ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etam,
                     ROOT::VecOps::RVec<float> &phim) {

  /* Calculates the DeltaR from the closest muon object,
          if none present within 0.4, sets DR to 0.4
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> distances;
  distances.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    distances.emplace_back(0.5);
    float closest = 0.4;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
      }
    }
    if (closest < 0.4) {
      distances[i] = closest;
    }
  }
  return distances;
}

auto closest_muon_pt(ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etam,
                     ROOT::VecOps::RVec<float> &phim,
                     ROOT::VecOps::RVec<float> &ptm) {

  /* Calculates the pt of the closest muon object,
          if none present within 0.4, sets DR to 0.4 and pt to 0 GeV
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> pts;
  pts.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    pts.emplace_back(0.0);
    float closest = 0.4;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        pts[i] = ptm[j];
      }
    }
  }
  return pts;
}

auto closest_muon_deta(ROOT::VecOps::RVec<float> &etaj,
                       ROOT::VecOps::RVec<float> &phij,
                       ROOT::VecOps::RVec<float> &etam,
                       ROOT::VecOps::RVec<float> &phim) {

  /* Calculates the DeltaEta of the closest muon object,
          if none present within 0.4, sets DR to 0.4 and DeltaEta to 0.5
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> detas;
  detas.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    detas.emplace_back(0.5);
    float closest = 0.4;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        detas[i] = deta;
      }
    }
  }
  return detas;
}

auto closest_muon_dphi(ROOT::VecOps::RVec<float> &etaj,
                       ROOT::VecOps::RVec<float> &phij,
                       ROOT::VecOps::RVec<float> &etam,
                       ROOT::VecOps::RVec<float> &phim) {

  /* Calculates the DeltaPhi of the closest muon object,
          if none present within 0.4, sets DR to 0.4 and DeltaPhi to 0.5
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> dphis;
  dphis.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    dphis.emplace_back(0.5);
    float closest = 0.4;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        dphis[i] = dphi;
      }
    }
  }
  return dphis;
}

auto second_muon_dr(ROOT::VecOps::RVec<float> &etaj,
                    ROOT::VecOps::RVec<float> &phij,
                    ROOT::VecOps::RVec<float> &etam,
                    ROOT::VecOps::RVec<float> &phim) {

  /* Calculates the DeltaR from the second closest muon object,
          if none present within 0.4, sets DR to 0.4
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> distances;
  distances.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    distances.emplace_back(0.5);
    float closest = 0.4;
    float second_closest = 0.5;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        second_closest = closest;
        closest = dr;
      }
    }
    if (second_closest < 0.4) {
      distances[i] = second_closest;
    }
  }
  return distances;
}

auto second_muon_pt(ROOT::VecOps::RVec<float> &etaj,
                    ROOT::VecOps::RVec<float> &phij,
                    ROOT::VecOps::RVec<float> &etam,
                    ROOT::VecOps::RVec<float> &phim,
                    ROOT::VecOps::RVec<float> &ptm) {

  /* Calculates the pt of the second closest muon object,
          if none present within 0.4, sets DR to 0.4 and pt to 0 GeV
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> pts;
  pts.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    pts.emplace_back(0.0);
    float closest = 0.4;
    float second_closest = 0.5;
    float closest_pt = 0.0;
    float second_pt = 0.0;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        second_closest = closest;
        second_pt = closest_pt;
        closest = dr;
        closest_pt = ptm[j];
      }
      if (second_closest < 0.4) {
        pts[i] = second_pt;
      }
    }
  }
  return pts;
}

auto second_muon_deta(ROOT::VecOps::RVec<float> &etaj,
                      ROOT::VecOps::RVec<float> &phij,
                      ROOT::VecOps::RVec<float> &etam,
                      ROOT::VecOps::RVec<float> &phim) {

  /* Calculates the DeltaEta of the second closest muon object,
          if none present within 0.4, sets DR to 0.4 and DeltaEta to 0.5
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> detas;
  detas.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    detas.emplace_back(0.5);
    float closest = 0.4;
    float second_closest = 0.5;
    float closest_deta = 0.0;
    float second_deta = 0.0;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        second_closest = closest;
        second_deta = closest_deta;
        closest = dr;
        closest_deta = deta;
      }
      if (second_closest < 0.4) {
        detas[i] = second_deta;
      }
    }
  }
  return detas;
}

auto second_muon_dphi(ROOT::VecOps::RVec<float> &etaj,
                      ROOT::VecOps::RVec<float> &phij,
                      ROOT::VecOps::RVec<float> &etam,
                      ROOT::VecOps::RVec<float> &phim) {

  /* Calculates the DeltaPhi of the second closest muon object,
          if none present within 0.4, sets DR to 0.4 and DeltaPhi to 0.5
  */

  auto size_outer = etaj.size();
  auto size_inner = etam.size();
  ROOT::VecOps::RVec<float> dphis;
  dphis.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    dphis.emplace_back(0.5);
    float closest = 0.4;
    float second_closest = 0.5;
    float closest_dphi = 0.0;
    float second_dphi = 0.0;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etaj[i] - etam[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phij[i] - phim[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        second_closest = closest;
        second_dphi = closest_dphi;
        closest = dr;
        closest_dphi = dphi;
      }
      if (second_closest < 0.4) {
        dphis[i] = second_dphi;
      }
    }
  }
  return dphis;
}

void gen_jet_graph(int argc, char** argv) {

  /* The main function. Uses ROOT::RDataFrame to select only jets NOT matching
     to a GenJet, then extracts all the conditioning variables of the event and
     the target variables of the fake jet in a single row (each row contains all
     the conditioning and target variables for exactly one jet)
  */

  
  // enable multithreading, open file and init rdataframe
  // BECAUSE OF MT ORIGINAL ORDERING OF FILE IS NOT PRESERVED
  ROOT::EnableImplicitMT(); // disabling if you want to use range
  TFile *f =
      TFile::Open(argv[0]);
                  /*"root://cmsxrootd.fnal.gov///store/mc/RunIIAutumn18NanoAODv6/"
                  "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/"
                  "Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/250000/"
                  "047F4368-97D4-1A4E-B896-23C6C72DD2BE.root");*/
  ROOT::RDataFrame d("Events", f);

  // create first mask
  auto d_def =
      d.Define("JetMask", "Jet_genJetIdx < 0 || Jet_genJetIdx >= nGenJet");
  // Optionally print columns names
  // auto v2 = d_matched.GetColumnNames();
  // for (auto &&colName : v2) std::cout <<"\""<< colName<<"\", ";

  // The main rdataframe, LAZILY defining all the new processed columns.
  // Notice that muon conditioning variables are defined starting from GenMuons
  // Lines commented out are variables missing in some NanoAODs
  auto d_matched = d_def.Define("FJet_phi", "Jet_phi[JetMask]")
                       .Define("FJet_eta", "Jet_eta[JetMask]")
                       .Define("FJet_pt", "Jet_pt[JetMask]")
                       .Define("FJet_dphi", DeltaPhiMEFJ, {"FJet_phi"})
                       .Define("FJet_deta", ABSDeltaEtaMEFJ, {"FJet_eta"});

  // Define variables to save
  vector<string> col_to_save = {"nGenJet",
                                "nGenPart",
                                "GenJet_eta",
                                "GenJet_mass",
                                "GenJet_phi",
                                "GenJet_pt",
                                "GenJet_partonFlavour",
                                "GenJet_hadronFlavour",
                                "GenPart_eta",
                                "GenPart_genPartIdxMother",
                                "GenPart_mass",
                                "GenPart_pdgId",
                                "GenPart_phi",
                                "GenPart_pt",
                                "GenPart_status",
                                "GenPart_statusFlags",
                                "Pileup_gpudensity",
                                "Pileup_nPU",
                                "Pileup_nTrueInt",
                                "Pileup_pudensity",
                                "Pileup_sumEOOT",
                                "Pileup_sumLOOT",
                                "event",
                                "FJet_phi",
                                "FJet_eta",
                                "FJet_pt",
                                "FJet_dphi",
                                "FJet_deta"};

  // finally process columns and save to .root file
  d_matched.Snapshot("FJets", argv[1], col_to_save);
  /*
  gStyle->SetOptStat(0);
  auto h = d_matched.Histo1D({"FJet_dphi", "FJet_dphi", 100, 0, 3.14},
                             "FJet_dphi");
  auto c = new TCanvas();
  h->Draw();
  c->SaveAs("FJet_dphi.png");
  TFile *ffig = new TFile("FJet_dphi.root", "RECREATE");
  h->Write();
  ffig->Close();

  auto h2 = d_matched.Histo1D({"FJet_deta", "FJet_deta", 100, 0, 5},
                              "FJet_deta");
  auto c2 = new TCanvas();
  h2->Draw();
  c2->SaveAs("FJet_deta.png");
  TFile *ffig2 = new TFile("FJet_deta.root", "RECREATE");
  h2->Write();
  ffig2->Close();


  auto h3 = d_matched.Histo2D({"FJet_dphi_vs_deta", "FJet_dphi_vs_deta", 100,
                               0, 3.14, 100, 0, 5},
                              "FJet_dphi", "FJet_deta");
  auto c3 = new TCanvas();
  h3->Draw("colz");
  c3->SaveAs("FJet_dphi_vs_deta.png");
  TFile *ffig3 = new TFile("FJet_dphi_vs_deta.root", "RECREATE");
  h3->Write();
  ffig3->Close();
  */
}
