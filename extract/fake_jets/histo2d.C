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

void histo2d() {
  // open ROOT data frame
  ROOT::RDataFrame d_matched("FJets", "FJets.root", {"FJet_dphi", "FJet_deta"});


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
  h3->Draw("LEGO2");
  // add axes names
  h3->GetXaxis()->SetTitle("dphi");
  h3->GetYaxis()->SetTitle("abs(deta)");
  c3->SaveAs("FJet_dphi_vs_deta.png");
  TFile *ffig3 = new TFile("FJet_dphi_vs_deta.root", "RECREATE");
  h3->Write();
  ffig3->Close();
}