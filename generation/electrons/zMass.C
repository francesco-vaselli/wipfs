auto InvariantMass(ROOT::VecOps::RVec<float> &pt,
                   ROOT::VecOps::RVec<float> &eta,
                   ROOT::VecOps::RVec<float> &phi) {

  auto m = 0.51099895000e-3; // GeV

  ROOT::Math::PtEtaPhiMVector p1(pt[0], eta[0], phi[0], m);
  ROOT::Math::PtEtaPhiMVector p2(pt[1], eta[1], phi[1], m);

  return (p1 + p2).mass();
}

void zMass() {

  ROOT::EnableImplicitMT();

  auto f = TFile::Open("gpfs/ddn/cms/user/cattafe/DYJets/EM1/230000/"
                       "8244ED99-0F95-9D4F-B393-22EBC589A46D_synth.root",
                       "r");

  auto d = ROOT::RDataFrame("Events", f);

  auto d_f = d.Filter("nElectron == 2")
                 .Filter("All(abs(Electron_eta) < 2.5)")
                 .Filter("All(Electron_pt > 20)")
                 .Filter("Sum(Electron_charge) == 0")
                 .Filter("All(Electron_ip3d < 0.015)")
                 .Define("Z_mass", InvariantMass,
                         {"Electron_pt", "Electron_eta", "Electron_phi"});

  auto h = d_f.Histo1D("Z_mass");
  h->DrawCopy();
  f->Close();
}