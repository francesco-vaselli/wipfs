
void ZMass() {

  ROOT::EnableImplicitMT();

  auto f = TFile::Open("/gpfs/ddn/cms/user/cattafe/TTJets/60000/"
                       "16ADF854-8C85-DB4F-84F0-339B292E3CBD_synth.root")

      auto d = ROOT::RDataFrame("Events", f);

  d.Filter("nElectrons >= 2")
      .Filter("Electron_pt[0] > 20")
      .Filter("Electron_pt[1] > 20")
      .Define("ZMass",
              "sqrt(2*Electron_pt[0]*Electron_pt[1]*(cosh(Electron_eta[0]-"
              "Electron_eta[1])-cos(Electron_phi[0]-Electron_phi[1])))");

  auto h = d.Histo1D("ZMass");
  h->DrawCopy();
}