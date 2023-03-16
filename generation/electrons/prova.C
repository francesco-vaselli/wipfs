

void prova() {

  auto col = "Electron_sip3d";

  ROOT::EnableImplicitMT();

  auto f =
      TFile::Open("/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/"
                  "DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/"
                  "NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/"
                  "230000/8244ED99-0F95-9D4F-B393-22EBC589A46D.root",
                  "r");

  auto full_dy = ROOT::RDataFrame("Events", f);

  auto g = TFile::Open("/gpfs/ddn/cms/user/cattafe/DYJets/230000/"
                       "8244ED99-0F95-9D4F-B393-22EBC589A46D_synth.root",
                       "r");

  auto synth_dy = ROOT::RDataFrame("Events", g);

  auto h1 = full_dy.Histo1D({"", "", 50, 0, 10}, col);
  h1->Scale(1./h1->Integral());

  auto h2 = synth_dy.Histo1D({"", "", 50, 0, 10}, col);
  h2->Scale(1./h2->Integral());

  auto c = new TCanvas("c", "c", 800, 600);
  h1->Draw("PLC HIST");
  h2->Draw("same PLC HIST");

  c->SaveAs("dy.pdf");

  auto m =
      TFile::Open("~/16ADF854-8C85-DB4F-84F0-339B292E3CBD_synth.root", "r");

  auto synt_tt = ROOT::RDataFrame("Events", m);

  auto p = TFile::Open("/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/"
                       "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/"
                       "Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/"
                       "60000/16ADF854-8C85-DB4F-84F0-339B292E3CBD.root",
                       "r");

  auto full_tt = ROOT::RDataFrame("Events", p);

  auto h3 = full_tt.Histo1D({"", "", 50, 0, 10}, col);
  h3->Scale(1./h3->Integral());

  auto h4 = synt_tt.Histo1D({"", "", 50, 0, 10}, col);
  h4->Scale(1./h4->Integral());

  auto c1 = new TCanvas("c1", "c1", 800, 600);

  h4->Draw("PLC HIST");
  h3->Draw("PLC HIST same");

  c1->SaveAs("tt.pdf");
}
