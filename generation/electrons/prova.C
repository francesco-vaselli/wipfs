

void prova(col) {

  ROOT::EnableImplicitMT();

  auto f =
      TFile::Open("/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/"
                  "DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/"
                  "NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/"
                  "230000/8244ED99-0F95-9D4F-B393-22EBC589A46D.root",
                  "r");

  auto full_dy = ROOT::RDataFrame("Events", f);

  auto g =
      TFile::Open("~/16ADF854-8C85-DB4F-84F0-339B292E3CBD_synth.root", "r");

  auto synth_dy = ROOT::RDataFrame("Events", g);

  auto h1 = full_dy.Histo1D(col);
  auto h2 = synth_dy.Histo1D(col);

  c = new TCanvas("c", "c", 800, 600);
  h1->Draw();
  h2->Draw("same");

  c->SaveAs(col + ".pdf");
}