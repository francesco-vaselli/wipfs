int counter() {

	TFile *f = TFile::Open("MElectrons_v7.root");
	ROOT::RDataFrame d("MElectrons", f);
	auto h = d.Histo1D("MElectron_ptRatio");
	auto n = h->GetEntries();
	
	return n;


}
