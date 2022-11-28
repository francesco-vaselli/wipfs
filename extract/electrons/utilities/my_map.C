
auto my_atan(ROOT::VecOps::RVec<float> &x) {
	auto size = x.size();
	ROOT::VecOps::RVec<float> res;
	res.reserve(size);

	for (size_t i = 0; i < size; i++) {
	res.emplace_back(TMath::ATan(x[i]) * 1000);
	}
	return res;
}

void my_map() {

	ROOT::RDataFrame d("Events", "047F4368-97D4-1A4E-B896-23C6C72DD2BE.root");

	auto d_map = d.Define("tr_Electron_dz", my_atan, {"Electron_dz"});

	auto h = d_map.Histo1D({"", "", 50, -100, 100}, "tr_Electron_dz");
	auto h1 = d.Histo1D({"", "", 50, -100, 100}, "Electron_dz");

	auto c = new TCanvas();

	h->Draw("PLC");
	h1->Draw("SAME PLC");

	c->SaveAs("map.pdf");
	c->Close();
}
