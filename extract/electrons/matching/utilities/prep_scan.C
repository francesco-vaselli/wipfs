
void prep_scan() {
   
	ROOT::RDataFrame d("Events", "047F4368-97D4-1A4E-B896-23C6C72DD2BE.root");
	
	std::string_view column = "Electron_sip3d";

	auto h = d.Histo1D({"", "", 50, 0, 700}, column);

	auto c = new TCanvas();

	h->Draw();

	c->SaveAs("scan.pdf");
	c->Close();
}
