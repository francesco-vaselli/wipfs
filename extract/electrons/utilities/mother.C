
void mother() {
	
	ROOT::RDataFrame d("Events", "047F4368-97D4-1A4E-B896-23C6C72DD2BE.root");

	auto d_m = d
		    .Define("MGenPartIdx", "Electron_genPartIdx[Electron_genPartIdx >= 0]")
	            .Define("MGenPart_pdgId", "Take(GenPart_pdgId, MGenPartIdx)")
	            .Define("MGenElectronMask", "abs(MGenPart_pdgId) == 11")
          	    .Define("MGenElectronIdx", "MGenPartIdx[MGenElectronMask]")
		    .Define("GenPartMotherIdx", "GenPart_genPartIdxMother[GenPart_genPartIdxMother >= 0]")
		    .Define("GenPartMother_pdgId", "Take(GenPart_pdgId, GenPartMotherIdx)")
		    .Define("MGenElectronMother_pdgId", "Take(GenPartMother_pdgId, MGenElectronIdx)");

	auto h = d_m.Histo1D("GenPartMother_pdgId");
	auto h1 = d_m.Histo1D("MGenElectronMother_pdgId");

	auto c = new TCanvas();

//	h->Draw("PLC");
	h1->Draw("SAME PLC");

	c->SaveAs("mother.pdf");
	c->Close();
}
