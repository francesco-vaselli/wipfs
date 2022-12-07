void f_DR_MElectron_plot() {
        
        auto f_MGenPart = 0.431386;
	auto f_MJRElectron = 0.463636;
        
        TGraph *g = new TGraph("f_DR_MElectron.txt");
        
        auto c = new TCanvas(); c->SetGrid();
        
        g->Draw("AP");
        g->SetMarkerStyle(20);
        
        g->SetTitle("RECO/GenPart DeltaR-based matching fraction");
        auto xaxis = g->GetXaxis();
        xaxis->SetTitle("#DeltaR Threshold");
        
        auto yaxis = g->GetYaxis();
        yaxis->SetTitle("Matching Fraction");
        yaxis->SetRangeUser(0., 0.11);

        auto pt1 = new TPaveText(.35, .25, .9, .4);
        pt1->SetOption("NDC BR");
        //pt1->SetFillStyle(1001);
        //pt1->SetLineColorAlpha(kBlack, 1.); 
        //pt1->SetLineWidth(10);
        pt1->SetTextFont(42);
        pt1->SetTextSize(.04);
        
        auto sMatch = Form("MRElectron + MJRElectron_{#DeltaR<0.2}: %.3f", f_MGenPart + f_MJRElectron);
        
        pt1->AddText(sMatch);
        pt1->Draw();
        
        auto pt2 = new TPaveText(.15, .75, .4, .93);
        pt2->SetOption("NDC NB");
        pt2->SetFillStyle(0);
        pt2->SetLineColorAlpha(0, 0.);
        pt2->SetTextFont(42);
        pt2->SetTextSize(.04);
        pt2->SetTextColor(kBlue);

        pt2->AddText("100% RECOElectrons");
        pt2->Draw();

        auto full_match = new TF1("full_match", "1-[0]", 0, 1.2);
	full_match->SetParameter(0, f_MGenPart + f_MJRElectron);

        full_match->Draw("SAME");
        full_match->SetLineStyle(9);
        full_match->SetLineColor(kBlue);
        full_match->SetLineWidth(2);

        c->Update();
        c->SaveAs("figures/f_DR_MElectron_vs_DR.pdf");
}
                                              
