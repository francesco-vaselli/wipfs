auto clean_genjet_mask(ROOT::VecOps::RVec<float> &jet_pt,
                       ROOT::VecOps::RVec<float> &jet_eta,
                       ROOT::VecOps::RVec<float> &jet_phi,
                       ROOT::VecOps::RVec<float> &lep_pt,
                       ROOT::VecOps::RVec<float> &lep_eta,
                       ROOT::VecOps::RVec<float> &lep_phi) {
  /* Mask to remove GenElectrons  and GenMuons from the GenJet collection.*/
  auto lep_size = lep_pt.size();
  auto jet_size = jet_pt.size();

  ROOT::VecOps::RVec<int> clean_jet_mask;
  clean_jet_mask.reserve(jet_size);

  for (size_t i = 0; i < jet_size; i++) {
    clean_jet_mask.push_back(1);
    for (size_t j = 0; j < lep_size; j++) {
      auto dpt = jet_pt[i] - lep_pt[j];
      auto deta = jet_eta[i] - lep_eta[j];
      auto dphi = TVector2::Phi_mpi_pi(jet_phi[i] - lep_phi[j]);
      auto dr = TMath::Sqrt(deta * deta + dphi * dphi);

      if ((dr <= 0.01) && ((dpt / lep_pt[j]) <= 0.001)) {
        clean_jet_mask[i] = 0;
      }
    }
  }
  return clean_jet_mask;
}

auto DeltaPhi(ROOT::VecOps::RVec<float> &Phi1,
              ROOT::VecOps::RVec<float> &Phi2) {
  /* Calculates the DeltaPhi between two RVecs
   */
  auto size = Phi1.size();
  ROOT::VecOps::RVec<float> dphis;
  dphis.reserve(size);
  for (size_t i = 0; i < size; i++) {
    Double_t dphi = TVector2::Phi_mpi_pi(Phi1[i] - Phi2[i]);
    dphis.emplace_back(dphi);
  }
  return dphis;
}

auto closest_jet_dr(ROOT::VecOps::RVec<float> &etaj,
                    ROOT::VecOps::RVec<float> &phij,
                    ROOT::VecOps::RVec<float> &etae,
                    ROOT::VecOps::RVec<float> &phie) {
  /* Calculates the DeltaR from the closest Jet object,
          if none present within 10, sets DR to 10
  */
  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> distances;
  distances.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    distances.emplace_back(10);
    float closest = 10;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
      }
    }
    if (closest < 10) {
      distances[i] = closest;
    }
  }
  return distances;
}

auto closest_jet_mass(ROOT::VecOps::RVec<float> &etaj,
                      ROOT::VecOps::RVec<float> &phij,
                      ROOT::VecOps::RVec<float> &etae,
                      ROOT::VecOps::RVec<float> &phie,
                      ROOT::VecOps::RVec<float> &massj) {

  /* Calculates the mass of the closest Jet object,
          if none present within 10, sets mass to 0 GeV
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> masses;
  masses.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    masses.emplace_back(0.0);
    float closest = 10;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        masses[i] = massj[j];
      }
    }
  }
  return masses;
}

auto closest_jet_pt(ROOT::VecOps::RVec<float> &etaj,
                    ROOT::VecOps::RVec<float> &phij,
                    ROOT::VecOps::RVec<float> &etae,
                    ROOT::VecOps::RVec<float> &phie,
                    ROOT::VecOps::RVec<float> &ptj) {

  /* Calculates the pt of the closest Jet object,
          if none present within 10, sets pt to 0 GeV
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> pts;
  pts.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    pts.emplace_back(0.0);
    float closest = 10;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        pts[i] = ptj[j];
      }
    }
  }
  return pts;
}

auto closest_jet_deta(ROOT::VecOps::RVec<float> &etaj,
                      ROOT::VecOps::RVec<float> &phij,
                      ROOT::VecOps::RVec<float> &etae,
                      ROOT::VecOps::RVec<float> &phie) {

  /* Calculates the DeltaEta of the closest Jet object,
          if none present within 10, sets DeltaEta to 0.5
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> detas;
  detas.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    detas.emplace_back(4);
    float closest = 10;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        detas[i] = deta;
      }
    }
  }
  return detas;
}

auto closest_jet_dphi(ROOT::VecOps::RVec<float> &etaj,
                      ROOT::VecOps::RVec<float> &phij,
                      ROOT::VecOps::RVec<float> &etae,
                      ROOT::VecOps::RVec<float> &phie) {

  /* Calculates the DeltaPhi of the closest Jet object,
          if none present within 0.4, sets DR to 0.4 and DeltaPhi to 0.5
  */
  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> dphis;
  dphis.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    dphis.emplace_back(4);
    float closest = 10;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        dphis[i] = dphi;
      }
    }
  }
  return dphis;
}

auto closest_jet_flavour_encoder(ROOT::VecOps::RVec<float> &etaj,
                                 ROOT::VecOps::RVec<float> &phij,
                                 ROOT::VecOps::RVec<float> &etae,
                                 ROOT::VecOps::RVec<float> &phie,
                                 ROOT::VecOps::RVec<int> &fj,
                                 ROOT::VecOps::RVec<int> &flavours) {

  /* General function to encode the hadron and parton flavour of the closest Jet
     object. To be used for flavour one-hot encoding for training.
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  auto n_flavours = flavours.size();
  ROOT::VecOps::RVec<int> fenc;
  fenc.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    fenc.emplace_back(0);
    float closest = 10;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
        for (size_t k = 0; k < n_flavours; k++) {
          if (abs(fj[j]) == flavours[k]) {
            fenc[i] = 1;
          }
        }
      }
    }
  }
  return fenc;
}

auto BitwiseDecoder(ROOT::VecOps::RVec<int> &ints, int &bit) {
  /* Utility function for performing bitwise decoding of
          GenPart_statusFlags conditioning variable
  */
  auto size = ints.size();
  ROOT::VecOps::RVec<float> bits;
  bits.reserve(size);
  int num = pow(2, (bit));
  for (size_t i = 0; i < size; i++) {
    Double_t bAND = ints[i] & num;
    if (bAND == num) {
      bits.emplace_back(1);
    } else {
      bits.emplace_back(0);
    }
  }
  return bits;
}

auto charge(ROOT::VecOps::RVec<int> &pdgId) {
  /* Assigns the correct charge to the electron according to its gen pdg id
   */
  auto size = pdgId.size();
  ROOT::VecOps::RVec<float> charge;
  charge.reserve(size);
  for (size_t i = 0; i < size; i++) {
    if (pdgId[i] == -11)
      charge.emplace_back(-1);
    else
      charge.emplace_back(+1);
  }
  return charge;
}

int isInteresting(int &pdgId) {
  /* Check if pdgId is compatible with c or b quarks
   */
  int num = abs(pdgId);
  if ((pdgId == 4) || (pdgId == 5)) {
    return 1;
  }
  return 0;
}

auto mother_genpart_pt(ROOT::VecOps::RVec<int> &mother_idx,
                       ROOT::VecOps::RVec<int> &genpart_pdgId,
                       ROOT::VecOps::RVec<float> &genpart_pt,
                       ROOT::VecOps::RVec<float> &ele_pt) {
  /* If GenPartMother of MGenElectron is b or c quark, computes its pt
   */
  auto ele_size = ele_pt.size();
  ROOT::VecOps::RVec<float> mother_pt;

  for (size_t i = 0; i < ele_size; i++) {
    mother_pt.emplace_back(0);
    int idx = mother_idx[i];
    int pdgId;
    while (idx >= 0) {
      pdgId = genpart_pdgId[idx];
      if (isInteresting(pdgId) == 1) {
        mother_pt[i] = genpart_pt[idx];
      }
      idx = mother_idx[idx];
    }
  }

  return mother_pt;
}

auto mother_genpart_pdgId(ROOT::VecOps::RVec<int> &mother_idx,
                          ROOT::VecOps::RVec<int> &genpart_pdgId,
                          ROOT::VecOps::RVec<int> &ele_pdgId) {
  /* If GenPartMother of MGenElectron is b or c quark, save its pdgId
   */
  auto ele_size = ele_pdgId.size();
  ROOT::VecOps::RVec<int> mother_pdgId;

  for (size_t i = 0; i < ele_size; i++) {
    mother_pdgId.emplace_back(0);
    int idx = mother_idx[i];
    int pdgId;
    while (idx >= 0) {
      pdgId = genpart_pdgId[idx];
      if (isInteresting(pdgId) == 1) {
        mother_pdgId[i] = genpart_pdgId[idx];
      }
      idx = mother_idx[idx];
    }
  }

  return mother_pdgId;
}

auto mother_genpart_deta(ROOT::VecOps::RVec<int> &mother_idx,
                         ROOT::VecOps::RVec<int> &genpart_pdgId,
                         ROOT::VecOps::RVec<float> &genpart_eta,
                         ROOT::VecOps::RVec<float> &ele_eta) {
  /* If GenPartMother of MGenElectron is b or c quark, computes deta between the
   * two particles
   */

  auto ele_size = ele_eta.size();
  ROOT::VecOps::RVec<float> mother_deta;

  for (size_t i = 0; i < ele_size; i++) {
    mother_deta.emplace_back(0.5);
    int idx = mother_idx[i];
    int pdgId;
    while (idx >= 0) {
      pdgId = genpart_pdgId[idx];
      if (isInteresting(pdgId) == 1) {
        mother_deta[i] = ele_eta[i] - genpart_eta[idx];
      }
      idx = mother_idx[idx];
    }
  }

  return mother_deta;
}

auto mother_genpart_dphi(ROOT::VecOps::RVec<int> &mother_idx,
                         ROOT::VecOps::RVec<int> &genpart_pdgId,
                         ROOT::VecOps::RVec<float> &genpart_phi,
                         ROOT::VecOps::RVec<float> &ele_phi) {
  /* If GenPartMother of MGenElectron is b or c quark, computes dphi between the
   * two particles
   */

  auto ele_size = ele_phi.size();
  ROOT::VecOps::RVec<float> mother_dphi;

  for (size_t i = 0; i < ele_size; i++) {
    mother_dphi.emplace_back(0.5);
    int idx = mother_idx[i];
    int pdgId;
    while (idx >= 0) {
      pdgId = genpart_pdgId[idx];
      if (isInteresting(pdgId) == 1) {
        mother_dphi[i] = TVector2::Phi_mpi_pi(ele_phi[i] - genpart_phi[idx]);
      }
      idx = mother_idx[idx];
    }
  }

  return mother_dphi;
}

void gens(std::string x) {

  ROOT::EnableImplicitMT();

  ROOT::RDataFrame d("Events", x);

  auto pre =
      d.Define("EGenElectronMask", "abs(GenPart_pdgId) == 11")
          .Define("EGenElectron_pt", "GenPart_pt[EGenElectronMask]")
          .Define("EGenElectron_eta", "GenPart_eta[EGenElectronMask]")
          .Define("EGenElectron_phi", "GenPart_phi[EGenElectronMask]")
          .Define("EGenElectron_pdgId", "GenPart_pdgId[EGenElectronMask]")
          .Define("EGenElectron_charge", charge, {"EGenElectron_pdgId"})
          .Define("GenMuonMask", "abs(GenPart_pdgId) == 13")
          .Define("GenMuon_pt", "GenPart_pt[GenMuonMask]")
          .Define("GenMuon_eta", "GenPart_eta[GenMuonMask]")
          .Define("GenMuon_phi", "GenPart_phi[GenMuonMask]")
          .Define("CleanGenJet_mask_ele", clean_genjet_mask,
                  {"GenJet_pt", "GenJet_eta", "GenJet_phi", "EGenElectron_pt",
                   "EGenElectron_eta", "EGenElectron_phi"})
          .Define("CleanGenJet_mask_muon", clean_genjet_mask,
                  {"GenJet_pt", "GenJet_eta", "GenJet_phi", "GenMuon_pt",
                   "GenMuon_eta", "GenMuon_phi"})
          .Define("CleanGenJetMask",
                  "CleanGenJet_mask_ele && CleanGenJet_mask_muon")
          .Define("CleanGenJet_pt", "GenJet_pt[CleanGenJetMask]")
          .Define("CleanGenJet_eta", "GenJet_eta[CleanGenJetMask]")
          .Define("CleanGenJet_phi", "GenJet_phi[CleanGenJetMask]")
          .Define("CleanGenJet_mass", "GenJet_mass[CleanGenJetMask]")
          .Define("CleanGenJet_hadronFlavour_uchar",
                  "GenJet_hadronFlavour[CleanGenJetMask]")
          .Define("CleanGenJet_hadronFlavour",
                  "static_cast<ROOT::VecOps::RVec<int>>(CleanGenJet_"
                  "hadronFlavour_uchar)")
          .Define("CleanGenJet_partonFlavour",
                  "GenJet_partonFlavour[CleanGenJetMask]");

  auto matched =
      pre.Define("MGenPartIdx", "Electron_genPartIdx[Electron_genPartIdx >= 0]")
          .Define("MGenPart_pdgId", "Take(GenPart_pdgId, MGenPartIdx)")
          .Define("MGenElectronMask", "abs(MGenPart_pdgId) == 11")
          .Define("MGenElectronIdx", "MGenPartIdx[MGenElectronMask]")
          .Define("GenElectron_pt", "Take(GenPart_pt, MGenElectronIdx)")
          .Define("GenElectron_eta", "Take(GenPart_eta, MGenElectronIdx)")
          .Define("GenElectron_phi", "Take(GenPart_phi, MGenElectronIdx)")
          .Define("GenElectron_pdgId", "Take(GenPart_pdgId, MGenElectronIdx)")
          .Define("GenElectron_charge", charge, {"GenElectron_pdgId"})
          .Define("GenPartMother_pdgId", mother_genpart_pdgId,
                  {"GenPart_genPartIdxMother", "GenPart_pdgId",
                   "GenElectron_pdgId"})
          .Define("GenPartMother_pt", mother_genpart_pt,
                  {"GenPart_genPartIdxMother", "GenPart_pdgId", "GenPart_pt",
                   "GenElectron_pt"})
          .Define("GenPartMother_deta", mother_genpart_deta,
                  {"GenPart_genPartIdxMother", "GenPart_pdgId", "GenPart_eta",
                   "GenElectron_eta"})
          .Define("GenPartMother_dphi", mother_genpart_dphi,
                  {"GenPart_genPartIdxMother", "GenPart_pdgId", "GenPart_phi",
                   "GenElectron_phi"})
          .Define("GenElectron_statusFlags",
                  "Take(GenPart_statusFlags, MGenElectronIdx)")
          .Define("GenElectron_statusFlag0",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 0;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag1",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 1;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag2",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 2;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag3",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 3;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag4",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 4;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag5",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 5;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag6",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 6;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag7",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 7;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag8",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 8;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag9",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 9;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag10",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 10;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag11",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 11;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag12",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 12;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag13",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 13;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("GenElectron_statusFlag14",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 14;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"GenElectron_statusFlags"})
          .Define("ClosestJet_dr", closest_jet_dr,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi"})
          .Define("ClosestJet_dphi", closest_jet_dphi,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi"})
          .Define("ClosestJet_deta", closest_jet_deta,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi"})
          .Define("ClosestJet_pt", closest_jet_pt,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_pt"})
          .Define("ClosestJet_mass", closest_jet_mass,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_mass"})
          .Define("ClosestJet_EncodedPartonFlavour_light",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{1, 2, 3};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_partonFlavour"})
          .Define("ClosestJet_EncodedPartonFlavour_gluon",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{21};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_partonFlavour"})
          .Define("ClosestJet_EncodedPartonFlavour_c",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{4};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_partonFlavour"})
          .Define("ClosestJet_EncodedPartonFlavour_b",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{5};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_partonFlavour"})
          .Define("ClosestJet_EncodedPartonFlavour_undefined",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{0};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_partonFlavour"})
          .Define("ClosestJet_EncodedHadronFlavour_b",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{5};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_hadronFlavour"})
          .Define("ClosestJet_EncodedHadronFlavour_c",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{4};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_hadronFlavour"})
          .Define("ClosestJet_EncodedHadronFlavour_light",
                  [](ROOT::VecOps::RVec<float> &etaj,
                     ROOT::VecOps::RVec<float> &phij,
                     ROOT::VecOps::RVec<float> &etae,
                     ROOT::VecOps::RVec<float> &phie,
                     ROOT::VecOps::RVec<int> &fj) {
                    ROOT::VecOps::RVec<int> flavours{0};
                    return closest_jet_flavour_encoder(etaj, phij, etae, phie,
                                                       fj, flavours);
                  },
                  {"CleanGenJet_eta", "CleanGenJet_phi", "GenElectron_eta",
                   "GenElectron_phi", "CleanGenJet_hadronFlavour"});

  vector<string> col_to_save = {"GenElectron_eta",
                                "GenElectron_phi",
                                "GenElectron_pt",
                                "GenElectron_charge",
                                "GenPartMother_pdgId",
                                "GenPartMother_pt",
                                "GenPartMother_deta",
                                "GenPartMother_dphi",
                                "GenElectron_statusFlag0",
                                "GenElectron_statusFlag1",
                                "GenElectron_statusFlag2",
                                "GenElectron_statusFlag3",
                                "GenElectron_statusFlag4",
                                "GenElectron_statusFlag5",
                                "GenElectron_statusFlag6",
                                "GenElectron_statusFlag7",
                                "GenElectron_statusFlag8",
                                "GenElectron_statusFlag9",
                                "GenElectron_statusFlag10",
                                "GenElectron_statusFlag11",
                                "GenElectron_statusFlag12",
                                "GenElectron_statusFlag13",
                                "GenElectron_statusFlag14",
                                "ClosestJet_dr",
                                "ClosestJet_dphi",
                                "ClosestJet_deta",
                                "ClosestJet_pt",
                                "ClosestJet_mass",
                                "ClosestJet_EncodedPartonFlavour_light",
                                "ClosestJet_EncodedPartonFlavour_gluon",
                                "ClosestJet_EncodedPartonFlavour_c",
                                "ClosestJet_EncodedPartonFlavour_b",
                                "ClosestJet_EncodedPartonFlavour_undefined",
                                "ClosestJet_EncodedHadronFlavour_b",
                                "ClosestJet_EncodedHadronFlavour_c",
                                "ClosestJet_EncodedHadronFlavour_light",
                                "Pileup_gpudensity",
                                "Pileup_nPU",
                                "Pileup_nTrueInt",
                                "Pileup_pudensity",
                                "Pileup_sumEOOT",
                                "Pileup_sumLOOT",
                                "event",
                                "run"};

  matched.Snapshot("Gens", "testGens.root", col_to_save);
}