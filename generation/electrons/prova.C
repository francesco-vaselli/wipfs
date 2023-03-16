
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

auto BitwiseDecoder_3bit(ROOT::VecOps::RVec<int> &ints, int &cut) {
  /* Utility function for performing bitwise decoding of
     Electron_vidNestedWPBitmap, which stores 3 bits per cut.
   */
  auto size_decoded = ints.size();
  ROOT::VecOps::RVec<int> decoded_cuts;
  decoded_cuts.reserve(size_decoded);
  ROOT::VecOps::RVec<int> bits{(cut * 3), (cut * 3) + 1, (cut * 3) + 2};
  auto size_bits = bits.size();
  for (size_t i = 0; i < size_decoded; i++) {
    int decoded = 0;
    for (size_t j = 0; j < size_bits; j++) {
      int num = pow(2, bits[j]);
      auto bAND = ints[i] & num;
      if (bAND == num) {
        decoded += pow(2, j);
      } else {
        decoded = decoded;
      }
    }
    decoded_cuts.emplace_back(decoded);
  }
  return decoded_cuts;
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

auto genElectronIdx_maker(ROOT::VecOps::RVec<int> &genpart_idx,
                          ROOT::VecOps::RVec<int> &genpart_pdgId) {

  auto size_electron = genpart_idx.size();
  ROOT::VecOps::RVec<int> electron_idx;
  electron_idx.reserve(size_electron);

  for (size_t i = 0; i < size_electron; i++) {
    if (genpart_idx[i] >= 0) {
      if (abs(genpart_pdgId[genpart_idx[i]]) == 11) {
        electron_idx.emplace_back(genpart_idx[i]);
      } else {
        electron_idx.emplace_back(-1);
      }
    } else {
      electron_idx.emplace_back(-1);
    }
  }

  return electron_idx;
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

auto extract(ROOT::RDataFrame &d) {

  ROOT::EnableImplicitMT();

  auto pre =
      d.Define("GenElectronMask", "abs(GenPart_pdgId) == 11")
          .Define("GenElectron_pt", "GenPart_pt[GenElectronMask]")
          .Define("GenElectron_eta", "GenPart_eta[GenElectronMask]")
          .Define("GenElectron_phi", "GenPart_phi[GenElectronMask]")
          .Define("GenMuonMask", "abs(GenPart_pdgId) == 13")
          .Define("GenMuon_pt", "GenPart_pt[GenMuonMask]")
          .Define("GenMuon_eta", "GenPart_eta[GenMuonMask]")
          .Define("GenMuon_phi", "GenPart_phi[GenMuonMask]")
          .Define("CleanGenJet_mask_ele", clean_genjet_mask,
                  {"GenJet_pt", "GenJet_eta", "GenJet_phi", "GenElectron_pt",
                   "GenElectron_eta", "GenElectron_phi"})
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
          .Define("MGenElectorMask", "abs(MGenPart_pdgId) == 11")
          .Define("MGenElectronIdx", "MGenPartIdx[MGenElectorMask]")
          .Define("MGenElectron_pt", "Take(GenPart_pt, MGenElectronIdx)")
          .Define("MGenElectron_eta", "Take(GenPart_eta, MGenElectronIdx)")
          .Define("MGenElectron_phi", "Take(GenPart_phi, MGenElectronIdx)")
          .Define("MGenElectron_pdgId", "Take(GenPart_pdgId, MGenElectronIdx)")
          .Define("MGenElectron_charge", charge, {"MGenElectron_pdgId"})
          .Define("MGenElectron_statusFlags",
                  "Take(GenPart_statusFlags, MGenElectronIdx)")
          .Define("MGenElectron_statusFlag0",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 0;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag1",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 1;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag2",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 2;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag3",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 3;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag4",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 4;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag5",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 5;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag6",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 6;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag7",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 7;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag8",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 8;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag9",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 9;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag10",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 10;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag11",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 11;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag12",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 12;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag13",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 13;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("MGenElectron_statusFlag14",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 14;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MGenElectron_statusFlags"})
          .Define("ClosestJet_dr", closest_jet_dr,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi"})
          .Define("ClosestJet_dphi", closest_jet_dphi,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi"})
          .Define("ClosestJet_deta", closest_jet_deta,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi"})
          .Define("ClosestJet_pt", closest_jet_pt,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_pt"})
          .Define("ClosestJet_mass", closest_jet_mass,
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_mass"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_partonFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_partonFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_partonFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_partonFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_partonFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_hadronFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_hadronFlavour"})
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
                  {"CleanGenJet_eta", "CleanGenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "CleanGenJet_hadronFlavour"})
          .Define("Electron_genElectronIdx", genElectronIdx_maker,
                  {"Electron_genPartIdx", "GenPart_pdgId"})
          .Define("Electron_MGenElectronMask", "Electron_genElectronIdx >= 0")
          .Define("MElectron_convVeto",
                  "Electron_convVeto[Electron_MGenElectronMask]")
          .Define("MElectron_deltaEtaSC",
                  "Electron_deltaEtaSC[Electron_MGenElectronMask]")
          .Define("MElectron_dr03EcalRecHitSumEt",
                  "Electron_dr03EcalRecHitSumEt[Electron_MGenElectronMask]")
          .Define(
              "MElectron_dr03HcalDepth1TowerSumEt",
              "Electron_dr03HcalDepth1TowerSumEt[Electron_MGenElectronMask]")
          .Define("MElectron_dr03TkSumPt",
                  "Electron_dr03TkSumPt[Electron_MGenElectronMask]")
          .Define("MElectron_dr03TkSumPtHEEP",
                  "Electron_dr03TkSumPtHEEP[Electron_MGenElectronMask]")
          .Define("MElectron_dxy", "Electron_dxy[Electron_MGenElectronMask]")
          .Define("MElectron_dxyErr",
                  "Electron_dxyErr[Electron_MGenElectronMask]")
          .Define("MElectron_dz", "Electron_dz[Electron_MGenElectronMask]")
          .Define("MElectron_dzErr",
                  "Electron_dzErr[Electron_MGenElectronMask]")
          .Define("MElectron_eInvMinusPInv",
                  "Electron_eInvMinusPInv[Electron_MGenElectronMask]")
          .Define("MElectron_energyErr",
                  "Electron_energyErr[Electron_MGenElectronMask]")
          .Define("MElectron_eta",
                  "Electron_eta[Electron_MGenElectronMask]")
          .Define("MElectron_hoe", "Electron_hoe[Electron_MGenElectronMask]")
          .Define("MElectron_ip3d", "Electron_ip3d[Electron_MGenElectronMask]")
          .Define("MElectron_isPFcand",
                  "Electron_isPFcand[Electron_MGenElectronMask]")
          .Define("MElectron_jetPtRelv2",
                  "Electron_jetPtRelv2[Electron_MGenElectronMask]")
          .Define("MElectron_jetRelIso",
                  "Electron_jetRelIso[Electron_MGenElectronMask]")
          .Define("MElectron_lostHits",
                  "Electron_lostHits[Electron_MGenElectronMask]")
          .Define("MElectron_miniPFRelIso_all",
                  "Electron_miniPFRelIso_all[Electron_MGenElectronMask]")
          .Define("MElectron_miniPFRelIso_chg",
                  "Electron_miniPFRelIso_chg[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1Iso",
                  "Electron_mvaFall17V1Iso[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1Iso_WP80",
                  "Electron_mvaFall17V1Iso_WP80[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1Iso_WP90",
                  "Electron_mvaFall17V1Iso_WP90[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1Iso_WPL",
                  "Electron_mvaFall17V1Iso_WPL[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1noIso",
                  "Electron_mvaFall17V1noIso[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1noIso_WP80",
                  "Electron_mvaFall17V1noIso_WP80[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1noIso_WP90",
                  "Electron_mvaFall17V1noIso_WP90[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V1noIso_WPL",
                  "Electron_mvaFall17V1noIso_WPL[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2Iso",
                  "Electron_mvaFall17V2Iso[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2Iso_WP80",
                  "Electron_mvaFall17V2Iso_WP80[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2Iso_WP90",
                  "Electron_mvaFall17V2Iso_WP90[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2Iso_WPL",
                  "Electron_mvaFall17V2Iso_WPL[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2noIso",
                  "Electron_mvaFall17V2noIso[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2noIso_WP80",
                  "Electron_mvaFall17V2noIso_WP80[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2noIso_WP90",
                  "Electron_mvaFall17V2noIso_WP90[Electron_MGenElectronMask]")
          .Define("MElectron_mvaFall17V2noIso_WPL",
                  "Electron_mvaFall17V2noIso_WPL[Electron_MGenElectronMask]")
          .Define("MElectron_mvaTTH",
                  "Electron_mvaTTH[Electron_MGenElectronMask]")
          .Define("MElectron_pfRelIso03_all",
                  "Electron_pfRelIso03_all[Electron_MGenElectronMask]")
          .Define("MElectron_pfRelIso03_chg",
                  "Electron_pfRelIso03_chg[Electron_MGenElectronMask]")
          .Define("MElectron_phi", "Electron_phi[Electron_MGenElectronMask]")
          .Define("MElectron_phiMinusGen", DeltaPhi,
                  {"MElectron_phi", "MGenElectron_phi"})
          .Define("MElectron_pt",
                  "Electron_pt[Electron_MGenElectronMask]")
          .Define("MElectron_r9", "Electron_r9[Electron_MGenElectronMask]")
          .Define("MElectron_seedGain",
                  "Electron_seedGain[Electron_MGenElectronMask]")
          .Define("MElectron_sieie",
                  "Electron_sieie[Electron_MGenElectronMask]")
          .Define("MElectron_sip3d",
                  "Electron_sip3d[Electron_MGenElectronMask]")
          .Define("MElectron_tightCharge",
                  "Electron_tightCharge[Electron_MGenElectronMask]");

  return matched;
}

void prova() {

  auto col = "Electron_pt";
  auto col2 = "MElectron_pt";

  ROOT::EnableImplicitMT();

  // auto f =
  //     TFile::Open("/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/"
  //                 "DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/"
  //                 "NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/"
  //                 "230000/8244ED99-0F95-9D4F-B393-22EBC589A46D.root",
  //                 "r");

  // auto dy = ROOT::RDataFrame("Events", f);

  // auto full_dy = extract(dy);


  // auto g = TFile::Open("/gpfs/ddn/cms/user/cattafe/DYJets/230000/"
  //                      "8244ED99-0F95-9D4F-B393-22EBC589A46D_synth.root",
  //                      "r");

  // auto synth_dy = ROOT::RDataFrame("Events", g);

  // auto h1 = full_dy.Histo1D({"", "", 50, 0, 10}, col2);
  // h1->Scale(1. / h1->Integral());

  // auto h2 = synth_dy.Histo1D({"", "", 50, 0, 10}, col);
  // h2->Scale(1. / h2->Integral());

  // auto c = new TCanvas("c", "c", 800, 600);
  // h1->Draw("PLC HIST");
  // h2->Draw("same PLC HIST");

  // c->SaveAs("dy.pdf");

  auto m =
      TFile::Open("/gpfs/ddn/cms/user/cattafe/TTJets/EM1/60000/16ADF854-8C85-DB4F-84F0-339B292E3CBD_synth.root", "r");

  auto synt_tt = ROOT::RDataFrame("Events", m);

  auto p = TFile::Open("/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/"
                       "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/"
                       "Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/"
                       "60000/16ADF854-8C85-DB4F-84F0-339B292E3CBD.root",
                       "r");

  auto tt = ROOT::RDataFrame("Events", p);
  auto h_t  = tt.Histo1D({"", "", 50, 0., 100}, col);
  h_t->Scale(1. / h_t->Integral());
  auto n = h_t->GetEntries();
  auto full_tt = extract(tt);
  auto n2 = full_tt.Histo1D({"", "", 50, 0., 100}, col2)->GetEntries();

  cout << n << " " << n2 << endl;

  auto h3 = full_tt.Histo1D({"", "", 50, 0., 100}, col2);


  h3->Scale(1. / h3->Integral());

  auto n3 = h3->GetEntries();
  cout << n3 << endl;

  auto h4 = synt_tt.Histo1D({"", "", 50, 0., 100}, col);
  h4->Scale(1. / h4->Integral());

  auto c1 = new TCanvas("c1", "c1", 800, 600);

  h3->Draw("HIST");
  h4->Draw("HIST same");
  h3->SetLineColor(kRed);
  h4->SetLineColor(kBlue);

  c1->SaveAs("tt_pt.pdf");
}
