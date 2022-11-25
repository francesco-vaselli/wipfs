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
          if none present within 0.4, sets DR to 0.4
  */
  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> distances;
  distances.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    distances.emplace_back(0.5);
    float closest = 0.4;
    for (size_t j = 0; j < size_inner; j++) {
      Double_t deta = etae[i] - etaj[j];
      Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
      float dr = TMath::Sqrt(deta * deta + dphi * dphi);
      if (dr < closest) {
        closest = dr;
      }
    }
    if (closest < 0.4) {
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
          if none present within 0.4, sets DR to 0.4 and mass to 0 GeV
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> masses;
  masses.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    masses.emplace_back(0.0);
    float closest = 0.4;
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
          if none present within 0.4, sets DR to 0.4 and pt to 0 GeV
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> pts;
  pts.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    pts.emplace_back(0.0);
    float closest = 0.4;
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
          if none present within 0.4, sets DR to 0.4 and DeltaEta to 0.5
  */

  auto size_outer = etae.size();
  auto size_inner = etaj.size();
  ROOT::VecOps::RVec<float> detas;
  detas.reserve(size_outer);
  for (size_t i = 0; i < size_outer; i++) {
    detas.emplace_back(0.5);
    float closest = 0.4;
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
    dphis.emplace_back(0.5);
    float closest = 0.4;
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
    float closest = 0.4;
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

auto mother_genpart_pt(ROOT::VecOps::RVec<int> &mother_idx,
                       ROOT::VecOps::RVec<int> &genpart_pdgId,
                       ROOT::VecOps::RVec<float> &genpart_pt,
                       ROOT::VecOps::RVec<float> &ele_pt) {

  auto ele_size = ele_pt.size();
  auto genpart_size = genpart_pdgId.size();
  ROOT::VecOps::RVec<float> mother_pt;
  mother_pt.reserve(ele_size);

  for (size_t i = 0; i < ele_size; i++) {
    mother_pt.emplace_back(-10);
    if (genpart_pdgId[mother_idx[i]] isInteresting) {
      mother_pt[i] = genpart_pt[mother_idx[i]];
    }
  }
  return mother_pt;
}

int Interesting(int &pdgId) {
  int num = abs(pdgId);
  int digit;
  int res = 0;

  const int b = 5;
  const int c = 4;

  while (num != 0) {
    digit = num % 10;
    num = num / 10;
    if (digit == b || digit == c) {
      res = 1;
      break;
    }
  }

  return res;
}

auto var_mother_genpart_pt(ROOT::VecOps::RVec<int> &mother_idx,
                           ROOT::VecOps::RVec<int> &genpart_pdgId,
                           ROOT::VecOps::RVec<float> &genpart_pt,
                           ROOT::VecOps::RVec<float> &ele_pt) {

  auto ele_size = ele_pt.size();
  auto genpart_size = genpart_pdgId.size();
  ROOT::VecOps::RVec<float> mother_pt;
  mother_pt.reserve(ele_size);

  const int b = 5;
  const int c = 4;

  for (size_t i = 0; i < ele_size; i++) {
    mother_pt.emplace_back(-10);
    idx = mother_idx[i];
    while (idx >= 0) {
      pdgId = genpart_pdgId[idx];
      if (Interesting(pdgId) == 1) {
        mother_pt[i] = genpart_pt[idx];
        break;
      } else {
        idx = mother_idx[idx];
      }
    }
  }

  return mother_pt;
}

void electrons_extraction() {

  // ROOT::EnableImplicitMT();

  ROOT::RDataFrame d("Events", "047F4368-97D4-1A4E-B896-23C6C72DD2BE.root");

  auto d_matched =
      d.Define("MGenPartIdx", "Electron_genPartIdx[Electron_genPartIdx >= 0]")
          .Define("MGenPart_pdgId", "Take(GenPart_pdgId, MGenPartIdx)")
          .Define("MGenElectronMask", "abs(MGenPart_pdgId) == 11")
          .Define("MGenElectronIdx", "MGenPartIdx[MGenElectronMask]")
          .Define("MGenElectron_eta", "Take(GenPart_eta, MGenElectronIdx)")
          .Define("MGenElectron_phi", "Take(GenPart_phi, MGenElectronIdx)")
          .Define("MGenElectron_pt", "Take(GenPart_pt, MGenElectronIdx)")
          .Define("MGenElectron_pdgId", "Take(GenPart_pdgId, MGenElectronIdx)")
          .Define("MGenElectron_charge", charge, {"MGenElectron_pdgId"})
          .Define("MGenElectron_genPartIdxMother",
                  "GenPart_genPartIdxMother[MGenElectronMask]")
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi"})
          .Define("ClosestJet_deta", closest_jet_deta,
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi"})
          .Define("ClosestJet_dphi", closest_jet_dphi,
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi"})
          .Define("ClosestJet_pt", closest_jet_pt,
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_pt"})
          .Define("ClosestJet_mass", closest_jet_mass,
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_mass"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
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
                  {"GenJet_eta", "GenJet_phi", "MGenElectron_eta",
                   "MGenElectron_phi", "GenJet_partonFlavour"})
          .Define("Electron_genElectronIdx", genElectronIdx_maker,
                  {"Electron_genPartIdx", "GenPart_pdgId"})
          .Define("Electron_MGenElectronMask", "Electron_genElectronIdx >= 0")
          .Define("MElectron_charge",
                  "Electron_charge[Electron_MGenElectronMask]")
          .Define("MElectron_convVeto",
                  "Electron_convVeto[Electron_MGenElectronMask]")
          .Define("MElectron_cutBased",
                  "Electron_cutBased[Electron_MGenElectronMask]")
          .Define("MElectron_cutBased_Fall17_V1",
                  "Electron_cutBased_Fall17_V1[Electron_MGenElectronMask]")
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
          .Define("MElectron_eCorr",
                  "Electron_eCorr[Electron_MGenElectronMask]")
          .Define("MElectron_eInvMinusPInv",
                  "Electron_eInvMinusPInv[Electron_MGenElectronMask]")
          .Define("MElectron_energyErr",
                  "Electron_energyErr[Electron_MGenElectronMask]")
          .Define("MElectron_etaMinusGen",
                  "Electron_eta[Electron_MGenElectronMask] - MGenElectron_eta")
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
          .Define("MElectron_ptRatio",
                  "Electron_pt[Electron_MGenElectronMask] / MGenElectron_pt")
          .Define("MElectron_r9", "Electron_r9[Electron_MGenElectronMask]")
          .Define("MElectron_seedGain",
                  "Electron_seedGain[Electron_MGenElectronMask]")
          .Define("MElectron_sieie",
                  "Electron_sieie[Electron_MGenElectronMask]")
          .Define("MElectron_sip3d",
                  "Electron_sip3d[Electron_MGenElectronMask]")
          .Define("MElectron_tightCharge",
                  "Electron_tightCharge[Electron_MGenElectronMask]")
          .Define("MElectron_vidNestedWPBitmap",
                  "Electron_vidNestedWPBitmap[Electron_MGenElectronMask]")
          .Define("MElectron_vidNestedWPBitmap0",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 0;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap1",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 1;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap2",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 2;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap3",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 3;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap4",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 4;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap5",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 5;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap6",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 6;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap7",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 7;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap8",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 8;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmap9",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int cut = 9;
                    return BitwiseDecoder_3bit(ints, cut);
                  },
                  {"MElectron_vidNestedWPBitmap"})
          .Define("MElectron_vidNestedWPBitmapHEEP",
                  "Electron_vidNestedWPBitmapHEEP[Electron_MGenElectronMask]")
          .Define("MElectron_vidNestedWPBitmapHEEP0",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 0;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP1",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 1;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP2",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 2;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP3",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 3;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP4",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 4;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP5",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 5;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP6",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 6;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP7",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 7;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP8",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 8;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP9",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 9;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP10",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 10;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"MElectron_vidNestedWPBitmapHEEP"})
          .Define("MElectron_vidNestedWPBitmapHEEP11",
                  [](ROOT::VecOps::RVec<int> &ints) {
                    int bit = 11;
                    return BitwiseDecoder(ints, bit);
                  },
                  {"Electron_vidNestedWPBitmapHEEP"});

  auto h = d_matched.Histo1D("MGenElectron_genPartIdxMother");
  auto c = new TCanvas();
  h->Draw();
  c->SaveAs("extraction.pdf");
  c->Close();
}
