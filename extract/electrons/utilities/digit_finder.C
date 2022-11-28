int digit_finder() {

  auto pdgId = -10631;
  auto flavour = 4;

  int num = abs(pdgId);
  cout << "num: " << num << " flav: " << flavour << endl;
  int digit;
  int res = 0;

  while (num != 0) {
    digit = num  % 10;
    cout << "digit: " << digit << endl;
    num = num / 10;
    cout << "num: " << num << endl;
    if (digit == flavour) {
        res = 1;
        break;
    }
  }
  
  return res;
}

