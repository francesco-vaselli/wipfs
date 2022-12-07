int NBitwiseDecoder() {

  // cut id: to be passed to the function. From 0 to 9. From cut we build the
  // 3-uple of bits representing quality level of the cut.

  int cut = 6;

  int bits[3] = {(cut * 3), (cut * 3) + 1, (cut * 3) + 2};

  // ints: bitmap encoding

  int ints = 605029412;

  // binary encoding 100100000100000000010000100100

  auto decoded = 0;

  for (int i = 0; i < 3; i++) {

    int num = pow(2, bits[i]);

    auto bAND = ints & num;

    if (bAND == num) {
      decoded += pow(2, i);
    } else {
      decoded = decoded;
    }
  }

  return decoded;
}
