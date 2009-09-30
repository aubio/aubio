// main.java

public class main {
  static {
    System.loadLibrary("aubiowrapper");
  }

  public static void main(String argv[]) {
    float freq = (float)440.;
    float midi = aubiowrapper.aubio_freqtomidi( (float)440.);
    if (midi != (float) 69.0) {
      throw new Error((String) "Error: aubio_freqtomidi(440.) != 69.", null);
    }
    System.out.print(freq);
    System.out.print(" Hz corresponds to midi note ");
    System.out.println(midi);
  }
}

