package neural.project;

/**
 * Container of MNIST digits
 * @param no Digit number in the database
 * @param pixels 8-bit pixels
 * @param label Corresponding label
 * @author Ron.Coleman
 */
public record MDigit(int no, double[] pixels, int label) {
    public String toString() {
        Integer counter = 1;
        Integer c2 = 0;
        StringBuilder built = new StringBuilder();
        built.append("--- testing \n #"+no+" label: "+label+"\n"+"   0123456789012345678901234567\n0  ");
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (this.pixels()[i * 28 + j] == 0.0) {
                    built.append(".");
                } else {
                    built.append(Integer.toHexString((int) this.pixels()[i * 28 + j] / 16));
                }
            }
            built.append("\n");
            if(c2 != 2 || counter != 8){
                built.append(counter+"  ");
            }
            if(counter == 9){
                counter = 0;
                c2++;
            }else{
                counter++;
            }
        }
        return built.toString();
    }
}
